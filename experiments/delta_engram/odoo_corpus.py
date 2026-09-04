# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Build and read a compact, source-aware Odoo continual-training corpus.

v2 sources: reference documents (handbook, tutorial; full NTP, repeated), per-bill chatter
threads (full NTP, no bill JSON), agentic memory edits with synthesized reasoning
(assistant-only, last turn), and agent trajectories (assistant-only).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


FORMAT_VERSION = 2
SOURCE_DOCS = "offline_docs"
SOURCE_MESSAGES = "offline_messages"
SOURCE_MEMORY = "memory_edits"
SOURCE_AGENTS = "agent_trajectories"
ALL_SOURCES = (SOURCE_DOCS, SOURCE_MESSAGES, SOURCE_MEMORY, SOURCE_AGENTS)
SOURCE_TO_ID = {source: index for index, source in enumerate(ALL_SOURCES)}
_BILL_ID_RE = re.compile(r"bill\s*\(id=(\d+)\)", re.IGNORECASE)
_WORKER_TOKENIZER: Any | None = None
_WORKER_MAX_CONTEXT = 131072


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text:
            self.parts.append(text)


def _html_to_text(value: Any) -> str:
    if not value:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(str(value))
    parser.close()
    return html.unescape("\n".join(parser.parts))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_validation_groups(groups: Iterable[str], fraction: float, seed: int) -> set[str]:
    unique = sorted(set(groups), key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    count = max(1, round(len(unique) * fraction))
    return set(unique[:count])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _doc_tasks(paths: list[Path], repeat: int) -> list[dict[str, Any]]:
    """Whole reference documents (handbook, tutorial) as full next-token-prediction text.

    Each document is emitted ``repeat`` times per epoch so two short files are
    not drowned by hundreds of long trajectories. Documents are train-only: the
    Delta-on vs Delta-off gate scores them directly during validation.
    """
    if repeat <= 0:
        raise ValueError(f"docs repeat must be positive, got {repeat}")
    tasks = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for index in range(repeat):
            tasks.append(
                {
                    "source": SOURCE_DOCS,
                    "split": "train",
                    "group": path.name,
                    "sample_id": f"doc-{path.stem}-{index:02d}",
                    "mode": "full_ntp",
                    "text": text,
                }
            )
    return tasks


def _render_message(message: dict[str, Any]) -> str:
    author = (message.get("author") or {}).get("name") or "unknown"
    recipients = ", ".join(r.get("name", "?") for r in (message.get("recipients") or [])) or "-"
    subject = message.get("subject") or ""
    attachments = ", ".join(a.get("name", "?") for a in (message.get("attachments") or []))
    body = _html_to_text(message.get("body_html", ""))
    head = f"[{message.get('date')}] {author} -> {recipients} ({message.get('message_type')})"
    if subject:
        head += f" | {subject}"
    if attachments:
        head += f" | attachments: {attachments}"
    return f"{head}\n{body}".rstrip()


def _message_tasks(path: Path, validation_fraction: float, seed: int) -> list[dict[str, Any]]:
    """Per-bill chatter threads (e-mails, comments, notifications) as plain text; no bill JSON."""
    rows = _read_jsonl(path)
    val_groups = _stable_validation_groups((str(row["bill_id"]) for row in rows), validation_fraction, seed)
    tasks = []
    for row in rows:
        messages = sorted(row.get("messages") or [], key=lambda m: str(m.get("date")))
        if not messages:
            continue
        header = row.get("header") or {}
        vendor = header.get("vendor")
        vendor_name = vendor.get("name") if isinstance(vendor, dict) else vendor
        text = (
            f"Acme Home Builders accounts payable. Chatter thread of vendor bill {header.get('ref') or row['bill_id']} "
            f"(vendor: {vendor_name}, invoice date: {header.get('invoice_date')}, state: {header.get('state')}).\n\n"
            + "\n\n".join(_render_message(message) for message in messages)
        )
        group = str(row["bill_id"])
        tasks.append(
            {
                "source": SOURCE_MESSAGES,
                "split": "validation" if group in val_groups else "train",
                "group": group,
                "sample_id": f"messages-{group}",
                "mode": "full_ntp",
                "text": text,
            }
        )
    return tasks


def _memory_tasks(path: Path, validation_fraction: float, seed: int) -> list[dict[str, Any]]:
    """Agentic memory-edit samples built by ``build_memory_edit_samples.py`` (with synthesized reasoning)."""
    rows = _read_jsonl(path)
    val_groups = _stable_validation_groups((row["group"] for row in rows), validation_fraction, seed)
    tasks = []
    for row in rows:
        target = row["messages"][-1]
        if target.get("role") != "assistant" or not target.get("tool_calls"):
            raise ValueError(f"{row['sample_id']}: last message must be the supervised assistant edit")
        if not target.get("reasoning_content"):
            raise ValueError(f"{row['sample_id']}: missing reasoning_content (run the builder with chain-of-thought)")
        tasks.append(
            {
                "source": SOURCE_MEMORY,
                "split": "validation" if row["group"] in val_groups else "train",
                "group": row["group"],
                "sample_id": row["sample_id"],
                "mode": "assistant_only",
                "messages": row["messages"],
                "tools": row.get("tools"),
            }
        )
    return tasks


def _agent_group(row: dict[str, Any], index: int) -> str:
    for message in row.get("messages", []):
        if message.get("role") != "user":
            continue
        match = _BILL_ID_RE.search(str(message.get("content", "")))
        if match:
            return match.group(1)
    raise ValueError(f"Agent trajectory {index} has no TASK bill id")


def _outdated_accounting_turn(message: dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if not str(function.get("name", "")).endswith("edit_bill_header"):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict) and "accounting_date" in arguments:
            return True
    return False


def _has_tool_call(message: dict[str, Any], suffix: str) -> bool:
    if message.get("role") != "assistant":
        return False
    return any(str((call.get("function") or {}).get("name", "")).endswith(suffix) for call in message.get("tool_calls") or [])


def _mask_post_bill_tail(messages: list[dict[str, Any]]) -> int:
    """Do not supervise what the trajectory does after its final ``post_bill``.

    957/960 ccsdk trajectories go post_bill -> finish_task (2 ever send a chatter message
    afterwards): they come from a world without the asynchronous vendor reply. The
    evaluation world sends one, and the base host answers it before finishing; every Delta
    variant trained on these trajectories stopped doing that (18_0120/18_0121/D01_1/D05_1,
    "communication with the vendor: not complete"). Masking the tail keeps the turns as
    context but stops teaching "post, then finish". Returns the number of masked turns.
    """
    last_post = max((i for i, m in enumerate(messages) if _has_tool_call(m, "post_bill")), default=None)
    if last_post is None:
        return 0
    masked = 0
    for message in messages[last_post + 1 :]:
        if message.get("role") == "assistant" and message.get("step_loss_mask", 1) != 0:
            message["step_loss_mask"] = 0
            masked += 1
    return masked


def _agent_tasks(path: Path, validation_fraction: float, seed: int) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    groups = [_agent_group(row, index) for index, row in enumerate(rows)]
    val_groups = _stable_validation_groups(groups, validation_fraction, seed)
    tasks = []
    for index, (row, group) in enumerate(zip(rows, groups, strict=True)):
        messages = copy.deepcopy(row["messages"])
        masked_turns = 0
        for message in messages:
            if _outdated_accounting_turn(message):
                # The August-2026 evaluator changed this convention to invoice
                # date. Keep the turn as context but do not reinforce its stale
                # action in the assistant-only objective.
                message["step_loss_mask"] = 0
                masked_turns += 1
        masked_tail = _mask_post_bill_tail(messages)
        tasks.append(
            {
                "source": SOURCE_AGENTS,
                "split": "validation" if group in val_groups else "train",
                "group": group,
                "sample_id": f"agent-{index:04d}",
                "mode": "assistant_only",
                "messages": messages,
                "tools": row.get("tools"),
                "masked_outdated_accounting_turns": masked_turns,
                "masked_post_bill_tail_turns": masked_tail,
            }
        )
    return tasks


def _init_worker(model_id: str, max_context: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_MAX_CONTEXT
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    _WORKER_MAX_CONTEXT = max_context


def _restore_mapping_arguments(original: list[dict[str, Any]], normalized: list[dict[str, Any]]) -> None:
    for source_message, target_message in zip(original, normalized, strict=True):
        source_calls = source_message.get("tool_calls")
        target_calls = target_message.get("tool_calls")
        if source_message.get("role") != "assistant" or not isinstance(source_calls, list):
            continue
        if not isinstance(target_calls, list):
            continue
        for source_call, target_call in zip(source_calls, target_calls, strict=True):
            arguments = source_call.get("function", {}).get("arguments")
            if isinstance(arguments, dict):
                target_call["function"]["arguments"] = dict(arguments)


def _tokenize_task(task: dict[str, Any]) -> dict[str, Any]:
    from nemo_automodel.components.datasets.llm.chat_dataset import _normalize_messages
    from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template

    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("Tokenizer worker was not initialized")
    eos = int(tokenizer.eos_token_id)
    if task["mode"] == "full_ntp":
        ids = tokenizer(task["text"], add_special_tokens=True, truncation=False)["input_ids"]
        if not ids or ids[-1] != eos:
            ids.append(eos)
        rendered_tokens = len(ids)
        if rendered_tokens > _WORKER_MAX_CONTEXT:
            return {"dropped": True, "rendered_tokens": rendered_tokens, **_task_metadata(task)}
        input_ids = np.asarray(ids[:-1], dtype=np.int32)
        labels = np.asarray(ids[1:], dtype=np.int32)
    else:
        messages = task["messages"]
        normalized = _normalize_messages(messages)
        _restore_mapping_arguments(messages, normalized)
        sample = format_chat_template(
            tokenizer,
            normalized,
            eos,
            eos,
            seq_length=None,
            padding="do_not_pad",
            truncation="do_not_truncate",
            tools=task.get("tools"),
            answer_only_loss_mask=True,
        )
        input_ids = np.asarray(sample["input_ids"], dtype=np.int32)
        labels = np.asarray(sample["labels"], dtype=np.int32)
        rendered_tokens = len(input_ids) + 1
        if rendered_tokens > _WORKER_MAX_CONTEXT:
            return {"dropped": True, "rendered_tokens": rendered_tokens, **_task_metadata(task)}
        if not np.any(labels != -100):
            raise ValueError(f"{task['sample_id']} has no supervised assistant tokens")
    return {
        "dropped": False,
        "rendered_tokens": rendered_tokens,
        "input_ids": input_ids,
        "labels": labels,
        "supervised_tokens": int(np.count_nonzero(labels != -100)),
        **_task_metadata(task),
    }


def _task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": task["source"],
        "split": task["split"],
        "group": task["group"],
        "sample_id": task["sample_id"],
        "masked_outdated_accounting_turns": int(task.get("masked_outdated_accounting_turns", 0)),
        "masked_post_bill_tail_turns": int(task.get("masked_post_bill_tail_turns", 0)),
    }


def _quantiles(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)
    return {
        str(q): ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
        for q in (0.5, 0.9, 0.95, 0.99, 1.0)
    }


def build_cache(args: argparse.Namespace) -> None:
    tasks = (
        _doc_tasks([args.handbook, args.tutorial], args.docs_repeat)
        + _message_tasks(args.bills_jsonl, args.validation_fraction, args.seed)
        + _memory_tasks(args.memory_samples_jsonl, args.validation_fraction, args.seed)
        + _agent_tasks(args.agent_jsonl, args.validation_fraction, args.seed)
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = [output_dir / "input_ids.i32", output_dir / "labels.i32", output_dir / "manifest.json"]
    if any(path.exists() for path in final_paths) and not args.overwrite:
        raise FileExistsError(f"Cache already exists under {output_dir}; pass --overwrite to replace known cache files")
    temp_ids = output_dir / "input_ids.i32.tmp"
    temp_labels = output_dir / "labels.i32.tmp"
    temp_manifest = output_dir / "manifest.json.tmp"
    for path in (temp_ids, temp_labels, temp_manifest):
        if path.exists():
            path.unlink()

    records: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    offset = 0
    context = mp.get_context("spawn")
    with (
        temp_ids.open("wb") as ids_stream,
        temp_labels.open("wb") as labels_stream,
        context.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.model_id, args.max_context),
        ) as pool,
    ):
        for completed, result in enumerate(pool.imap(_tokenize_task, tasks, chunksize=1), start=1):
            if result.pop("dropped"):
                dropped.append(result)
            else:
                input_ids = result.pop("input_ids")
                labels = result.pop("labels")
                if len(input_ids) != len(labels):
                    raise RuntimeError(f"Mismatched cached tensors for {result['sample_id']}")
                input_ids.tofile(ids_stream)
                labels.tofile(labels_stream)
                records.append({"offset": offset, "length": len(input_ids), **result})
                offset += len(input_ids)
            if completed % 25 == 0 or completed == len(tasks):
                print(f"tokenized {completed}/{len(tasks)} kept={len(records)} dropped={len(dropped)}", flush=True)

    summary: dict[str, Any] = {}
    for source in ALL_SOURCES:
        summary[source] = {}
        for split in ("train", "validation"):
            selected = [record for record in records if record["source"] == source and record["split"] == split]
            lengths = [record["rendered_tokens"] for record in selected]
            summary[source][split] = {
                "samples": len(selected),
                "rendered_tokens": sum(lengths),
                "supervised_tokens": sum(record["supervised_tokens"] for record in selected),
                "length_quantiles": _quantiles(lengths) if lengths else {},
                "groups": len({record["group"] for record in selected}),
            }
    manifest = {
        "format_version": FORMAT_VERSION,
        "model_id": args.model_id,
        "max_context": args.max_context,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "num_samples": len(records),
        "num_tokens_shifted": offset,
        "masked_outdated_accounting_turns": sum(
            record["masked_outdated_accounting_turns"] for record in records
        ),
        "masked_post_bill_tail_turns": sum(record.get("masked_post_bill_tail_turns", 0) for record in records),
        "dropped": dropped,
        "summary": summary,
        "docs_repeat": args.docs_repeat,
        "inputs": {
            "handbook": {"path": str(args.handbook), "sha256": _sha256_file(args.handbook)},
            "tutorial": {"path": str(args.tutorial), "sha256": _sha256_file(args.tutorial)},
            "bills_jsonl": {"path": str(args.bills_jsonl), "sha256": _sha256_file(args.bills_jsonl)},
            "memory_samples_jsonl": {
                "path": str(args.memory_samples_jsonl),
                "sha256": _sha256_file(args.memory_samples_jsonl),
            },
            "agent_jsonl": {"path": str(args.agent_jsonl), "sha256": _sha256_file(args.agent_jsonl)},
        },
        "records": records,
    }
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_ids, final_paths[0])
    os.replace(temp_labels, final_paths[1])
    os.replace(temp_manifest, final_paths[2])
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


@dataclass
class OdooCorpusDatasetConfig:
    """Construction-time configuration for a filtered pretokenized cache."""

    cache_dir: str
    split: str
    sources: list[str] | None = None
    # Keep one record per group (e.g. one copy of each repeated document) for gate loaders.
    unique_groups: bool = False

    def build(self) -> "OdooCorpusDataset":
        return OdooCorpusDataset(
            cache_dir=self.cache_dir, split=self.split, sources=self.sources, unique_groups=self.unique_groups
        )


class OdooCorpusDataset(Dataset):
    """Memory-mapped variable-length input/label pairs with source filtering."""

    def __init__(
        self, *, cache_dir: str, split: str, sources: list[str] | None = None, unique_groups: bool = False
    ) -> None:
        root = Path(cache_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Odoo cache format_version={manifest.get('format_version')}, expected {FORMAT_VERSION}"
            )
        requested = set(sources or ALL_SOURCES)
        unknown = requested - set(ALL_SOURCES)
        if unknown:
            raise ValueError(f"Unknown Odoo corpus source(s): {sorted(unknown)}")
        self.records = [
            record
            for record in manifest["records"]
            if record["split"] == split and record["source"] in requested
        ]
        if unique_groups:
            seen: set[str] = set()
            deduplicated = []
            for record in self.records:
                if record["group"] in seen:
                    continue
                seen.add(record["group"])
                deduplicated.append(record)
            self.records = deduplicated
        if not self.records:
            raise ValueError(f"No cached records for split={split!r}, sources={sorted(requested)}")
        self.lengths = [int(record["length"]) for record in self.records]
        self.input_ids = np.memmap(root / "input_ids.i32", mode="r", dtype=np.int32)
        self.labels = np.memmap(root / "labels.i32", mode="r", dtype=np.int32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        start = int(record["offset"])
        end = start + int(record["length"])
        input_ids = torch.from_numpy(np.asarray(self.input_ids[start:end], dtype=np.int64))
        labels = torch.from_numpy(np.asarray(self.labels[start:end], dtype=np.int64))
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            # ``default_collater`` concatenates tensor-valued fields along dim 0,
            # so metadata must keep an explicit per-example batch dimension.
            "source_id": torch.tensor([SOURCE_TO_ID[record["source"]]], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handbook", type=Path, required=True, help="COMPANY-HANDBOOK.md")
    parser.add_argument("--tutorial", type=Path, required=True, help="ODOO-MCP-TUTORIAL.md")
    parser.add_argument("--docs-repeat", type=int, default=16, help="copies of each document per epoch")
    parser.add_argument("--bills-jsonl", type=Path, required=True, help="offline export; only its chatter messages are used")
    parser.add_argument("--memory-samples-jsonl", type=Path, required=True, help="output of build_memory_edit_samples.py")
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
