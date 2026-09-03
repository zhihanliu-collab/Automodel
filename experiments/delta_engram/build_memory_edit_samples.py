# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Turn the Opus-5 memory-study transcript into agentic memory-edit training samples.

The study is one Claude Code session (compacted 11 times) in which the agent
reads the Odoo AP world and writes/edits notes under ``memory/``. The v1 corpus
kept only ``instruction -> note text`` pairs, which is neither agentic nor
reasoned. This script rebuilds, for every memory ``Write``/``Edit`` call, the
conversation the agent actually had inside its compaction segment (task or
summary, tool calls, tool results) and supervises only the target turn, whose
missing chain of thought is synthesized by a served model conditioned on the
context and on the exact edit.

Output: one JSON object per line with ``messages`` (system/user/assistant/tool
in the OpenAI chat shape used by the agent-trajectory corpus; earlier assistant
turns carry ``step_loss_mask: 0``), ``tools`` (schemas inferred from the
transcript), ``group`` (note file name), ``sample_id`` and provenance fields.

    python -m experiments.delta_engram.build_memory_edit_samples \
        --trace /mnt/data/zhihan/delta-engram/corpus/raw/opus5_memory_trace.jsonl \
        --out /mnt/data/zhihan/delta-engram/corpus/raw/memory_edit_samples_v2.jsonl \
        --endpoint b200-0:30000 --endpoint b200-1:30000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the accounts-payable apprentice of Acme Home Builders working inside Odoo through tools. "
    "Study the firm's records and keep grounded, source-cited notes under memory/ for your future self."
)
COT_SYSTEM = (
    "You write the private reasoning of an AI assistant that is studying an accounts-payable world in Odoo "
    "and keeps notes under memory/. You are shown the recent conversation and the assistant's next action. "
    "Write the assistant's internal reasoning, in the first person, that leads to exactly that action: which "
    "evidence it just read, why it is worth recording or correcting, and what it decided to write. Be concrete, "
    "cite the records it saw, 3 to 8 sentences, no headings, no restating the action verbatim."
)


@dataclass
class Turn:
    """One linearized conversation turn recovered from the transcript."""

    kind: str  # "user_text" | "summary" | "tool_results" | "assistant"
    text: str = ""
    tool_uses: list[dict[str, Any]] = field(default_factory=list)  # assistant: [{id, name, input}]
    tool_results: list[tuple[str, str]] = field(default_factory=list)  # tool_results: [(tool_use_id, text)]
    message_id: str | None = None


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict) and block.get("type") == "image":
                parts.append("[image]")
        return "\n".join(parts)
    return "" if content is None else str(content)


def linearize(records: list[dict[str, Any]]) -> list[Turn]:
    """Collapse Claude Code records into ordered turns, merging streamed assistant parts."""
    turns: list[Turn] = []
    for record in records:
        kind = record.get("type")
        message = record.get("message") or {}
        if kind == "user":
            content = message.get("content")
            if record.get("isCompactSummary"):
                turns.append(Turn("summary", text=_block_text(content)))
            elif isinstance(content, str):
                turns.append(Turn("user_text", text=content))
            else:
                results = []
                for block in content or []:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        results.append((str(block.get("tool_use_id")), _block_text(block.get("content"))))
                text_blocks = [b for b in (content or []) if isinstance(b, dict) and b.get("type") == "text"]
                if results:
                    if turns and turns[-1].kind == "tool_results":
                        turns[-1].tool_results.extend(results)
                    else:
                        turns.append(Turn("tool_results", tool_results=results))
                elif text_blocks:
                    turns.append(Turn("user_text", text=_block_text(content)))
        elif kind == "assistant":
            message_id = message.get("id")
            if turns and turns[-1].kind == "assistant" and turns[-1].message_id == message_id and message_id:
                target = turns[-1]
            else:
                target = Turn("assistant", message_id=message_id)
                turns.append(target)
            for block in message.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    target.text = (target.text + "\n" + str(block.get("text", ""))).strip()
                elif block.get("type") == "tool_use":
                    target.tool_uses.append(
                        {"id": str(block.get("id")), "name": str(block.get("name")), "input": block.get("input") or {}}
                    )
    return turns


def is_memory_edit(tool_use: dict[str, Any]) -> bool:
    return tool_use["name"] in ("Write", "Edit") and "/memory/" in str(tool_use["input"].get("file_path", ""))


def infer_tool_schemas(turns: list[Turn]) -> list[dict[str, Any]]:
    """Infer OpenAI-style tool schemas from every observed call (name, argument keys, JSON types)."""
    seen: dict[str, dict[str, Any]] = {}
    for turn in turns:
        for use in turn.tool_uses:
            entry = seen.setdefault(use["name"], {"props": {}, "counts": {}, "n": 0})
            entry["n"] += 1
            for key, value in (use["input"] or {}).items():
                json_type = {bool: "boolean", int: "integer", float: "number", list: "array", dict: "object"}.get(
                    type(value), "string"
                )
                entry["props"].setdefault(key, json_type)
                entry["counts"][key] = entry["counts"].get(key, 0) + 1
    tools = []
    for name in sorted(seen):
        entry = seen[name]
        required = sorted(k for k, c in entry["counts"].items() if c == entry["n"])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} tool as used by the Odoo AP apprentice.",
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": t} for k, t in sorted(entry["props"].items())},
                        "required": required,
                    },
                },
            }
        )
    return tools


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return text[:head] + f"\n... [{len(text) - cap} characters omitted] ...\n" + text[-tail:]


def _assistant_message(turn: Turn, *, supervised: bool) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.text,
        "reasoning_content": "",
        "tool_calls": [
            {"id": use["id"], "type": "function", "function": {"name": use["name"], "arguments": use["input"]}}
            for use in turn.tool_uses
        ],
    }
    if not supervised:
        message["step_loss_mask"] = 0
    return message


def build_context(
    segment: list[Turn], target_index: int, *, context_char_cap: int, tool_result_cap: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Messages preceding ``segment[target_index]``, trimmed from the front to the character cap.

    The segment's first turn (task prompt or compaction summary) is always kept;
    trimming removes whole assistant+tool_results pairs so no tool message is
    left without the call that produced it.
    """
    head = segment[0]
    body = segment[1:target_index]
    # Group body into units: an assistant turn followed by its tool_results turn.
    units: list[list[Turn]] = []
    for turn in body:
        if turn.kind == "tool_results" and units and units[-1][-1].kind == "assistant":
            units[-1].append(turn)
        else:
            units.append([turn])
    head_message = {"role": "user", "content": head.text}
    budget = context_char_cap - len(head.text)
    kept: list[list[Turn]] = []
    used = 0
    for unit in reversed(units):
        size = sum(
            len(t.text) + sum(len(json.dumps(u["input"], ensure_ascii=False)) for u in t.tool_uses)
            + sum(min(len(r[1]), tool_result_cap) for r in t.tool_results)
            for t in unit
        )
        if used + size > budget and kept:
            break
        kept.append(unit)
        used += size
    kept.reverse()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, head_message]
    for unit in kept:
        for turn in unit:
            if turn.kind == "assistant":
                messages.append(_assistant_message(turn, supervised=False))
            elif turn.kind == "tool_results":
                known_calls = {c["id"] for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])}
                for call_id, text in turn.tool_results:
                    if call_id not in known_calls:
                        continue  # the call was streamed into a record we could not attach; drop the orphan result
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": _truncate(text, tool_result_cap)})
            elif turn.kind in ("user_text", "summary"):
                messages.append({"role": "user", "content": _truncate(turn.text, tool_result_cap * 4)})
    stats = {"context_chars": used + len(head.text), "dropped_units": len(units) - len(kept), "kept_units": len(kept)}
    return messages, stats


def render_for_cot(messages: list[dict[str, Any]], target: dict[str, Any], *, cap: int) -> str:
    lines = []
    for message in messages[1:]:
        role = message["role"]
        if role == "assistant":
            calls = "; ".join(
                f"{c['function']['name']}({json.dumps(c['function']['arguments'], ensure_ascii=False)[:400]})"
                for c in message.get("tool_calls", [])
            )
            lines.append(f"ASSISTANT: {message.get('content','')}\n  calls: {calls}")
        elif role == "tool":
            lines.append(f"TOOL RESULT ({message['tool_call_id']}): {message['content'][:2500]}")
        else:
            lines.append(f"{role.upper()}: {message['content'][:6000]}")
    context = "\n\n".join(lines)
    if len(context) > cap:
        context = "[earlier context omitted]\n" + context[-cap:]
    action = json.dumps(
        {"text": target.get("content", ""), "tool_calls": target.get("tool_calls", [])}, ensure_ascii=False, indent=1
    )
    return f"{context}\n\n=== The assistant's NEXT action (do not repeat it, explain the reasoning behind it) ===\n{action}"


def synthesize_cot(endpoint: str, model: str, prompt: str, *, max_tokens: int, timeout: int) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"http://{endpoint}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    message = payload["choices"][0]["message"]
    text = (message.get("content") or "").strip()
    if not text and message.get("reasoning_content"):
        text = message["reasoning_content"].strip()
    if not text:
        raise RuntimeError("empty chain-of-thought completion")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", default=[], help="host:port of an OpenAI-compatible server")
    parser.add_argument("--model", default=None, help="served model id; default = first id from /v1/models")
    parser.add_argument("--context-char-cap", type=int, default=96000, help="~24k tokens of preceding context")
    parser.add_argument("--tool-result-cap", type=int, default=6000)
    parser.add_argument("--cot-context-cap", type=int, default=60000)
    parser.add_argument("--cot-max-tokens", type=int, default=700)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-cot", action="store_true", help="emit samples with empty reasoning_content")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = [json.loads(line) for line in args.trace.open(encoding="utf-8") if line.strip()]
    turns = linearize(records)
    tools = infer_tool_schemas(turns)
    # A segment is what the agent could see at once: it starts at the task
    # prompt or at a compaction summary. Other user texts (task notifications,
    # follow-ups) stay inside the current segment.
    segments: list[list[Turn]] = []
    for turn in turns:
        if not segments or turn.kind == "summary":
            segments.append([turn])
        else:
            segments[-1].append(turn)
    logger.info("turns=%d segments=%d tools=%d", len(turns), len(segments), len(tools))

    samples = []
    for segment_index, segment in enumerate(segments):
        for index, turn in enumerate(segment):
            if turn.kind != "assistant" or not any(is_memory_edit(u) for u in turn.tool_uses):
                continue
            messages, stats = build_context(
                segment, index, context_char_cap=args.context_char_cap, tool_result_cap=args.tool_result_cap
            )
            target = _assistant_message(turn, supervised=True)
            edits = [u for u in turn.tool_uses if is_memory_edit(u)]
            group = Path(str(edits[0]["input"].get("file_path"))).name
            samples.append(
                {
                    "sample_id": f"memory-s{segment_index:02d}-t{index:04d}",
                    "group": group,
                    "segment": segment_index,
                    "messages": messages + [target],
                    "tools": tools,
                    "edit_ops": [u["name"] for u in edits],
                    **stats,
                }
            )
    logger.info("memory-edit samples=%d", len(samples))

    if not args.no_cot:
        if not args.endpoint:
            parser.error("--endpoint is required unless --no-cot")
        model = args.model
        if model is None:
            with urllib.request.urlopen(f"http://{args.endpoint[0]}/v1/models", timeout=30) as response:
                model = json.load(response)["data"][0]["id"]

        def work(item: tuple[int, dict[str, Any]]) -> tuple[int, str, float]:
            index, sample = item
            prompt = render_for_cot(sample["messages"][:-1], sample["messages"][-1], cap=args.cot_context_cap)
            endpoint = args.endpoint[index % len(args.endpoint)]
            started = time.time()
            for attempt in range(3):
                try:
                    return index, synthesize_cot(endpoint, model, prompt, max_tokens=args.cot_max_tokens, timeout=600), time.time() - started
                except Exception as error:  # noqa: BLE001 - retry transient server errors
                    logger.warning("cot attempt %d failed for %s: %s", attempt + 1, sample["sample_id"], error)
                    time.sleep(5)
            raise RuntimeError(f"chain-of-thought synthesis failed for {sample['sample_id']}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done, (index, cot, seconds) in enumerate(pool.map(work, list(enumerate(samples))), start=1):
                samples[index]["messages"][-1]["reasoning_content"] = cot
                samples[index]["cot_model"] = model
                samples[index]["cot_seconds"] = round(seconds, 1)
                if done % 20 == 0 or done == len(samples):
                    logger.info("cot %d/%d", done, len(samples))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    context_chars = [s["context_chars"] for s in samples]
    cot_chars = [len(s["messages"][-1]["reasoning_content"]) for s in samples]
    summary = {
        "samples": len(samples),
        "segments": len(segments),
        "tools": len(tools),
        "context_chars_median": statistics.median(context_chars),
        "context_chars_max": max(context_chars),
        "samples_at_cap": sum(1 for s in samples if s["dropped_units"] > 0),
        "cot_chars_median": statistics.median(cot_chars) if cot_chars else 0,
        "groups": len({s["group"] for s in samples}),
        "edit_ops": {"Write": sum(s["edit_ops"].count("Write") for s in samples), "Edit": sum(s["edit_ops"].count("Edit") for s in samples)},
    }
    logger.info("summary %s", json.dumps(summary))
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
