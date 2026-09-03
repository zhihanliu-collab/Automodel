# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Export the human-readable agent subset used by the Delta-Engram recipe.

Example::

    python -m experiments.delta_engram.export_agent_dataset \
      --agent-jsonl /data/agents.jsonl \
      --cache-manifest /data/cache/manifest.json \
      --output-jsonl /data/export/agents.jsonl \
      --output-manifest /data/export/agents.manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from experiments.delta_engram.odoo_corpus import SOURCE_AGENTS, _agent_tasks, _sha256_file

logger = logging.getLogger(__name__)


def _export_agent_dataset(
    *,
    agent_jsonl: Path,
    cache_manifest: Path,
    output_jsonl: Path,
    output_manifest: Path,
    validation_fraction: float,
    seed: int,
) -> None:
    tasks = _agent_tasks(agent_jsonl, validation_fraction, seed)
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    records = {record["sample_id"]: record for record in cache["records"] if record["source"] == SOURCE_AGENTS}
    missing = sorted(task["sample_id"] for task in tasks if task["sample_id"] not in records)
    if missing:
        raise ValueError(f"Cache manifest is missing {len(missing)} agent samples, first={missing[0]}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    jsonl_temp = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    manifest_temp = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    with jsonl_temp.open("w", encoding="utf-8") as stream:
        for task in tasks:
            record = records[task["sample_id"]]
            if task["split"] != record["split"] or task["group"] != record["group"]:
                raise ValueError(f"Split/group mismatch for {task['sample_id']}")
            row = {
                **task,
                "rendered_tokens": record["rendered_tokens"],
                "shifted_tokens": record["length"],
                "supervised_tokens": record["supervised_tokens"],
            }
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(jsonl_temp, output_jsonl)

    split_counts = {split: sum(task["split"] == split for task in tasks) for split in ("train", "validation")}
    exported_manifest = {
        "format_version": 1,
        "description": "Human-readable agent trajectories exactly matching the Delta-Engram training cache.",
        "source_jsonl": str(agent_jsonl),
        "source_sha256": _sha256_file(agent_jsonl),
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": _sha256_file(cache_manifest),
        "output_jsonl": str(output_jsonl),
        "output_sha256": _sha256_file(output_jsonl),
        "num_samples": len(tasks),
        "split_counts": split_counts,
        "masked_outdated_accounting_turns": sum(task["masked_outdated_accounting_turns"] for task in tasks),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "transformations": [
            "Preserve Jian's messages, tools, and separate reasoning_content fields.",
            "Assign a deterministic bill-grouped 90/10 train/validation split.",
            "Set step_loss_mask=0 on assistant edit_bill_header calls carrying the stale accounting_date argument.",
            "Attach rendered, shifted, and supervised token counts from the Qwen3.8-Flash-Next 131072-token cache.",
        ],
    }
    manifest_temp.write_text(json.dumps(exported_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(manifest_temp, output_manifest)
    logger.info("Exported %d samples to %s", len(tasks), output_jsonl)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-jsonl", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def main() -> None:
    """Export agent trajectories and their training metadata from CLI arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    _export_agent_dataset(
        agent_jsonl=args.agent_jsonl,
        cache_manifest=args.cache_manifest,
        output_jsonl=args.output_jsonl,
        output_manifest=args.output_manifest,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
