# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Measure raw N-gram cardinality, Delta hash occupancy, and row visits."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    QWEN3_8_FLASH_NEXT_DELTA_LAYER_MULTIPLIERS,
    build_delta_ngram_layout,
)


TOKEN_BITS = 18
TOKEN_MASK = (1 << TOKEN_BITS) - 1
EOS_TOKEN_ID = 248044


def _ngram_keys(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = ids.astype(np.uint64, copy=False)
    previous = np.empty_like(current)
    previous[0] = EOS_TOKEN_ID
    previous[1:] = current[:-1]
    previous2 = np.empty_like(current)
    previous2[: min(2, len(previous2))] = EOS_TOKEN_ID
    if len(previous2) > 2:
        previous2[2:] = current[:-2]
    if len(current) > 1:
        invalid = previous == EOS_TOKEN_ID
        previous2[invalid] = EOS_TOKEN_ID
    return (previous << TOKEN_BITS) | current, (previous2 << (2 * TOKEN_BITS)) | (previous << TOKEN_BITS) | current


def _mixed(keys: np.ndarray, order: int) -> np.ndarray:
    current = keys & TOKEN_MASK
    previous = (keys >> TOKEN_BITS) & TOKEN_MASK
    multipliers = tuple(np.uint64(value) for value in QWEN3_8_FLASH_NEXT_DELTA_LAYER_MULTIPLIERS)
    with np.errstate(over="ignore"):
        value = (current * multipliers[0]) ^ (previous * multipliers[1])
        if order == 3:
            previous2 = (keys >> (2 * TOKEN_BITS)) & TOKEN_MASK
            value ^= previous2 * multipliers[2]
    return value


def analyze(cache_dir: Path, rows_per_head: int, chunk_tokens: int) -> dict[str, object]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = np.memmap(cache_dir / "input_ids.i32", mode="r", dtype=np.int32)
    key_paths = (cache_dir / "train_bigrams.u64.tmp", cache_dir / "train_trigrams.u64.tmp")
    for path in key_paths:
        if path.exists():
            path.unlink()
    train_tokens = 0
    with key_paths[0].open("wb") as bigram_stream, key_paths[1].open("wb") as trigram_stream:
        for index, record in enumerate(manifest["records"], start=1):
            if record["split"] != "train":
                continue
            start = int(record["offset"])
            end = start + int(record["length"])
            ids = np.asarray(inputs[start:end], dtype=np.uint64)
            bigrams, trigrams = _ngram_keys(ids)
            bigrams.tofile(bigram_stream)
            trigrams.tofile(trigram_stream)
            train_tokens += len(ids)
            if index % 100 == 0:
                print(f"encoded records={index}/{len(manifest['records'])} train_tokens={train_tokens}", flush=True)

    keys_by_order = {
        2: np.memmap(key_paths[0], mode="r", dtype=np.uint64),
        3: np.memmap(key_paths[1], mode="r", dtype=np.uint64),
    }
    unique = {order: int(len(np.unique(keys))) for order, keys in keys_by_order.items()}
    sizes, offsets, padded_rows = build_delta_ngram_layout(rows_per_head, 16)
    counts = np.zeros(padded_rows, dtype=np.uint64)
    heads: list[dict[str, object]] = []
    for order, head_start in ((2, 0), (3, 8)):
        keys = keys_by_order[order]
        order_counts = [np.zeros(sizes[head], dtype=np.uint64) for head in range(head_start, head_start + 8)]
        for start in range(0, len(keys), chunk_tokens):
            mixed = _mixed(np.asarray(keys[start : start + chunk_tokens]), order).view(np.int64)
            for local_head, head in enumerate(range(head_start, head_start + 8)):
                hashed = np.remainder(mixed, sizes[head])
                order_counts[local_head] += np.bincount(hashed, minlength=sizes[head]).astype(np.uint64)
            if start and start % (chunk_tokens * 10) == 0:
                print(f"hashed order={order} tokens={start}/{len(keys)}", flush=True)
        for local_head, head in enumerate(range(head_start, head_start + 8)):
            head_counts = order_counts[local_head]
            offset = offsets[head]
            counts[offset : offset + sizes[head]] = head_counts
            occupied = int(np.count_nonzero(head_counts))
            expected = sizes[head] * (1.0 - math.exp(-unique[order] / sizes[head]))
            heads.append(
                {
                    "head": head,
                    "order": order,
                    "rows": sizes[head],
                    "offset": offset,
                    "unique_raw_ngrams": unique[order],
                    "unique_hash_ids": occupied,
                    "expected_uniform_occupied": expected,
                    "actual_over_expected": occupied / max(expected, 1.0),
                    "raw_to_hash_collision_fraction": 1.0 - occupied / max(unique[order], 1),
                    "visit_count_max": int(head_counts.max()),
                    "visit_count_mean_occupied": float(head_counts.sum() / max(occupied, 1)),
                }
            )

    if counts.max() > np.iinfo(np.uint32).max:
        raise OverflowError(f"Row visit count exceeds uint32: {counts.max()}")
    counts_path = cache_dir / "delta_access_counts.u32"
    temp_counts = cache_dir / "delta_access_counts.u32.tmp"
    counts.astype(np.uint32).tofile(temp_counts)
    os.replace(temp_counts, counts_path)
    result = {
        "rows_per_head_nominal": rows_per_head,
        "padded_rows": padded_rows,
        "train_tokens": train_tokens,
        "unique_bigrams": unique[2],
        "unique_trigrams": unique[3],
        "access_counts_path": str(counts_path),
        "heads": heads,
    }
    result_path = cache_dir / "delta_hash_occupancy.json"
    temp_result = cache_dir / "delta_hash_occupancy.json.tmp"
    temp_result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_result, result_path)
    for path in key_paths:
        path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--rows-per-head", type=int, default=1_000_000)
    parser.add_argument("--chunk-tokens", type=int, default=4_000_000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cache_dir, args.rows_per_head, args.chunk_tokens), indent=2))


if __name__ == "__main__":
    main()
