#!/usr/bin/env python3
"""Build an SGLang-servable model directory for a Delta-Engram checkpoint.

A training checkpoint holds only the trainable Delta state (32 DCP safetensors
shards: the Delta n-gram table and the Delta PLE key/value projections). SGLang
needs one Hugging Face style directory, so this script:

1. reassembles the sharded Delta tensors with the DCP ``saved_offsets``;
2. copies the frozen reader parts the Delta PLE shares with the base PLE
   (norm_key / norm_query / norm_conv / conv1d), exactly as
   ``Qwen3_8_FlashNextPLELayer.copy_reader_from`` did at training init;
3. writes the Delta hash layout as buffers (layer_multipliers,
   ngram_heads_vocab_sizes, ngram_heads_offsets), mirroring the base PLE's
   checkpoint contract, and splits the table into ``split_ngram_parts`` shards
   the SGLang loader already understands;
4. symlinks every base snapshot file and writes a patched ``config.json``
   (``delta_engram_enabled``) plus a merged ``model.safetensors.index.json``.

It also prints integrity statistics (table nonzero fraction, K/V relative
change) to compare against the numbers recorded in the training handoff.

Run inside the SGLang container (torch + safetensors), CPU only::

    python export_delta_serving_dir.py \
        --ckpt-model-dir /mnt/data/zhihan/delta-engram/checkpoints/odoo-delta-formal-4ep-v5/epoch_0_step_329/model \
        --base-snapshot /mnt/data/zhihan/hf_cache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/<rev> \
        --out /mnt/data/zhihan/delta-engram/serving/s329
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Mirrors nemo_automodel/components/models/qwen3_8_flash_next/engram.py.
DELTA_LAYER_MULTIPLIERS = (
    6364136223846793005,
    1442695040888963407,
    3202034522624059733,
)
READER_COPIED_FROM_BASE = ("norm_key", "norm_query", "norm_conv", "conv1d")
DELTA_FILE = "delta_ple.safetensors"
LORA_FILE = "delta_lora_merged.safetensors"  # merged W + scale*B@A tensors, fed last by the patched loader
# safetensors header dtype strings -> torch dtypes
_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
    "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn,
}


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    limit = int(math.isqrt(value))
    for divisor in range(3, limit + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _next_prime_at_least(value: int) -> int:
    candidate = max(int(value), 2)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def build_delta_ngram_layout(rows_per_head: int, ngram_heads: int, alignment: int):
    """Port of engram.build_delta_ngram_layout (the training-side layout)."""
    sizes = []
    candidate = rows_per_head
    for _ in range(ngram_heads):
        prime = _next_prime_at_least(candidate)
        sizes.append(prime)
        candidate = prime + 1
    offsets = tuple(sum(sizes[:head]) for head in range(ngram_heads))
    unpadded = sum(sizes)
    padded = ((unpadded + alignment - 1) // alignment) * alignment
    return tuple(sizes), offsets, padded


def sglang_layout(rows_per_head: int, ngram_heads: int, alignment: int):
    """What the (patched) SGLang Qwen4ExpNGramEmbedding derives at construction:
    head k gets the (k+1)-th prime strictly after rows_per_head-1."""
    sizes = []
    prime = rows_per_head - 1
    for _ in range(ngram_heads):
        prime = _next_prime_at_least(prime + 1)
        sizes.append(prime)
    offsets = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    padded = ((total + alignment - 1) // alignment) * alignment
    return tuple(sizes), tuple(offsets), padded


def reassemble_delta(model_dir: str, delta_prefix: str, match=None) -> dict[str, torch.Tensor]:
    """Reassemble row-sharded DCP tensors whose key starts with ``delta_prefix`` (or satisfies ``match``)."""
    match = match or (lambda key: key.startswith(delta_prefix))
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        sys.exit(f"no safetensors shards under {model_dir}")
    pieces = defaultdict(list)  # key -> [(file, offsets, shape, dtype)]
    for path in files:
        with safe_open(path, framework="pt") as handle:
            meta = handle.metadata() or {}
            info = json.loads(meta.get("DCP_SHARDING_INFO", "{}"))
            for key in handle.keys():
                if not match(key):
                    continue
                slice_ = handle.get_slice(key)
                pieces[key].append(
                    (path, tuple(info[key]["saved_offsets"]), tuple(slice_.get_shape()), slice_.get_dtype())
                )
    if not pieces:
        sys.exit(f"no {delta_prefix}* tensors in {model_dir}")
    full: dict[str, torch.Tensor] = {}
    for key, parts in sorted(pieces.items()):
        for _, offsets, shape, _ in parts:
            if any(o != 0 for o in offsets[1:]):
                sys.exit(f"{key}: expected row sharding only, got offsets {offsets}")
        rows = max(o[0] + s[0] for _, o, s, _ in parts)
        tail = parts[0][2][1:]
        dtype = _SAFETENSORS_DTYPES[str(parts[0][3]).upper()]
        out = torch.empty((rows, *tail), dtype=dtype)
        covered = torch.zeros(rows, dtype=torch.bool)
        for path, offsets, shape, _ in parts:
            with safe_open(path, framework="pt") as handle:
                tensor = handle.get_tensor(key)
            start = offsets[0]
            if tuple(tensor.shape) != shape or tensor.shape[1:] != tail:
                sys.exit(f"{key}: shard shape {tuple(tensor.shape)} disagrees with {shape}/{tail}")
            if covered[start : start + tensor.shape[0]].any():
                sys.exit(f"{key}: overlapping shard at row {start}")
            out[start : start + tensor.shape[0]] = tensor
            covered[start : start + tensor.shape[0]] = True
        if not bool(covered.all()):
            sys.exit(f"{key}: {int((~covered).sum())} rows never written")
        full[key] = out
        print(f"[reassemble] {key} -> {tuple(out.shape)} {out.dtype} from {len(parts)} shards")
    return full


def load_base_tensor(base_dir: str, weight_map: dict, key: str) -> torch.Tensor:
    path = os.path.join(base_dir, weight_map[key])
    with safe_open(path, framework="pt") as handle:
        return handle.get_tensor(key)


def tensor_stats(t: torch.Tensor) -> dict:
    f = t.float()
    return {
        "elements": t.numel(),
        "rms": float(f.pow(2).mean().sqrt()),
        "max_abs": float(f.abs().max()),
        "nonzero_fraction": float((f != 0).float().mean()),
    }


def rel_change(new: torch.Tensor, old: torch.Tensor) -> float:
    return float((new.float() - old.float()).norm() / old.float().norm().clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-model-dir", required=True)
    ap.add_argument("--base-snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-head", type=int, default=1_000_000,
                    help="delta_ngram_vocab_size_per_head used in training (hashed table)")
    ap.add_argument("--table", choices=("hashed", "exact"), default="hashed",
                    help="Delta table kind the checkpoint was trained with")
    ap.add_argument("--keys", default=None, help="exact table: the delta_exact_keys_train.pt used in training")
    ap.add_argument("--alpha", type=float, default=1.0, help="delta_alpha the checkpoint was trained with")
    ap.add_argument("--lora-dim", type=int, default=None,
                    help="Delta+LoRA checkpoints: LoRA rank (peft.dim); with --lora-alpha merges W + alpha/dim * B@A")
    ap.add_argument("--lora-alpha", type=float, default=None, help="peft.alpha of the checkpoint")
    ap.add_argument("--no-delta", action="store_true",
                    help="LoRA-only control checkpoints: no Delta tensors in the ckpt; export base + merged LoRA with the Delta branch disabled")
    args = ap.parse_args()
    if args.table == "exact" and not args.keys:
        ap.error("--table exact requires --keys")
    if (args.lora_dim is None) != (args.lora_alpha is None):
        ap.error("--lora-dim and --lora-alpha go together")
    if args.no_delta and args.lora_dim is None:
        ap.error("--no-delta only makes sense with --lora-dim/--lora-alpha")

    base = os.path.realpath(args.base_snapshot)
    with open(os.path.join(base, "config.json")) as fh:
        config = json.load(fh)
    text = config["text_config"]
    ple_layer_ids = list(text["ple_layer_ids"])
    if len(ple_layer_ids) != 1:
        sys.exit(f"expected exactly one PLE layer, got {ple_layer_ids}")
    layer = int(ple_layer_ids[0]) - 1  # decoder index (ple_layer_ids are 1-based "layer 2")
    heads = (int(text["ngram_size"]) - 1) * int(text["heads_per_ngram"])
    alignment = int(text["make_ngram_vocab_size_divisible_by"])
    parts = int(text["split_ngram_parts"])
    ple_prefix = f"model.language_model.layers.{layer}.ple."
    delta_prefix = f"model.language_model.layers.{layer}.delta_ple."

    if args.table == "exact":
        keys = torch.load(args.keys, map_location="cpu", weights_only=True)
        n_bi, n_tri = int(keys["bigram"].numel()), int(keys["trigram"].numel())
        padded = ((1 + n_bi + n_tri + alignment - 1) // alignment) * alignment
        sizes, offsets = (n_bi, n_tri), (1, 1 + n_bi)
        print(f"[layout] exact table: bigrams={n_bi} trigrams={n_tri} padded_rows={padded} shards={parts}")
    else:
        sizes, offsets, padded = build_delta_ngram_layout(args.rows_per_head, heads, alignment)
        s_sizes, s_offsets, s_padded = sglang_layout(args.rows_per_head, heads, alignment)
        if (sizes, offsets, padded) != (s_sizes, s_offsets, s_padded):
            sys.exit(f"layout disagreement training={sizes, offsets, padded} sglang={s_sizes, s_offsets, s_padded}")
        print(f"[layout] heads={heads} primes {sizes[0]}..{sizes[-1]} padded_rows={padded} shards={parts}")

    with open(os.path.join(base, "model.safetensors.index.json")) as fh:
        index = json.load(fh)
    weight_map = dict(index["weight_map"])

    out_tensors: dict[str, torch.Tensor] = {}
    stats: dict = {"delta": "disabled (--no-delta)"}
    if not args.no_delta:
        delta = reassemble_delta(args.ckpt_model_dir, delta_prefix)
        table_key = delta_prefix + "ple_embedding.ngram_embedding.weight"
        key_key = delta_prefix + "key_proj.weight"
        value_key = delta_prefix + "value_proj.weight"
        for needed in (table_key, key_key, value_key):
            if needed not in delta:
                sys.exit(f"missing {needed}; have {sorted(delta)}")
        table = delta[table_key]
        if table.shape[0] != padded:
            sys.exit(f"table rows {table.shape[0]} != padded layout rows {padded}")

        out_tensors[key_key] = delta[key_key].contiguous()
        out_tensors[value_key] = delta[value_key].contiguous()
        for name in READER_COPIED_FROM_BASE:
            base_key = f"{ple_prefix}{name}.weight"
            out_tensors[f"{delta_prefix}{name}.weight"] = load_base_tensor(base, weight_map, base_key).contiguous()
        if args.table == "hashed":
            out_tensors[delta_prefix + "ple_embedding.layer_multipliers"] = torch.tensor(DELTA_LAYER_MULTIPLIERS, dtype=torch.long)
            out_tensors[delta_prefix + "ple_embedding.ngram_heads_vocab_sizes"] = torch.tensor(sizes, dtype=torch.long)
            out_tensors[delta_prefix + "ple_embedding.ngram_heads_offsets"] = torch.tensor(offsets, dtype=torch.long)
        shard_size = (padded + parts - 1) // parts
        for i in range(parts):
            chunk = table[i * shard_size : (i + 1) * shard_size]
            if chunk.shape[0] == 0:
                sys.exit(f"shard {i} would be empty (rows={padded}, parts={parts})")
            out_tensors[f"{delta_prefix}ple_embedding.ngram_embedding.shard_{i}.weight"] = chunk.clone()

        # Integrity numbers to compare with the training handoff.
        stats = {
            "table": tensor_stats(table),
            "key_proj_rel_change_vs_base": rel_change(delta[key_key], load_base_tensor(base, weight_map, f"{ple_prefix}key_proj.weight")),
            "value_proj_rel_change_vs_base": rel_change(delta[value_key], load_base_tensor(base, weight_map, f"{ple_prefix}value_proj.weight")),
            "key_proj_shape": list(delta[key_key].shape),
            "value_proj_shape": list(delta[value_key].shape),
            "trainable_parameters": int(table.numel() + delta[key_key].numel() + delta[value_key].numel()),
        }
    print("[stats]", json.dumps(stats, indent=1))

    # Delta+LoRA checkpoints: merge every lora_A/lora_B pair into its base tensor. The result is
    # written to LORA_FILE (kept out of the weight map so the default loader ignores it) and the
    # patched sglang model feeds it last in load_weights (config.delta_lora_merged_path).
    merged: dict[str, torch.Tensor] = {}
    lora_stats: dict = {}
    if args.lora_dim is not None:
        lora = reassemble_delta(args.ckpt_model_dir, "", match=lambda k: ".lora_A." in k or ".lora_B." in k)
        scale = float(args.lora_alpha) / float(args.lora_dim)
        modules = sorted({k.rsplit(".lora_", 1)[0] for k in lora})
        changes = []
        for module in modules:
            a_key, b_key = f"{module}.lora_A.weight", f"{module}.lora_B.weight"
            if a_key not in lora or b_key not in lora:
                sys.exit(f"incomplete LoRA pair for {module}: have {[k for k in lora if k.startswith(module)]}")
            a, b = lora[a_key].float(), lora[b_key].float()
            if a.shape[0] != args.lora_dim or b.shape[1] != args.lora_dim:
                sys.exit(f"{module}: lora shapes A{tuple(a.shape)} B{tuple(b.shape)} do not match --lora-dim {args.lora_dim}")
            # nemo module FQN -> HF key (the MoE adapter renames shared_experts -> shared_expert).
            candidates = [f"{module}.weight", f"{module}.weight".replace(".shared_experts.", ".shared_expert.")]
            hf_key = next((c for c in candidates if c in weight_map), None)
            if hf_key is None:
                sys.exit(f"{module}: no base tensor for any of {candidates}")
            w = load_base_tensor(base, weight_map, hf_key)
            update = scale * (b @ a)
            if tuple(update.shape) != tuple(w.shape):
                sys.exit(f"{hf_key}: LoRA update {tuple(update.shape)} vs base {tuple(w.shape)}")
            new = (w.float() + update).to(w.dtype).contiguous()
            merged[hf_key] = new
            changes.append(rel_change(new, w))
        lora_stats = {
            "modules": len(modules), "scale": scale, "dim": args.lora_dim, "alpha": args.lora_alpha,
            "rel_change_mean": float(sum(changes) / max(len(changes), 1)),
            "rel_change_max": float(max(changes)) if changes else 0.0,
            "lora_parameters": int(sum(t.numel() for t in lora.values())),
            "merged_bytes": int(sum(t.numel() * t.element_size() for t in merged.values())),
        }
        print("[lora]", json.dumps(lora_stats, indent=1))

    os.makedirs(args.out, exist_ok=True)
    for entry in sorted(os.listdir(base)):
        if entry in ("config.json", "model.safetensors.index.json"):
            continue
        dst = os.path.join(args.out, entry)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.realpath(os.path.join(base, entry)), dst)

    text["delta_engram_enabled"] = not args.no_delta
    text["delta_engram_table"] = args.table
    text["delta_alpha"] = float(args.alpha)
    if args.no_delta:
        pass
    elif args.table == "exact":
        keys_dst = os.path.join(args.out, "delta_exact_keys.pt")
        shutil.copyfile(args.keys, keys_dst)
        text["delta_exact_keys_path"] = keys_dst
    else:
        text["delta_ngram_vocab_size_per_head"] = int(args.rows_per_head)
    if merged:
        text["delta_lora_merged_path"] = os.path.join(args.out, LORA_FILE)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    if out_tensors:
        save_file(out_tensors, os.path.join(args.out, DELTA_FILE), metadata={"format": "pt"})
    if merged:
        save_file(merged, os.path.join(args.out, LORA_FILE), metadata={"format": "pt"})
    delta_bytes = sum(t.numel() * t.element_size() for t in out_tensors.values())
    for key in out_tensors:
        if key in weight_map:
            sys.exit(f"{key} already in the base weight map")
        weight_map[key] = DELTA_FILE
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})
    index["metadata"]["total_size"] = int(index["metadata"].get("total_size", 0)) + delta_bytes
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)

    try:
        commit = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        commit = "unknown"
    manifest = {
        "ckpt_model_dir": os.path.realpath(args.ckpt_model_dir),
        "lora": lora_stats,
        "base_snapshot": base,
        "delta_layer": layer,
        "table": args.table,
        "delta_alpha": float(args.alpha),
        "keys": os.path.realpath(args.keys) if args.keys else None,
        "layout": {"sizes": list(sizes), "offsets": list(offsets), "padded_rows": padded, "shards": parts},
        "delta_file_bytes": delta_bytes,
        "stats": stats,
        "export_script_commit": commit,
    }
    with open(os.path.join(args.out, "SERVING_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[done] {args.out} ({len(out_tensors)} delta tensors, {delta_bytes/2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
