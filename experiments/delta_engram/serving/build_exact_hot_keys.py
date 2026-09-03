#!/usr/bin/env python3
"""Collect the set of n-grams the Delta table was trained on ("exact-hot" keys).

The Delta table is hashed: an n-gram the training never saw still lands on a
row, and with ~51% occupancy that row usually belongs to some trained n-gram,
so novel strings (paths, e-mail domains, product names) get a foreign memory
injected into them. The serving-side fix masks the Delta lookup to n-grams that
actually occurred in the TRAIN split; this script builds that set from the
pretokenized corpus cache, using the same EOS-segment context rule as
Qwen3_8_FlashNextNGramEmbedding._shift_right_after_eos (a position whose
context crosses an EOS or the sequence start reads the EOS id instead).

Keys are packed int64s: bigram = prev*V + cur, trigram = (prev2*V + prev)*V + cur
with V = 2**18 (> vocab). Output: a torch file with sorted unique tensors.

    python build_exact_hot_keys.py --cache /mnt/data/zhihan/delta-engram/corpus/qwen38-131k-v1 \
        --out /mnt/data/zhihan/delta-engram/serving/exact_hot_keys_train.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

EOS = 248044
V = 1 << 18


def shifted(ids: torch.Tensor, shift: int) -> torch.Tensor:
    """Token `shift` positions back within the same EOS-delimited segment, else EOS."""
    if shift == 0:
        return ids
    n = ids.shape[0]
    pos = torch.arange(n, dtype=torch.long)
    eos_pos = torch.where(ids == EOS, pos, torch.full_like(pos, -1))
    prev_incl = torch.cummax(eos_pos, dim=0).values
    prev = torch.cat([torch.tensor([-1]), prev_incl[:-1]])
    pos_in_seg = pos - (prev + 1)
    src = pos - shift
    out = ids[src.clamp_min(0)]
    valid = (pos_in_seg >= shift) & (src >= 0)
    return torch.where(valid, out, torch.full_like(out, EOS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    manifest = json.load(open(f"{args.cache}/manifest.json"))
    ids_all = np.memmap(f"{args.cache}/input_ids.i32", dtype=np.int32, mode="r")
    assert ids_all.shape[0] == manifest["num_tokens_shifted"], (ids_all.shape, manifest["num_tokens_shifted"])
    vocab_max = 0
    bigrams, trigrams = [], []
    n_samples = n_tokens = 0
    for rec in manifest["records"]:
        if rec["split"] != args.split:
            continue
        ids = torch.from_numpy(np.array(ids_all[rec["offset"] : rec["offset"] + rec["length"]], dtype=np.int64))
        vocab_max = max(vocab_max, int(ids.max()))
        t0, t1, t2 = ids, shifted(ids, 1), shifted(ids, 2)
        bigrams.append(torch.unique(t1 * V + t0))
        trigrams.append(torch.unique((t2 * V + t1) * V + t0))
        n_samples += 1
        n_tokens += ids.shape[0]
    assert vocab_max < V, vocab_max
    bi = torch.unique(torch.cat(bigrams))
    tri = torch.unique(torch.cat(trigrams))
    torch.save({"bigram": bi, "trigram": tri, "eos": EOS, "pack_base": V, "split": args.split,
                "samples": n_samples, "tokens": n_tokens, "cache": args.cache}, args.out)
    print(f"split={args.split} samples={n_samples} tokens={n_tokens} unique bigrams={bi.numel()} unique trigrams={tri.numel()} -> {args.out}")


if __name__ == "__main__":
    main()
