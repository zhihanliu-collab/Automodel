#!/usr/bin/env python3
"""Training-side vs SGLang-side Delta n-gram hashing on the same token windows.

Both implementations are pure torch; the bodies below are copied from
nemo_automodel .../qwen3_8_flash_next/engram.py (`_hash_input_ids`) and from
sglang/srt/models/qwen4_exp.py (`_hash_contexts` + `_shift_right_ignore_eos`).
The Delta multipliers are ~2^62, so token*multiplier wraps in int64; the test
checks that both sides agree row-for-row after the wrap, on random sequences
with EOS boundaries, on CPU and (if present) on CUDA. Exit code 1 on mismatch.
"""

import sys

import torch

EOS = 248044
MULT = (6364136223846793005, 1442695040888963407, 3202034522624059733)
SIZES = (1000003, 1000033, 1000037, 1000039, 1000081, 1000099, 1000117, 1000121, 1000133, 1000151, 1000159, 1000171, 1000183, 1000187, 1000193, 1000199)
OFFS = tuple(sum(SIZES[:i]) for i in range(16))
NGRAM, HEADS = 3, 8


def _next_prime(v):
    def isp(n):
        if n < 2:
            return False
        i = 2
        while i * i <= n:
            if n % i == 0:
                return False
            i += 1
        return True
    while not isp(v):
        v += 1
    return v


def training_layout():
    sizes, c = [], 1_000_000
    for _ in range(16):
        p = _next_prime(c)
        sizes.append(p)
        c = p + 1
    return tuple(sizes)


# ---- training side (nemo) ----
def nemo_shift(input_ids, shift):
    if shift == 0:
        return input_ids
    b, s = input_ids.shape
    pos = torch.arange(s, device=input_ids.device, dtype=torch.long)
    eos_pos = torch.where(input_ids == EOS, pos, -1)
    prev_incl = torch.cummax(eos_pos, dim=1).values
    prev = torch.cat([eos_pos.new_full((b, 1), -1), prev_incl[:, :-1]], dim=1)
    pis = pos.unsqueeze(0) - (prev + 1)
    src = pos - shift
    g = src.clamp_min(0).unsqueeze(0).expand(b, -1)
    shifted = input_ids.gather(1, g)
    valid = (pis >= shift) & (src.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, input_ids.new_full((), EOS))


def nemo_hash(input_ids, mult, sizes, offs):
    input_ids = input_ids.to(torch.long)
    st = [nemo_shift(input_ids, k) for k in range(NGRAM)]
    blocks = []
    for order in range(2, NGRAM + 1):
        hs, he = (order - 2) * HEADS, (order - 1) * HEADS
        mixed = st[0] * mult[0]
        for p in range(1, order):
            mixed = torch.bitwise_xor(mixed, st[p] * mult[p])
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes[hs:he]) + offs[hs:he])
    return torch.cat(blocks, dim=-1)  # [b, s, 16]


# ---- serving side (sglang, non-fused path) ----
def sgl_shift(t, n):
    if n == 0:
        return t
    b, s = t.shape
    idx = torch.arange(s, device=t.device, dtype=torch.long)
    eos_pos = torch.where(t == EOS, idx, -1)
    prev_incl = torch.cummax(eos_pos, dim=1).values
    prev = torch.cat([eos_pos.new_full((b, 1), -1), prev_incl[:, :-1]], dim=1)
    pis = idx.unsqueeze(0) - (prev + 1)
    src = idx - n
    g = torch.clamp(src, min=0).unsqueeze(0).expand(b, -1)
    shifted = t.gather(1, g)
    valid = (pis >= n) & (src.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, t.new_full((), EOS))


def sgl_hash(contexts, mult, sizes, offs):
    # contexts: [n, NGRAM] windows (oldest..newest), as compute_ngram_ids builds them
    contexts = contexts.to(torch.long)
    st = [contexts] + [sgl_shift(contexts, k) for k in range(1, NGRAM)]
    blocks = []
    for ng in range(2, NGRAM + 1):
        s0, s1 = (ng - 2) * HEADS, (ng - 1) * HEADS
        mix = st[0] * mult[0]
        for p in range(1, ng):
            mix = torch.bitwise_xor(mix, st[p] * mult[p])
        ids = torch.remainder(mix[:, -1:].unsqueeze(-1), sizes[s0:s1].view(1, 1, -1)) + offs[s0:s1].view(1, 1, -1)
        blocks.append(ids[:, 0])
    return torch.cat(blocks, dim=-1)  # [n, 16]


def run(device):
    torch.manual_seed(0)
    mult = torch.tensor(MULT, dtype=torch.long, device=device)
    sizes = torch.tensor(SIZES, dtype=torch.long, device=device)
    offs = torch.tensor(OFFS, dtype=torch.long, device=device)
    seq = torch.randint(0, 248000, (4, 512), device=device)
    seq[:, 100] = EOS
    seq[1, 5] = EOS
    seq[2, 0] = EOS
    ref = nemo_hash(seq, mult, sizes, offs)  # [4, 512, 16]
    # serving sees, for position t, the window [t-2, t-1, t] with EOS padding before segment start / sequence start
    pad = torch.full((4, NGRAM - 1), EOS, dtype=torch.long, device=device)
    padded = torch.cat([pad, seq], dim=1)
    win = padded.unfold(1, NGRAM, 1)  # [4, 512, 3]
    got = sgl_hash(win.reshape(-1, NGRAM), mult, sizes, offs).view(4, 512, 16)
    mism = (ref != got).sum().item()
    neg = (seq.to(torch.long) * mult[0] < 0).float().mean().item()
    print(f"[{device}] rows compared={ref.numel()} mismatches={mism} (fraction of products wrapping negative: {neg:.2f}); "
          f"row range [{got.min().item()}, {got.max().item()}] < 16001920")
    return mism == 0 and got.max().item() < 16001920


if __name__ == "__main__":
    assert training_layout() == SIZES, training_layout()
    ok = run("cpu")
    if torch.cuda.is_available():
        ok = run("cuda") and ok
    print("PARITY_OK" if ok else "PARITY_MISMATCH")
    sys.exit(0 if ok else 1)
