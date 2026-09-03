#!/usr/bin/env python3
"""Delta-on vs base held-out NLL through the live SGLang endpoints.

Uses ``/v1/completions`` with ``echo=True, logprobs=1`` to get the prompt
token logprobs, so the same text can be scored under several servers without
any GPU work of our own. Reports mean NLL per text per endpoint plus how many
tokens moved by more than 2 nats between base and Delta. Run from a compute
node (CPU step) where the endpoints are reachable::

    python nll_probe.py --base b200-0:30000 --delta s164=b200-2:30000 --delta s329=b200-4:30000
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request

TEXTS = {
    # in-domain natural prose the host reads every task
    "handbook": "/mnt/data/zhihan/reviewer_reduce100/seed_ws_opus5_v6/workspace/COMPANY-HANDBOOK.md",
    "mcp_tutorial": "/mnt/data/zhihan/reviewer_reduce100/seed_ws_opus5_v6/workspace/ODOO-MCP-TUTORIAL.md",
    # out-of-domain technical prose (general LM health)
    "handoff_prose": "/home/zhihan/delta-engram-automodel/experiments/delta_engram/HANDOFF_2026-09-03.md",
}
OFFLINE_BILLS = "/mnt/data/zhihan/delta-engram/corpus/raw/odoo_offline_bills_20260817.jsonl"


def models(hp: str) -> str:
    with urllib.request.urlopen(f"http://{hp}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def prompt_logprobs(hp: str, model: str, text: str) -> list[float]:
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://{hp}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        lp = json.load(r)["choices"][0]["logprobs"]["token_logprobs"]
    return [x for x in lp[:-1] if x is not None]  # drop the first (None) and the generated token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--delta", action="append", default=[], help="tag=host:port")
    ap.add_argument("--max-chars", type=int, default=24000)
    args = ap.parse_args()
    texts = {k: open(p).read()[: args.max_chars] for k, p in TEXTS.items()}
    with open(OFFLINE_BILLS) as fh:
        first = json.loads(fh.readline())
    texts["offline_bill_json"] = json.dumps({k: first[k] for k in ("header", "lines")}, indent=1)[: args.max_chars]
    endpoints = [("base", args.base)] + [tuple(d.split("=", 1)) for d in args.delta]
    names = {tag: models(hp) for tag, hp in endpoints}
    print("endpoints:", {t: (hp, names[t]) for t, hp in endpoints})
    print(f"{'text':18s} {'tokens':>6s} " + " ".join(f"{t + ' nll':>12s}" for t, _ in endpoints) + "   |dlp|>2 vs base")
    for name, text in texts.items():
        lps = {tag: prompt_logprobs(hp, names[tag], text) for tag, hp in endpoints}
        n = min(len(v) for v in lps.values())
        row = f"{name:18s} {n:6d} "
        for tag, _ in endpoints:
            row += f"{-sum(lps[tag][:n]) / n:12.4f} "
        moved = []
        for tag, _ in endpoints[1:]:
            moved.append(f"{tag}:{sum(1 for a, b in zip(lps['base'][:n], lps[tag][:n]) if abs(a - b) > 2)}")
        print(row + "   " + " ".join(moved))


if __name__ == "__main__":
    main()
