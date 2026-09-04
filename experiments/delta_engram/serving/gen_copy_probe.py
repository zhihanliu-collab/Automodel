#!/usr/bin/env python3
"""Generation-side copy fidelity: can the endpoint reproduce a long rare string it was just shown?

The echo-logprob probes only exercise prefill. Verbatim copying during agent
runs happens in decode (with speculative decoding when NEXTN is on), so this
probe asks each endpoint to repeat paths / e-mails / product names at
temperature 0 and reports the exact-match rate and the first divergence.

    python gen_copy_probe.py --ep base=h200-1:30001 --ep v2s339=h200-1:30000 [--n 24]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request

RUN_ROOT = "/mnt/data/zhihan/reviewer_reduce100/runs/odoo_ap_stateless_claude-sdk_seeded_Qwen-Qwen3.8-Flash-Next-Delta-v2s339_reduce100repair_dvsd_q3_base_0904-001830-b200-5-p161652"
FILES = ["memory/MEMORY.md", "memory/odoo-ap-tooling-notes.md", "memory/ap-retention-and-pay-apps.md", "downloads/LGFM-496160.pdf",
         "memory/ap-posting-and-dating-conventions.md", "downloads/PO-WSFH-2026-0121-subcontract.pdf"]
EXTRA = [
    "apetrov@foothillmechanicalservic.com",
    "office@goldengateguardiansystems.com",
    "sophie.delgado@eastbayrootandtree.com",
    "27433 Audio-Visual - Racks & Enclosures",
    "Pay App #2 (GGGS-2026-561701) on subcontract PO-BERH-2026-0247",
    "Los Gatos Fine Masonry / Valley Retaining Systems / East Bay Root & Tree Protection",
    "COI-GeneralLiability_Foothill-Mechanical-Service.pdf",
    "reduce100repair_d329sd_q1_base_0903-191719-b200-5-p149340",
]


def strings(n: int) -> list[str]:
    random.seed(7)
    out = []
    for i in range(n - len(EXTRA)):
        task = random.randint(0, 24)
        out.append(f"{RUN_ROOT}/task-{task:03d}/workspace/{random.choice(FILES)}")
    return out + EXTRA


def model_of(hp: str) -> str:
    with urllib.request.urlopen(f"http://{hp}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def repeat(hp: str, model: str, s: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"Repeat the following string exactly, with no quotes and nothing else:\n{s}"}],
        "temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"http://{hp}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return (json.load(r)["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", action="append", required=True, help="tag=host:port")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    eps = [tuple(e.split("=", 1)) for e in args.ep]
    names = {t: model_of(hp) for t, hp in eps}
    print("endpoints:", {t: (hp, names[t]) for t, hp in eps})
    tests = strings(args.n)
    for tag, hp in eps:
        ok = 0
        fails = []
        for s in tests:
            out = repeat(hp, names[tag], s)
            if out == s:
                ok += 1
            else:
                k = next((i for i, (a, b) in enumerate(zip(s, out)) if a != b), min(len(s), len(out)))
                fails.append(f"    ...{s[max(0,k-18):k]}[{s[k:k+8]!r} -> {out[k:k+8]!r}]")
        print(f"{tag:10s} exact {ok}/{len(tests)}")
        for f in fails[:8]:
            print(f)


if __name__ == "__main__":
    main()
