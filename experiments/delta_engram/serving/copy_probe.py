#!/usr/bin/env python3
"""How well does an endpoint copy a rare string that is right there in its context?

For each probe string S the prompt is "<label>: S\nRepeat exactly: S"; we score
(echo logprobs) only the tokens of the second S. Copying should be near-free
(NLL ~ 0.0x per token) for a healthy model; the Delta failure mode shows up as
tokens inside paths / e-mail domains / product names getting several nats worse
than base. Prints per-string mean NLL and the worst token per endpoint.

    python copy_probe.py --base b200-0:30000 --delta s329=b200-4:30000 --delta s329_xh=b200-2:30000
"""

from __future__ import annotations

import argparse
import json
import urllib.request

STRINGS = {
    "run_path": "/mnt/data/zhihan/reviewer_reduce100/runs/odoo_ap_stateless_claude-sdk_seeded_Qwen-Qwen3.8-Flash-Next/task-010/workspace/memory/odoo-ap-tooling-notes.md",
    "email": "apetrov@foothillmechanicalservic.com",
    "email2": "office@goldengateguardiansystems.com",
    "product": "27433 Audio-Visual - Racks & Enclosures\\nmedia room gear racks, wall-mounted",
    "invoice_ref": "Pay App #2 (GGGS-2026-561701) on subcontract PO-BERH-2026-0247, Berry Hill Residence",
    "vendor": "Los Gatos Fine Masonry / Valley Retaining Systems / East Bay Root & Tree Protection",
    "po": "PO-WSFH-2026-0121-subcontract.pdf and PO-LACO-2026-0127-subcontract.pdf",
}


def models(hp):
    with urllib.request.urlopen(f"http://{hp}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def echo(hp, model, text):
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://{hp}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        lp = json.load(r)["choices"][0]["logprobs"]
    toks, lps = lp["tokens"][:-1], lp["token_logprobs"][:-1]
    # SGLang returns text_offset=-1 for echoed prompts; rebuild char offsets from the tokens.
    offs, pos = [], 0
    for tok in toks:
        offs.append(pos)
        pos += len(tok)
    return toks, lps, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--delta", action="append", default=[])
    args = ap.parse_args()
    eps = [("base", args.base)] + [tuple(d.split("=", 1)) for d in args.delta]
    names = {t: models(hp) for t, hp in eps}
    print("endpoints:", {t: (hp, names[t]) for t, hp in eps})
    hdr = f"{'string':12s} {'toks':>4s} " + " ".join(f"{t:>10s}" for t, _ in eps) + "   worst token (tag: tok nll)"
    print(hdr)
    totals = {t: [0.0, 0] for t, _ in eps}
    for label, s in STRINGS.items():
        prompt = f"{label}: {s}\nRepeat exactly: {s}"
        start = len(prompt) - len(s) - 1  # a token may straddle the space before S
        row = f"{label:12s}"
        worst = []
        n_out = None
        for tag, hp in eps:
            toks, lps, offs = echo(hp, names[tag], prompt)
            sel = [(t, l) for t, l, o in zip(toks, lps, offs) if o >= start and l is not None]
            n = len(sel)
            n_out = n if n_out is None else n_out
            nll = -sum(l for _, l in sel) / max(n, 1)
            totals[tag][0] += -sum(l for _, l in sel)
            totals[tag][1] += n
            w = min(sel, key=lambda x: x[1]) if sel else ("", 0.0)
            worst.append(f"{tag}:{w[0]!r}{-w[1]:.1f}")
            row += f" {nll:10.3f}" if tag != "base" else f"{n:5d} {nll:10.3f}"
        print(row + "   " + " ".join(worst))
    print(f"{'ALL (mean)':12s}      " + " ".join(f"{totals[t][0]/max(totals[t][1],1):10.3f}" for t, _ in eps))


if __name__ == "__main__":
    main()
