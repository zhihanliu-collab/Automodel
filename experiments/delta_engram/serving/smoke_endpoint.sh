#!/bin/bash
# Smoke-test a Delta-Engram endpoint and show that it is not the base model.
# Usage: smoke_endpoint.sh <delta-host:port> [<base-host:port>]
# 1) /v1/models must list the served Delta name; 2) a chat completion must
# return content; 3) with a base endpoint given, the top-1 logprobs on an
# Odoo-flavoured prompt are printed side by side -- they should differ
# (the Delta table is not a no-op) while both stay coherent.
set -u
D="${1:?delta host:port}"; B="${2:-}"
echo "== models @ $D"; curl -sf -m 20 "http://$D/v1/models" | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])' || { echo "FAIL: /v1/models"; exit 1; }
MODEL=$(curl -sf -m 20 "http://$D/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
PROMPT='You are an accounts-payable clerk in Odoo 17. A vendor bill for ACME Drywall LLC shows a 10% retention line. In one sentence, what accounting date should the bill carry according to the company handbook?'
body() { python3 - "$1" "$PROMPT" <<'PY'
import json,sys
print(json.dumps({"model": sys.argv[1], "messages":[{"role":"user","content": sys.argv[2]}],
                  "max_tokens": 48, "temperature": 0, "logprobs": True, "top_logprobs": 1,
                  "chat_template_kwargs": {"enable_thinking": False}}))
PY
}
show() { python3 -c '
import json,sys
r=json.load(sys.stdin); ch=r["choices"][0]
print("  content:", (ch["message"].get("content") or "")[:200].replace("\n"," "))
lp=(ch.get("logprobs") or {}).get("content") or []
print("  first-8 token logprobs:", [round(t["logprob"],3) for t in lp[:8]])
print("  usage:", r.get("usage"))'; }
echo "== chat @ $D ($MODEL)"; curl -sf -m 120 "http://$D/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body "$MODEL")" | show || { echo "FAIL: chat"; exit 1; }
if [ -n "$B" ]; then
  BM=$(curl -sf -m 20 "http://$B/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  echo "== chat @ $B ($BM)"; curl -sf -m 120 "http://$B/v1/chat/completions" -H 'Content-Type: application/json' -d "$(body "$BM")" | show
fi
