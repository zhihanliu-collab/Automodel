#!/usr/bin/env bash
# Inference-side alpha ablation of one exported Delta checkpoint on the fixed eval20 list,
# one alpha at a time on the 4 free H200s (h200-1 GPUs 0-3; the base reference server on
# that node keeps GPUs 4-7). Per alpha: variant serving dir (config delta_alpha=<a>) ->
# serve_delta.sbatch -> eval20 arm on the CPU sandbox job -> one repair round -> stop server.
# Arm names: dvsd_<atag>e20_base = Delta v2 (s675), seeded stateless claude-sdk host, no
# reviewer, alpha <atag> (a01=0.1, a001=0.01, a0001=0.001), eval20 list.
# Usage: alpha_ablation_chain.sh <src-tag e.g. v2s675> <sandbox-jobid> <node> <port> <container-suffix> <alpha>...
set -euo pipefail
SRC="${1:?src serving tag}"; JOB="${2:?sandbox jobid}"; NODE="${3:?node}"; PORT="${4:?port}"; CNAME="${5:?container suffix}"; shift 5
ALPHAS=("$@"); [ ${#ALPHAS[@]} -gt 0 ] || { echo "need alphas"; exit 2; }
REPO=/home/zhihan/delta-engram-automodel; SV=/mnt/data/zhihan/delta-engram/serving
LAUNCH=$HOME/reduce100-launchers/pipeline_runs/nebius_cpu_sandbox
LIST=/mnt/data/zhihan/reviewer_reduce100/lists/eval20.txt; RUNS=/mnt/data/zhihan/reviewer_reduce100/runs
WORKERS="${WORKERS:-10}"; PARTITION="${PARTITION:-h200}"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
graded() { python3 - "$RUNS" "$1" <<'PY'
import glob, json, sys
ids = set()
for f in glob.glob(f"{sys.argv[1]}/*_reduce100*_{sys.argv[2]}_*/task-*/result.json"):
    try: ids.add(json.load(open(f))["task_id"])
    except Exception: pass
print(",".join(sorted(ids)))
PY
}
for A in "${ALPHAS[@]}"; do
  ATAG="a$(echo "$A" | sed 's/^0\.//; s/\.//g')"; TAG="${SRC}_${ATAG}"; ARM="dvsd_${ATAG}e20"
  log "=== alpha=$A tag=$TAG arm=${ARM}_base"
  [ -f "$SV/$TAG/SERVING_MANIFEST.json" ] || python3 "$REPO/experiments/delta_engram/serving/make_variant_dir.py" --src "$SV/$SRC" --dst "$SV/$TAG" --set "delta_alpha=$A"
  python3 -c "import json,sys; c=json.load(open('$SV/$TAG/config.json'))['text_config']; assert abs(c['delta_alpha']-$A)<1e-12, c['delta_alpha']; print('config delta_alpha', c['delta_alpha'])"
  SJOB=$(sbatch --parsable -p "$PARTITION" -w "$NODE" "$REPO/experiments/delta_engram/serving/serve_delta.sbatch" "$TAG" "$PORT" "$CNAME")
  log "serving job $SJOB"
  for i in $(seq 1 90); do
    curl -sf -m 5 "http://$NODE:$PORT/v1/models" 2>/dev/null | grep -q "Delta-$TAG" && break
    st=$(squeue -h -j "$SJOB" -o %T); [ -z "$st" ] && { log "serving job $SJOB died before ready"; break; }
    sleep 20
  done
  curl -sf -m 5 "http://$NODE:$PORT/v1/models" | grep -q "Delta-$TAG" || { log "endpoint never came up for $TAG; skipping alpha $A"; scancel "$SJOB" 2>/dev/null || true; continue; }
  log "endpoint up: $(curl -sf -m 5 http://$NODE:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')"
  export DELTA_MODEL="Qwen/Qwen3.8-Flash-Next-Delta-$TAG"
  bash "$LAUNCH/add_arms_overlap.sh" "$JOB" "${ARM}_base:$LIST:$WORKERS::$NODE:$PORT"
  sleep 120; while squeue -s -h -j "$JOB" -o %j | grep -q "rr100_${ARM}_"; do sleep 60; done
  done_ids=$(graded "$ARM"); n=$(echo "$done_ids" | tr ',' '\n' | grep -c . || true); log "graded $n/20"
  if [ "$n" -lt 20 ]; then
    python3 - "$LIST" "$done_ids" "/mnt/data/zhihan/reviewer_reduce100/lists/${ARM}_r1.txt" <<'PY'
import sys
full=[t for t in open(sys.argv[1]).read().replace("\n","").split(",") if t]; done=set(sys.argv[2].split(",")) if sys.argv[2] else set()
rest=[t for t in full if t not in done]; open(sys.argv[3],"w").write(",".join(rest)+"\n"); print("repair", len(rest))
PY
    bash "$LAUNCH/add_arms_overlap.sh" "$JOB" "${ARM}r1_base:/mnt/data/zhihan/reviewer_reduce100/lists/${ARM}_r1.txt:$WORKERS::$NODE:$PORT"
    sleep 120; while squeue -s -h -j "$JOB" -o %j | grep -q "rr100_${ARM}r1_"; do sleep 60; done
  fi
  scancel "$SJOB"; log "alpha=$A done, graded=$(graded "$ARM" | tr ',' '\n' | grep -c . || true); server $SJOB stopped"
  sleep 90
done
log "ALPHA CHAIN DONE"
