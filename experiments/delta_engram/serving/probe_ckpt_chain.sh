#!/usr/bin/env bash
# One checkpoint -> servable dir (Delta [+ merged LoRA]) -> one TP4 endpoint -> echo-NLL + copy probes
# vs a base endpoint -> eval20 arm (private launcher worktree) -> repair round -> stop the server.
# Usage: probe_ckpt_chain.sh <ckpt-model-dir> <tag> <alpha> <keys.pt> <node> <port> <container-suffix> <base host:port> [lora_dim lora_alpha]
#   e.g. probe_ckpt_chain.sh /mnt/data/zhihan/delta-engram/checkpoints/odoo-delta-v2-lora-r8/epoch_0_step_169/model \
#          v2lora_r8_s169 0.25 /mnt/data/zhihan/delta-engram/corpus/qwen38-131k-v3/delta_exact_keys_train.pt h200-1 30000 tp4 h200-1:30001 8 16
# Env: SANDBOX_JOB (default: my rr100_all job on b200-5), EXPORT_NODE (b200-2), WORKERS (10), PARTITION (h200),
#      ARM (default dvsd_<tag>e20; must match a repair_reduce100.sh case: dvsd_* uses DELTA_MODEL).
set -euo pipefail
CKPT="${1:?ckpt model dir}"; TAG="${2:?tag}"; ALPHA="${3:?alpha}"; KEYS="${4:?keys.pt}"; NODE="${5:?node}"; PORT="${6:?port}"
CNAME="${7:?container suffix}"; BASE_EP="${8:?base host:port}"; LORA_DIM="${9:-}"; LORA_ALPHA="${10:-}"
REPO=/home/zhihan/delta-engram-automodel; SV=/mnt/data/zhihan/delta-engram/serving; OUT=$SV/$TAG
BASE_SNAP=/mnt/data/zhihan/hf_cache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540
LAUNCH="${LAUNCH_DIR:-$HOME/reduce100-launchers-delta/pipeline_runs/nebius_cpu_sandbox}"
export SRC_DIR="${SRC_DIR:-$HOME/research-reviewer-agent-delta/src/odoo-baselines}"
LIST=/mnt/data/zhihan/reviewer_reduce100/lists/eval20.txt; RUNS=/mnt/data/zhihan/reviewer_reduce100/runs
WORKERS="${WORKERS:-10}"; PARTITION="${PARTITION:-h200}"; EXPORT_NODE="${EXPORT_NODE:-b200-2}"
JOB="${SANDBOX_JOB:-$(squeue --me -h -o "%i %j %N" | awk '$2=="rr100_all" && $3=="b200-5"{print $1}' | head -1)}"
ARM="${ARM:-dvsd_${TAG}e20}"
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
log "ckpt=$CKPT tag=$TAG alpha=$ALPHA lora=${LORA_DIM:-none}/${LORA_ALPHA:-} node=$NODE:$PORT sandbox_job=$JOB arm=${ARM}_base"
if [ ! -f "$OUT/SERVING_MANIFEST.json" ]; then
  LORA_ARGS=(); [ -n "$LORA_DIM" ] && LORA_ARGS=(--lora-dim "$LORA_DIM" --lora-alpha "$LORA_ALPHA")
  srun -p b200 -w "$EXPORT_NODE" --overlap --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name="export-$TAG" \
    --container-name=lf-gdn-smoke --container-mounts=/mnt/data:/mnt/data,$REPO:/workspace \
    bash -lc "export PYTHONNOUSERSITE=1; python /workspace/experiments/delta_engram/serving/export_delta_serving_dir.py \
      --ckpt-model-dir $CKPT --base-snapshot $BASE_SNAP --out $OUT --table exact --keys $KEYS --alpha $ALPHA ${LORA_ARGS[*]}" \
    2>&1 | grep -v -E "cpu-bind|ignoring --container|^\[reassemble\]" | tee "$SV/logs/export-$TAG.log" | grep -E "layout|\[lora\]|modules|rel_change|done|Error|exit" | head -20
  [ -f "$OUT/SERVING_MANIFEST.json" ] || { log "export failed"; exit 1; }
fi
SJOB=$(sbatch --parsable -p "$PARTITION" -w "$NODE" "$REPO/experiments/delta_engram/serving/serve_delta.sbatch" "$TAG" "$PORT" "$CNAME"); log "serving job $SJOB"
for i in $(seq 1 90); do
  curl -sf -m 5 "http://$NODE:$PORT/v1/models" 2>/dev/null | grep -q "Delta-$TAG" && break
  [ -z "$(squeue -h -j "$SJOB" -o %T)" ] && { log "serving job $SJOB died"; exit 1; }
  sleep 20
done
curl -sf -m 5 "http://$NODE:$PORT/v1/models" | grep -q "Delta-$TAG" || { log "endpoint never came up"; scancel "$SJOB"; exit 1; }
log "endpoint up"
RUN="srun --jobid=$JOB --overlap --ntasks=1 --cpus-per-task=1"
$RUN python3 "$REPO/experiments/delta_engram/serving/nll_probe.py" --base "$BASE_EP" --delta "$TAG=$NODE:$PORT" 2>&1 | grep -v cpu-bind | tee "$SV/logs/nll-$TAG.log" | tail -6
$RUN python3 "$REPO/experiments/delta_engram/serving/copy_probe.py" --base "$BASE_EP" --delta "$TAG=$NODE:$PORT" 2>&1 | grep -v cpu-bind | tee "$SV/logs/copy-$TAG.log" | tail -4
export DELTA_MODEL="Qwen/Qwen3.8-Flash-Next-Delta-$TAG"
bash "$LAUNCH/add_arms_overlap.sh" "$JOB" "${ARM}_base:$LIST:$WORKERS::$NODE:$PORT"
sleep 180; while squeue -s -h -j "$JOB" -o %j | grep -q "rr100_${ARM}_"; do sleep 60; done
done_ids=$(graded "$ARM"); n=$(echo "$done_ids" | tr ',' '\n' | grep -c . || true); log "graded $n/20"
if [ "$n" -lt 20 ]; then
  python3 - "$LIST" "$done_ids" "/mnt/data/zhihan/reviewer_reduce100/lists/${ARM}_r1.txt" <<'PY'
import sys
full=[t for t in open(sys.argv[1]).read().replace("\n","").split(",") if t]; done=set(sys.argv[2].split(",")) if sys.argv[2] else set()
rest=[t for t in full if t not in done]; open(sys.argv[3],"w").write(",".join(rest)+"\n"); print("repair", len(rest))
PY
  bash "$LAUNCH/add_arms_overlap.sh" "$JOB" "${ARM}r1_base:/mnt/data/zhihan/reviewer_reduce100/lists/${ARM}_r1.txt:$WORKERS::$NODE:$PORT"
  sleep 180; while squeue -s -h -j "$JOB" -o %j | grep -q "rr100_${ARM}r1_"; do sleep 60; done
fi
scancel "$SJOB"; log "PROBE CHAIN DONE tag=$TAG graded=$(graded "$ARM" | tr ',' '\n' | grep -c . || true); server $SJOB stopped"
