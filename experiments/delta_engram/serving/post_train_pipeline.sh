#!/bin/bash
# Checkpoint -> servable dir -> endpoints -> probes, end to end, on the Nebius cluster.
# Usage: [PARTITION=b200|h200] [EXPORT_NODE=b200-2] post_train_pipeline.sh <ckpt-model-dir> <tag> <table: exact|hashed> <alpha> <keys.pt|-> <node-a> <node-b>
#   LORA_DIM/LORA_ALPHA: set for Delta+LoRA checkpoints (merged into the export, see export_delta_serving_dir.py).
#   PARTITION: partition of the serving nodes (default b200). The export runs as a CPU step in the
#   training container (lf-gdn-smoke, repo at /workspace) on EXPORT_NODE (default b200-2).
#   e.g. post_train_pipeline.sh /mnt/data/zhihan/delta-engram/checkpoints/odoo-delta-v2/epoch_0_step_339/model v2s339 exact 0.25 \
#          /mnt/data/zhihan/delta-engram/corpus/qwen38-131k-v2/delta_exact_keys_train.pt b200-2 b200-3
# Serves 4 endpoints (2 per node, ports 30000/30001) under name Qwen/Qwen3.8-Flash-Next-Delta-<tag>,
# then runs nll_probe.py and copy_probe.py against the base replica b200-0:30000 and prints PROBE_GATE.
# The eval arms are launched separately (reduce100 launchers, add_arms_overlap.sh) once PROBE_GATE=pass.
set -euo pipefail
CKPT="${1:?ckpt model dir}"; TAG="${2:?tag}"; TABLE="${3:?exact|hashed}"; ALPHA="${4:?alpha}"; KEYS="${5:?keys.pt or -}"
NODE_A="${6:?node a}"; NODE_B="${7:?node b}"
PARTITION="${PARTITION:-b200}"; EXPORT_NODE="${EXPORT_NODE:-b200-2}"
REPO=/home/zhihan/delta-engram-automodel
SV=/mnt/data/zhihan/delta-engram/serving
BASE=/mnt/data/zhihan/hf_cache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540
OUT=$SV/$TAG
mkdir -p "$SV/logs"

if [[ ! -f "$OUT/SERVING_MANIFEST.json" ]]; then
  echo "[$(date -u +%FT%TZ)] export $CKPT -> $OUT"
  KEY_ARGS=(); [[ "$TABLE" == "exact" ]] && KEY_ARGS=(--keys "$KEYS")
  # Delta+LoRA checkpoints: LORA_DIM/LORA_ALPHA (peft.dim / peft.alpha of the run) merge the adapters.
  [[ -n "${LORA_DIM:-}" ]] && KEY_ARGS+=(--lora-dim "$LORA_DIM" --lora-alpha "${LORA_ALPHA:?LORA_ALPHA}")
  # The training container is usually already running on the node (attach ignores new
  # mounts), and it carries the repo at /workspace, torch and safetensors.
  srun -p b200 -w "$EXPORT_NODE" --ntasks=1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name="export-$TAG" \
    --container-name=lf-gdn-smoke --container-mounts=/mnt/data:/mnt/data,$REPO:/workspace \
    bash -lc "export PYTHONNOUSERSITE=1; python /workspace/experiments/delta_engram/serving/export_delta_serving_dir.py \
      --ckpt-model-dir $CKPT --base-snapshot $BASE --out $OUT --table $TABLE --alpha $ALPHA ${KEY_ARGS[*]}" \
    2>&1 | grep -v -E "cpu-bind|ignoring --container|^\[reassemble\]" | tee "$SV/logs/export-$TAG.log" | grep -E "layout|rms|nonzero|rel_change|done" | head -12
fi

echo "[$(date -u +%FT%TZ)] serving $TAG on $NODE_A $NODE_B"
S=$REPO/experiments/delta_engram/serving/serve_delta.sbatch
JOBS=()
for spec in "$NODE_A 30000 tp4-$NODE_A" "$NODE_A 30001 tp4b-$NODE_A" "$NODE_B 30000 tp4-$NODE_B" "$NODE_B 30001 tp4b-$NODE_B"; do
  set -- $spec
  JOBS+=("$(sbatch --parsable -p "$PARTITION" -w "$1" "$S" "$TAG" "$2" "$3")")
done
echo "serving jobs: ${JOBS[*]}"
EP="$NODE_A:30000 $NODE_A:30001 $NODE_B:30000 $NODE_B:30001"
for i in $(seq 1 80); do
  up=0; for e in $EP; do curl -sf -m 5 -o /dev/null "http://$e/v1/models" && up=$((up+1)); done
  [[ $up -eq 4 ]] && break
  sleep 20
done
echo "[$(date -u +%FT%TZ)] endpoints up: $up/4"
[[ $up -eq 4 ]] || { echo "PROBE_GATE=serving_failed"; exit 1; }

echo "[$(date -u +%FT%TZ)] probes"
PROBE_JOB="${SANDBOX_JOB:-$(squeue --me -h -o "%i %j" | awk '$2=="rr100_all"{print $1}' | head -1)}"
RUN="srun --jobid=$PROBE_JOB --overlap --ntasks=1 --cpus-per-task=1"
$RUN python3 "$REPO/experiments/delta_engram/serving/nll_probe.py" --base b200-0:30000 --delta "$TAG=$NODE_A:30000" 2>&1 | grep -v cpu-bind | tee "$SV/logs/nll-$TAG.log"
$RUN python3 "$REPO/experiments/delta_engram/serving/copy_probe.py" --base b200-0:30000 --delta "$TAG=$NODE_A:30000" 2>&1 | grep -v cpu-bind | tee "$SV/logs/copy-$TAG.log"
# Gate: handbook NLL must not exceed base, and mean copy NLL must stay within 0.05 nats of base.
python3 - "$SV/logs/nll-$TAG.log" "$SV/logs/copy-$TAG.log" <<'PY'
import re, sys
nll = open(sys.argv[1]).read(); cp = open(sys.argv[2]).read()
hb = re.search(r"^handbook\s+\d+\s+([\d.]+)\s+([\d.]+)", nll, re.M)
allrow = re.search(r"^ALL \(mean\)\s+([\d.]+)\s+([\d.]+)", cp, re.M)
if not hb or not allrow:
    print("PROBE_GATE=unparsed"); sys.exit(1)
base_hb, delta_hb = float(hb.group(1)), float(hb.group(2))
base_cp, delta_cp = float(allrow.group(1)), float(allrow.group(2))
ok = delta_hb <= base_hb + 0.02 and delta_cp <= base_cp + 0.05
print(f"handbook nll base={base_hb:.3f} delta={delta_hb:.3f} | copy nll base={base_cp:.3f} delta={delta_cp:.3f}")
print("PROBE_GATE=" + ("pass" if ok else "fail"))
PY
