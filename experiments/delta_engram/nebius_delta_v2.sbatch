#!/bin/bash
#SBATCH --job-name=q38-delta-v2
#SBATCH --partition=b200
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

# Delta-Engram v2 training launcher (4 x 8 B200, DP4 x CP8, EP16). All knobs are
# environment variables so a smoke run and the formal run share one script:
#   CONFIG_PATH       recipe yaml (default experiments/delta_engram/qwen3_8_flash_next_odoo_delta_v2.yaml)
#   CHECKPOINT_TAG    checkpoint dir name under /mnt/data/zhihan/delta-engram/checkpoints (required)
#   MAX_STEPS         override step_scheduler.max_steps (default: yaml)
#   NUM_EPOCHS        override step_scheduler.num_epochs (default: yaml)
#   VAL_EVERY_STEPS / CKPT_EVERY_STEPS / LR_WARMUP_STEPS  overrides (default: yaml; a short smoke needs warmup < max_steps)
#   DELTA_ALPHA       override model.config.delta_alpha (default: yaml)
#   KEYS_PATH         override model.config.delta_exact_keys_path (default: yaml)
#   RESTORE_FROM      checkpoint path to resume from (default: fresh)
#   DELTA_GATE_TOLERANCE  nats of Delta-on minus Delta-off tolerated on the docs gate (default 0.03)
#   DELTA_GATE_MIN_STEP   first step at which the gate can stop the run (default 100)
#   FREEZE_CONFIG     delta (default) | lora_delta | lora | raw JSON (see run_delta_v2.sh)
#   DELTA_ENABLED     false -> LoRA-only control arm (Delta branch off)
#   WANDB_NAME / WANDB_GROUP  as usual
# Usage: CHECKPOINT_TAG=odoo-delta-v2-smoke MAX_STEPS=3 sbatch experiments/delta_engram/nebius_delta_v2.sbatch
set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
DATA_ROOT=/mnt/data/zhihan/delta-engram
: "${CHECKPOINT_TAG:?set CHECKPOINT_TAG}"
export CONFIG_PATH="${CONFIG_PATH:-experiments/delta_engram/qwen3_8_flash_next_odoo_delta_v2.yaml}"
export CHECKPOINT_TAG MAX_STEPS NUM_EPOCHS VAL_EVERY_STEPS CKPT_EVERY_STEPS LR_WARMUP_STEPS DELTA_ALPHA KEYS_PATH RESTORE_FROM FREEZE_CONFIG DELTA_ENABLED GPUS_PER_NODE CPUS_PER_TASK MEM
export DELTA_GATE_TOLERANCE="${DELTA_GATE_TOLERANCE:-0.03}"
export DELTA_GATE_MIN_STEP="${DELTA_GATE_MIN_STEP:-100}"
export DELTA_GATE_VAL_NAME="${DELTA_GATE_VAL_NAME:-offline_docs}"
mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY=$(python3 -c "import netrc; a=netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '', end='')" 2>/dev/null || true)
fi
if [[ -n "$WANDB_API_KEY" ]]; then export WANDB_API_KEY; echo "wandb credential: present"; else echo "wandb credential: missing"; fi

echo "[$(date -u +%FT%TZ)] delta-v2 start job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST config=$CONFIG_PATH tag=$CHECKPOINT_TAG max_steps=${MAX_STEPS:-yaml} epochs=${NUM_EPOCHS:-yaml} alpha=${DELTA_ALPHA:-yaml} keys=${KEYS_PATH:-yaml} restore=${RESTORE_FROM:-none} gate_tol=$DELTA_GATE_TOLERANCE"
git -C "$CODE_ROOT" status --short --branch
export EXPECTED_CODE_COMMIT
EXPECTED_CODE_COMMIT=$(git -C "$CODE_ROOT" rev-parse HEAD)
echo "expected_code_commit=$EXPECTED_CODE_COMMIT"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((23000 + SLURM_JOB_ID % 10000))
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

declare -a STEP_PIDS=()
for NODE_RANK in "${!NODES[@]}"; do
  NODE=${NODES[$NODE_RANK]}
  case "$NODE" in
    b200-3) CONTAINER_NAME=ms-swift-ddebeb06 ;;
    *) CONTAINER_NAME=lf-gdn-smoke ;;
  esac
  echo "[$(date -u +%FT%TZ)] launch node=$NODE rank=$NODE_RANK container=$CONTAINER_NAME"
  srun --label --nodes=1 --ntasks=1 --nodelist="$NODE" \
    --gpus-per-node="${GPUS_PER_NODE:-8}" --cpus-per-task="${CPUS_PER_TASK:-32}" --mem="${MEM:-1000G}" \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
    --container-workdir=/workspace \
    bash experiments/delta_engram/run_delta_v2.sh "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT" "$SLURM_NNODES" &
  STEP_PIDS+=("$!")
done

set +e
for _ in "${STEP_PIDS[@]}"; do
  wait -n
  STATUS=$?
  if (( STATUS != 0 )); then
    kill "${STEP_PIDS[@]}" 2>/dev/null
    wait
    echo "[$(date -u +%FT%TZ)] delta-v2 FAILED status=$STATUS"
    exit "$STATUS"
  fi
done
set -e
echo "[$(date -u +%FT%TZ)] delta-v2 complete"
