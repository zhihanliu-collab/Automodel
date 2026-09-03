#!/bin/bash
#SBATCH --job-name=q38-delta
#SBATCH --partition=b200
#SBATCH --nodes=4
#SBATCH --nodelist=b200-[2-5]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

MAX_STEPS="${1:-2}"
CHECKPOINT_ENABLED="${2:-false}"
DELTA_ROWS_PER_HEAD="${3:-1000000}"
CHECKPOINT_TAG="${4:-delta-smoke-${SLURM_JOB_ID}}"
RESTORE_FROM="${5:-}"
NUM_EPOCHS="${6:-1}"
DATASET_LIMIT="${7:-256}"
EP_SIZE="${8:-16}"
CP_SIZE="${9:-1}"
READER_LR="${10:-0.00001}"
TABLE_LR_MULT="${11:-10.0}"
LR_WARMUP_STEPS="${12:-2}"
MIN_LR="${13:-0.000001}"
VAL_EVERY_STEPS="${14:-100}"
LR_DECAY_STEPS="${15:-$MAX_STEPS}"
CONFIG_PATH="${16:-examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml}"
CODE_ROOT=/home/zhihan/delta-engram-automodel
DATA_ROOT=/mnt/data/zhihan/delta-engram

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints"

# Slurm/pyxis does not expose the login node's ~/.netrc inside the container.
# Resolve the W&B credential in the host job environment and let srun pass it
# through without writing the secret to the job script or logs.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  WANDB_API_KEY=$(python3 -c "import netrc; a=netrc.netrc().authenticators('api.wandb.ai'); print(a[2] if a else '', end='')" 2>/dev/null || true)
fi
if [[ -n "$WANDB_API_KEY" ]]; then
  export WANDB_API_KEY
  echo "wandb credential: present"
else
  echo "wandb credential: missing"
fi

echo "[$(date -u +%FT%TZ)] delta start job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST rows_per_head=$DELTA_ROWS_PER_HEAD checkpoint_tag=$CHECKPOINT_TAG restore_from=$RESTORE_FROM num_epochs=$NUM_EPOCHS dataset_limit=$DATASET_LIMIT ep_size=$EP_SIZE cp_size=$CP_SIZE reader_lr=$READER_LR table_lr_mult=$TABLE_LR_MULT lr_warmup_steps=$LR_WARMUP_STEPS min_lr=$MIN_LR val_every_steps=$VAL_EVERY_STEPS lr_decay_steps=$LR_DECAY_STEPS config_path=$CONFIG_PATH"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

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
    --gpus-per-node=8 --cpus-per-task=32 --mem=1000G \
    --container-name="$CONTAINER_NAME" \
    --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
    --container-workdir=/workspace \
    bash experiments/delta_engram/run_delta_smoke.sh \
      "$MAX_STEPS" "$CHECKPOINT_ENABLED" "$NODE_RANK" "$MASTER_ADDR" "$MASTER_PORT" "$SLURM_NNODES" \
      "$DELTA_ROWS_PER_HEAD" "$CHECKPOINT_TAG" "$RESTORE_FROM" "$NUM_EPOCHS" "$DATASET_LIMIT" \
      "$EP_SIZE" "$CP_SIZE" "$READER_LR" "$TABLE_LR_MULT" "$LR_WARMUP_STEPS" "$MIN_LR" \
      "$VAL_EVERY_STEPS" "$LR_DECAY_STEPS" "$CONFIG_PATH" &
  STEP_PIDS+=("$!")
done

set +e
for _ in "${STEP_PIDS[@]}"; do
  wait -n
  STATUS=$?
  if (( STATUS != 0 )); then
    kill "${STEP_PIDS[@]}" 2>/dev/null
    wait
    exit "$STATUS"
  fi
done
set -e

echo "[$(date -u +%FT%TZ)] delta complete"
