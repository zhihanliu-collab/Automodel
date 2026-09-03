#!/bin/bash
#SBATCH --job-name=q38-delta-ep8cp8
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
EP_SIZE="${8:-8}"
CP_SIZE="${9:-8}"
CODE_ROOT=/home/zhihan/delta-engram-automodel
DATA_ROOT=/mnt/data/zhihan/delta-engram

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints"

echo "[$(date -u +%FT%TZ)] delta start job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST rows_per_head=$DELTA_ROWS_PER_HEAD checkpoint_tag=$CHECKPOINT_TAG restore_from=$RESTORE_FROM num_epochs=$NUM_EPOCHS dataset_limit=$DATASET_LIMIT ep_size=$EP_SIZE cp_size=$CP_SIZE"
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
      "$EP_SIZE" "$CP_SIZE" &
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
