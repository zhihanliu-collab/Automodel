#!/bin/bash
#SBATCH --job-name=q38-ple-ep32
#SBATCH --partition=b200
#SBATCH --nodes=4
#SBATCH --nodelist=b200-[2-5]
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

MAX_STEPS="${1:-2}"
CHECKPOINT_ENABLED="${2:-false}"
CODE_ROOT=/home/zhihan/delta-engram-automodel
DATA_ROOT=/mnt/data/zhihan/delta-engram
CONTAINER_NAME=miles-dev

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints"

echo "[$(date -u +%FT%TZ)] baseline start job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --gpus-per-node=8 --cpus-per-task=64 --mem=1000G \
  --container-name="$CONTAINER_NAME" \
  --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
  --container-workdir=/workspace \
  bash experiments/delta_engram/run_ple_baseline.sh "$MAX_STEPS" "$CHECKPOINT_ENABLED"

echo "[$(date -u +%FT%TZ)] baseline complete"
