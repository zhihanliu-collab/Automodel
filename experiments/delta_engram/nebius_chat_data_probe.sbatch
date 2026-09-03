#!/bin/bash
#SBATCH --job-name=q38-chat-data
#SBATCH --partition=b200
#SBATCH --nodelist=b200-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
RUN_ROOT=/mnt/data/zhihan/delta-engram

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/hf-datasets"

echo "[$(date -u +%FT%TZ)] chat data probe start job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G \
  --container-name=lf-gdn-smoke \
  --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
  --container-workdir=/workspace \
  bash -lc '
    set +u
    set -eo pipefail
    export PYTHONNOUSERSITE=1
    export HF_HOME=/mnt/data/zhihan/hf_cache
    export HF_DATASETS_CACHE=/mnt/data/zhihan/delta-engram/hf-datasets
    export PYTHONPATH=/workspace:/mnt/data/zhihan/delta-engram/python-overlay
    python experiments/delta_engram/probe_chat_dataset.py
  '

echo "[$(date -u +%FT%TZ)] chat data probe complete"

