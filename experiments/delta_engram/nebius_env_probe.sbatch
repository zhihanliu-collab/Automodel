#!/bin/bash
#SBATCH --job-name=q38-ple-env
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --nodelist=b200-5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
DATA_ROOT=/mnt/data/zhihan/delta-engram
HF_HOME=/mnt/data/zhihan/hf_cache
CONTAINER_NAME=miles-dev

mkdir -p "$DATA_ROOT/logs" "$DATA_ROOT/checkpoints" "$DATA_ROOT/hf-datasets"

echo "[$(date -u +%FT%TZ)] probe start job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

srun --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G \
  --container-name="$CONTAINER_NAME" \
  --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
  --container-workdir=/workspace \
  bash -lc '
    set +u
    export PYTHONPATH=/workspace
    export PYTHONNOUSERSITE=1
    export HF_HOME=/mnt/data/zhihan/hf_cache
    export HF_DATASETS_CACHE=/mnt/data/zhihan/delta-engram/hf-datasets
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    python -c "import torch, transformer_engine; import nemo_automodel; print(torch.__version__, getattr(transformer_engine, \"__version__\", \"unknown\"), nemo_automodel.__file__)"
    python -c "import nemo_automodel; from transformers import AutoConfig; c=AutoConfig.from_pretrained(\"Qwen/Qwen3.8-Flash-Next\", trust_remote_code=False); print(c.architectures, c.model_type)"
    python -c "from datasets import load_dataset; d=load_dataset(\"rowan/hellaswag\", split=\"train[:64]\"); print(d.num_rows, d.column_names)"
  '

echo "[$(date -u +%FT%TZ)] probe complete"
