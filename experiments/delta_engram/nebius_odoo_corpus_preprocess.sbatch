#!/bin/bash
#SBATCH --job-name=odoo-delta-data
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --nodelist=b200-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
CORPUS_ROOT=/mnt/data/zhihan/delta-engram/corpus
OUTPUT_DIR="${1:-$CORPUS_ROOT/qwen38-131k-v1}"
WORKERS="${2:-32}"

echo "[$(date -u +%FT%TZ)] host=$(hostname) job=$SLURM_JOB_ID output=$OUTPUT_DIR workers=$WORKERS"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=512G \
  --container-name=lf-gdn-smoke \
  --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
  --container-workdir=/workspace \
  bash -lc "export PYTHONNOUSERSITE=1 HF_HOME=/mnt/data/zhihan/hf_cache; \
    python -m experiments.delta_engram.odoo_corpus \
      --bills-jsonl $CORPUS_ROOT/raw/odoo_offline_bills_20260817.jsonl \
      --memory-trace $CORPUS_ROOT/raw/opus5_memory_trace.jsonl \
      --agent-jsonl /mnt/data/jian/datasets/ccsdk_flashnext/ccsdk960_flashnext_131k.jsonl \
      --output-dir $OUTPUT_DIR \
      --max-context 131072 \
      --validation-fraction 0.1 \
      --seed 20260903 \
      --workers $WORKERS"

echo "[$(date -u +%FT%TZ)] preprocessing complete"
