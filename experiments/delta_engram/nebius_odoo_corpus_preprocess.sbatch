#!/bin/bash
#SBATCH --job-name=odoo-delta-data
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

# Build the v2 Odoo corpus cache (docs + chatter threads + agentic memory edits +
# trajectories) and the exact-dictionary Delta keys derived from its TRAIN split.
# Usage: sbatch [-w b200-N] nebius_odoo_corpus_preprocess.sbatch <output-dir> [workers] [docs-repeat]
set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
CORPUS_ROOT=/mnt/data/zhihan/delta-engram/corpus
SEED_WS=/mnt/data/zhihan/reviewer_reduce100/seed_ws_opus5_v6/workspace
OUTPUT_DIR="${1:-$CORPUS_ROOT/qwen38-131k-v2}"
WORKERS="${2:-32}"
DOCS_REPEAT="${3:-16}"

echo "[$(date -u +%FT%TZ)] host=$(hostname) job=$SLURM_JOB_ID output=$OUTPUT_DIR workers=$WORKERS docs_repeat=$DOCS_REPEAT"
git -C "$CODE_ROOT" status --short --branch
git -C "$CODE_ROOT" rev-parse HEAD

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=512G \
  --container-name=lf-gdn-smoke \
  --container-mounts="$CODE_ROOT:/workspace,/mnt/data:/mnt/data" \
  --container-workdir=/workspace \
  bash -lc "export PYTHONNOUSERSITE=1 HF_HOME=/mnt/data/zhihan/hf_cache; \
    if [[ ! -f $OUTPUT_DIR/manifest.json ]]; then \
      python -m experiments.delta_engram.odoo_corpus \
        --handbook $SEED_WS/COMPANY-HANDBOOK.md \
        --tutorial $SEED_WS/ODOO-MCP-TUTORIAL.md \
        --docs-repeat $DOCS_REPEAT \
        --bills-jsonl $CORPUS_ROOT/raw/odoo_offline_bills_20260817.jsonl \
        --memory-samples-jsonl $CORPUS_ROOT/raw/memory_edit_samples_v2.jsonl \
        --agent-jsonl /mnt/data/jian/datasets/ccsdk_flashnext/ccsdk960_flashnext_131k.jsonl \
        --output-dir $OUTPUT_DIR \
        --max-context 131072 \
        --validation-fraction 0.1 \
        --seed 20260903 \
        --workers $WORKERS; \
    else \
      echo Reusing existing cache $OUTPUT_DIR; \
    fi && \
    python experiments/delta_engram/serving/build_exact_hot_keys.py \
      --cache $OUTPUT_DIR --split train --out $OUTPUT_DIR/delta_exact_keys_train.pt"

echo "[$(date -u +%FT%TZ)] preprocessing complete"
