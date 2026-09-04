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

# Build the Odoo corpus cache (docs + chatter threads + agentic memory edits +
# trajectories) and the exact-dictionary Delta keys derived from its TRAIN split.
# Usage: sbatch [-w b200-N] nebius_odoo_corpus_preprocess.sbatch <output-dir> [workers] [docs-repeat] [agent-jsonl]
# v3 (2026-09-04): the trajectory source is jian's v3-derived export. The v1-derived
# ccsdk960_flashnext_131k.jsonl (md5 78a24e1d, used for cache qwen38-131k-v2) split each
# step's N parallel tool calls into N assistant messages, leaving 36% of assistant turns
# with an empty think and no multi-call turns at all (mailbox thread af0fa15a).
set +u
set -eo pipefail

CODE_ROOT=/home/zhihan/delta-engram-automodel
CORPUS_ROOT=/mnt/data/zhihan/delta-engram/corpus
SEED_WS=/mnt/data/zhihan/reviewer_reduce100/seed_ws_opus5_v6/workspace
OUTPUT_DIR="${1:-$CORPUS_ROOT/qwen38-131k-v3}"
WORKERS="${2:-32}"
DOCS_REPEAT="${3:-16}"
AGENT_JSONL="${4:-/mnt/data/jian/datasets/ccsdk_flashnext/ccsdk960_v3_flashnext_131k.jsonl}"
AGENT_MD5_EXPECTED="${AGENT_MD5_EXPECTED:-c022fcc3956f1d09df9b51f38ac49901}"

echo "[$(date -u +%FT%TZ)] host=$(hostname) job=$SLURM_JOB_ID output=$OUTPUT_DIR workers=$WORKERS docs_repeat=$DOCS_REPEAT agent_jsonl=$AGENT_JSONL"
AGENT_MD5=$(md5sum "$AGENT_JSONL" | cut -d" " -f1)
echo "agent_jsonl md5=$AGENT_MD5 (expected $AGENT_MD5_EXPECTED)"
if [[ -n "$AGENT_MD5_EXPECTED" && "$AGENT_MD5" != "$AGENT_MD5_EXPECTED" ]]; then echo "FATAL: trajectory export md5 mismatch"; exit 3; fi
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
        --agent-jsonl $AGENT_JSONL \
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
