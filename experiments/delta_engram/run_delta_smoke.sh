#!/bin/bash

set +u
set -eo pipefail

MAX_STEPS="${1:-2}"
CHECKPOINT_ENABLED="${2:-false}"
NODE_RANK="${3:?node rank is required}"
MASTER_ADDR="${4:?master address is required}"
MASTER_PORT="${5:?master port is required}"
NNODES="${6:?node count is required}"
DELTA_ROWS_PER_HEAD="${7:-1000000}"
CHECKPOINT_TAG="${8:-delta-smoke-${SLURM_JOB_ID}}"
RESTORE_FROM="${9:-}"
NUM_EPOCHS="${10:-1}"
DATASET_LIMIT="${11:-256}"
EP_SIZE="${12:-16}"
CP_SIZE="${13:-1}"
READER_LR="${14:-0.00001}"
TABLE_LR_MULT="${15:-10.0}"
LR_WARMUP_STEPS="${16:-2}"
MIN_LR="${17:-0.000001}"
VAL_EVERY_STEPS="${18:-100}"
LR_DECAY_STEPS="${19:-$MAX_STEPS}"
CONFIG_PATH="${20:-examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml}"
GLOBAL_BATCH_SIZE="${21:-}"
CKPT_EVERY_STEPS="${22:-$MAX_STEPS}"
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
RUN_ROOT=/mnt/data/zhihan/delta-engram
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/$CHECKPOINT_TAG"
LOCAL_CACHE_ROOT="/tmp/zhihan/delta-engram-${SLURM_JOB_ID}"

export PYTHONNOUSERSITE=1
export HF_HOME=/mnt/data/zhihan/hf_cache
export HF_DATASETS_CACHE="$RUN_ROOT/hf-datasets"
export PYTHONPATH="/workspace:$RUN_ROOT/python-overlay"
export TORCHAO_SKIP_LOADING_SO_FILES=1
export NCCL_NET_PLUGIN=none
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

if (( WORLD_SIZE % EP_SIZE != 0 || WORLD_SIZE % CP_SIZE != 0 )); then
  echo "EP_SIZE=$EP_SIZE and CP_SIZE=$CP_SIZE must each divide WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi

# One sample is distributed over CP_SIZE ranks.  The maximum batch with no
# gradient accumulation is therefore the data-parallel replica count, not the
# raw world size.
if [[ -z "$GLOBAL_BATCH_SIZE" ]]; then
  GLOBAL_BATCH_SIZE=$((WORLD_SIZE / CP_SIZE))
fi
if (( GLOBAL_BATCH_SIZE < 1 )); then
  echo "GLOBAL_BATCH_SIZE must be positive, got $GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

echo "[$(date -u +%FT%TZ)] host=$(hostname) node_rank=$NODE_RANK world_size=$WORLD_SIZE ep_size=$EP_SIZE cp_size=$CP_SIZE global_batch_size=$GLOBAL_BATCH_SIZE delta_rows_per_head=$DELTA_ROWS_PER_HEAD checkpoint_tag=$CHECKPOINT_TAG restore_from=$RESTORE_FROM num_epochs=$NUM_EPOCHS dataset_limit=$DATASET_LIMIT reader_lr=$READER_LR table_lr_mult=$TABLE_LR_MULT lr_warmup_steps=$LR_WARMUP_STEPS min_lr=$MIN_LR val_every_steps=$VAL_EVERY_STEPS ckpt_every_steps=$CKPT_EVERY_STEPS lr_decay_steps=$LR_DECAY_STEPS config_path=$CONFIG_PATH"

RESTORE_ARGS=()
if [[ -n "$RESTORE_FROM" ]]; then
  RESTORE_ARGS+=(--checkpoint.restore_from "$RESTORE_FROM")
fi

DATASET_ARGS=()
if [[ "$CONFIG_PATH" == *hellaswag* ]]; then
  DATASET_ARGS+=(--dataset.num_samples_limit "$DATASET_LIMIT")
fi

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m nemo_automodel.cli.app "$CONFIG_PATH" \
  --model.config.delta_engram_enabled true \
  --model.config.delta_ngram_vocab_size_per_head "$DELTA_ROWS_PER_HEAD" \
  --model.backend.dispatcher torch \
  --distributed.ep_size "$EP_SIZE" \
  --distributed.cp_size "$CP_SIZE" \
  --freeze_config '{"freeze_modules":[{"glob":"*"}],"unfreeze_modules":[{"glob":"*.delta_ple.ple_embedding.ngram_embedding"},{"glob":"*.delta_ple.key_proj"},{"glob":"*.delta_ple.value_proj"}]}' \
  --optimizer.lr "$READER_LR" \
  --optimizer.weight_decay 0.0 \
  --optimizer.param_group_overrides "[{\"pattern\":\"\\\\.delta_ple\\\\.ple_embedding\\\\.ngram_embedding\\\\.weight$\",\"lr_mult\":$TABLE_LR_MULT,\"wd_mult\":0.0}]" \
  --lr_scheduler.lr_warmup_steps "$LR_WARMUP_STEPS" \
  --lr_scheduler.lr_decay_steps "$LR_DECAY_STEPS" \
  --lr_scheduler.min_lr "$MIN_LR" \
  --step_scheduler.global_batch_size "$GLOBAL_BATCH_SIZE" \
  --step_scheduler.max_steps "$MAX_STEPS" \
  --step_scheduler.num_epochs "$NUM_EPOCHS" \
  --step_scheduler.ckpt_every_steps "$CKPT_EVERY_STEPS" \
  --step_scheduler.val_every_steps "$VAL_EVERY_STEPS" \
  "${DATASET_ARGS[@]}" \
  --checkpoint.enabled "$CHECKPOINT_ENABLED" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
  --checkpoint.trainable_only true \
  "${RESTORE_ARGS[@]}"
