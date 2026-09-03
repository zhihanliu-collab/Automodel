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
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
CONFIG=examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml
RUN_ROOT=/mnt/data/zhihan/delta-engram
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/delta-smoke-${SLURM_JOB_ID}"
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

echo "[$(date -u +%FT%TZ)] host=$(hostname) node_rank=$NODE_RANK world_size=$WORLD_SIZE delta_rows_per_head=$DELTA_ROWS_PER_HEAD"

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m nemo_automodel.cli.app "$CONFIG" \
  --model.config.delta_engram_enabled true \
  --model.config.delta_ngram_vocab_size_per_head "$DELTA_ROWS_PER_HEAD" \
  --model.backend.dispatcher torch \
  --distributed.ep_size "$WORLD_SIZE" \
  --freeze_config '{"freeze_modules":[{"glob":"*"}],"unfreeze_modules":[{"glob":"*.delta_ple.ple_embedding.ngram_embedding"},{"glob":"*.delta_ple.key_proj"},{"glob":"*.delta_ple.value_proj"}]}' \
  --optimizer.lr 0.0001 \
  --optimizer.weight_decay 0.0 \
  --optimizer.param_group_overrides '[{"pattern":"\\.delta_ple\\.ple_embedding\\.ngram_embedding\\.weight$","lr_mult":10.0,"wd_mult":0.0}]' \
  --lr_scheduler.min_lr 0.00001 \
  --step_scheduler.global_batch_size "$WORLD_SIZE" \
  --step_scheduler.max_steps "$MAX_STEPS" \
  --step_scheduler.ckpt_every_steps "$MAX_STEPS" \
  --dataset.num_samples_limit 256 \
  --checkpoint.enabled "$CHECKPOINT_ENABLED" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
  --checkpoint.trainable_only true \
  --wandb.enable false
