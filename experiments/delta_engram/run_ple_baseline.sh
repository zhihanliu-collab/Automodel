#!/bin/bash

set +u
set -eo pipefail

MAX_STEPS="${1:-2}"
CHECKPOINT_ENABLED="${2:-false}"
NODE_RANK="${3:?node rank is required}"
MASTER_ADDR="${4:?master address is required}"
MASTER_PORT="${5:?master port is required}"
NNODES="${6:?node count is required}"
GPUS_PER_NODE=8
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
CONFIG=examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml
RUN_ROOT=/mnt/data/zhihan/delta-engram
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/ple-baseline-${SLURM_JOB_ID}"
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

echo "[$(date -u +%FT%TZ)] host=$(hostname) node_rank=$NODE_RANK world_size=$WORLD_SIZE"

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m nemo_automodel.cli.app "$CONFIG" \
  --model.backend.dispatcher torch \
  --distributed.ep_size "$WORLD_SIZE" \
  --step_scheduler.global_batch_size "$WORLD_SIZE" \
  --step_scheduler.max_steps "$MAX_STEPS" \
  --step_scheduler.ckpt_every_steps "$MAX_STEPS" \
  --dataset.num_samples_limit 256 \
  --checkpoint.enabled "$CHECKPOINT_ENABLED" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb.enable false
