#!/bin/bash

set +u
set -eo pipefail

MAX_STEPS="${1:-2}"
CHECKPOINT_ENABLED="${2:-false}"
GPUS_PER_NODE=8
WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((23000 + SLURM_JOB_ID % 10000))
CONFIG=examples/llm_finetune/qwen/qwen3_8_flash_next_180b_hellaswag_ep64.yaml
RUN_ROOT=/mnt/data/zhihan/delta-engram
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/ple-baseline-${SLURM_JOB_ID}"

export PYTHONPATH=/workspace
export PYTHONNOUSERSITE=1
export HF_HOME=/mnt/data/zhihan/hf_cache
export HF_DATASETS_CACHE="$RUN_ROOT/hf-datasets"
export NCCL_NET_PLUGIN=none
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "[$(date -u +%FT%TZ)] host=$(hostname) node_rank=$SLURM_NODEID world_size=$WORLD_SIZE"

exec torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$SLURM_NODEID" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m nemo_automodel.cli.app "$CONFIG" \
  --distributed.ep_size "$WORLD_SIZE" \
  --step_scheduler.global_batch_size "$WORLD_SIZE" \
  --step_scheduler.max_steps "$MAX_STEPS" \
  --step_scheduler.ckpt_every_steps "$MAX_STEPS" \
  --dataset.num_samples_limit 256 \
  --checkpoint.enabled "$CHECKPOINT_ENABLED" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
  --wandb.enable false
