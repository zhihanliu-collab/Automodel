#!/bin/bash
# Per-node entry for nebius_delta_v2.sbatch (runs inside the training container).
set +u
set -eo pipefail

NODE_RANK="${1:?node rank}"
MASTER_ADDR="${2:?master address}"
MASTER_PORT="${3:?master port}"
NNODES="${4:?node count}"
GPUS_PER_NODE=8
RUN_ROOT=/mnt/data/zhihan/delta-engram
CHECKPOINT_DIR="$RUN_ROOT/checkpoints/${CHECKPOINT_TAG:?CHECKPOINT_TAG}"
LOCAL_CACHE_ROOT="/tmp/zhihan/delta-engram-${SLURM_JOB_ID}"

export PYTHONNOUSERSITE=1
export HF_HOME=/mnt/data/zhihan/hf_cache
export HF_DATASETS_CACHE="$RUN_ROOT/hf-datasets"
export PYTHONPATH="/workspace:$RUN_ROOT/python-overlay"
export TORCHAO_SKIP_LOADING_SO_FILES=1
export NCCL_NET_PLUGIN=none
export NCCL_NET=IB
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHINDUCTOR_CACHE_DIR="$LOCAL_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$LOCAL_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$LOCAL_CACHE_ROOT/cuda"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

LOCAL_CODE_COMMIT=$(git -C /workspace rev-parse HEAD)
if [[ -n "${EXPECTED_CODE_COMMIT:-}" && "$LOCAL_CODE_COMMIT" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "Code revision mismatch on $(hostname): expected=$EXPECTED_CODE_COMMIT local=$LOCAL_CODE_COMMIT" >&2
  exit 2
fi

OVERRIDES=()
[[ -n "${MAX_STEPS:-}" ]] && OVERRIDES+=(--step_scheduler.max_steps "$MAX_STEPS" --lr_scheduler.lr_decay_steps "$MAX_STEPS")
[[ -n "${LR_WARMUP_STEPS:-}" ]] && OVERRIDES+=(--lr_scheduler.lr_warmup_steps "$LR_WARMUP_STEPS")
[[ -n "${NUM_EPOCHS:-}" ]] && OVERRIDES+=(--step_scheduler.num_epochs "$NUM_EPOCHS")
[[ -n "${VAL_EVERY_STEPS:-}" ]] && OVERRIDES+=(--step_scheduler.val_every_steps "$VAL_EVERY_STEPS")
[[ -n "${CKPT_EVERY_STEPS:-}" ]] && OVERRIDES+=(--step_scheduler.ckpt_every_steps "$CKPT_EVERY_STEPS")
[[ -n "${DELTA_ALPHA:-}" ]] && OVERRIDES+=(--model.config.delta_alpha "$DELTA_ALPHA")
[[ -n "${KEYS_PATH:-}" ]] && OVERRIDES+=(--model.config.delta_exact_keys_path "$KEYS_PATH")
[[ -n "${RESTORE_FROM:-}" ]] && OVERRIDES+=(--checkpoint.restore_from "$RESTORE_FROM")
# LoRA-only control arm: same recipe with the Delta branch switched off.
[[ -n "${DELTA_ENABLED:-}" ]] && OVERRIDES+=(--model.config.delta_engram_enabled "$DELTA_ENABLED")
# Trainable set. Default = Delta table + Delta K/V reader. With a `peft:` section the framework first
# leaves only lora_* trainable and then applies this policy, so a LoRA run must unfreeze both
# (FREEZE_CONFIG=lora_delta selects that preset).
DELTA_UNFREEZE='{"glob":"*.delta_ple.ple_embedding.ngram_embedding"},{"glob":"*.delta_ple.key_proj"},{"glob":"*.delta_ple.value_proj"}'
LORA_UNFREEZE='{"glob":"*.lora_A"},{"glob":"*.lora_B"}'
case "${FREEZE_CONFIG:-delta}" in
  delta)      FREEZE_JSON="{\"freeze_modules\":[{\"glob\":\"*\"}],\"unfreeze_modules\":[$DELTA_UNFREEZE]}" ;;
  lora_delta) FREEZE_JSON="{\"freeze_modules\":[{\"glob\":\"*\"}],\"unfreeze_modules\":[$DELTA_UNFREEZE,$LORA_UNFREEZE]}" ;;
  lora)       FREEZE_JSON="{\"freeze_modules\":[{\"glob\":\"*\"}],\"unfreeze_modules\":[$LORA_UNFREEZE]}" ;;
  *)          FREEZE_JSON="$FREEZE_CONFIG" ;;
esac

echo "[$(date -u +%FT%TZ)] host=$(hostname) node_rank=$NODE_RANK config=$CONFIG_PATH ckpt_dir=$CHECKPOINT_DIR overrides=${OVERRIDES[*]}"

exec torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m nemo_automodel.cli.app "$CONFIG_PATH" \
  --model.backend.dispatcher torch \
  --freeze_config "$FREEZE_JSON" \
  --checkpoint.checkpoint_dir "$CHECKPOINT_DIR" \
  --checkpoint.trainable_only true \
  "${OVERRIDES[@]}"
