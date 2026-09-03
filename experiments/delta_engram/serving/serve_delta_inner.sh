#!/bin/bash
# Runs inside the sglang:qwen38flashnext container (see serve_delta.sbatch).
set +u
TAG="${1:?tag}"
PORT="${2:?port}"
MODEL_DIR=/mnt/data/zhihan/delta-engram/serving/$TAG
SGL=/sgl-workspace/sglang/python/sglang/srt
echo "=== inner delta serve on $(hostname) job=$SLURM_JOB_ID tag=$TAG port=$PORT ==="
nvidia-smi -L
# Prove the patch is what this process imports (not the image's copy).
md5sum "$SGL/models/qwen4_exp.py" "$SGL/configs/qwen4_exp.py"
grep -c "delta_ple" "$SGL/models/qwen4_exp.py" | sed 's/^/delta_ple mentions in model file: /'
[ -f "$MODEL_DIR/SERVING_MANIFEST.json" ] || { echo "FATAL: $MODEL_DIR not exported (no SERVING_MANIFEST.json)"; exit 2; }
export HF_HUB_OFFLINE=1
# Same serving recipe as the base Flash-Next replicas (serve_inner_b200.sh /
# serve_inner_tp4_c.sh); only the model directory and the served name differ.
# The flashinfer linear-attention path on Hopper asserts a float32 SSM state
# ("initial_state must be float32"), Blackwell serves it in bf16.
MAMBA_DTYPE=bfloat16
if nvidia-smi -L | head -1 | grep -q "H200"; then MAMBA_DTYPE=float32; fi
echo "mamba ssm dtype: $MAMBA_DTYPE"
exec sglang serve \
  --model-path "$MODEL_DIR" \
  --served-model-name "Qwen/Qwen3.8-Flash-Next-Delta-$TAG" \
  --tp 4 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype "$MAMBA_DTYPE" \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --max-running-requests 96 \
  --reasoning-parser auto \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port "$PORT"
