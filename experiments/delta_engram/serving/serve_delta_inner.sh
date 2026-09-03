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
# Same serving recipe as the base Flash-Next B200 replicas (serve_inner_b200.sh);
# only the model directory and the served name differ.
exec sglang serve \
  --model-path "$MODEL_DIR" \
  --served-model-name "Qwen/Qwen3.8-Flash-Next-Delta-$TAG" \
  --tp 4 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --max-running-requests 96 \
  --reasoning-parser auto \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port "$PORT"
