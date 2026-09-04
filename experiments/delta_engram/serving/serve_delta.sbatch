#!/bin/bash
#SBATCH --job-name=q38delta-serve
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/data/zhihan/delta-engram/serving/logs/q38delta-serve-%N-%j.log

# Serve a Delta-Engram Flash-Next checkpoint with the patched SGLang model.
# Usage: sbatch -w <node> serve_delta.sbatch <tag: s164|s329> <port> <container-suffix>
# <container-suffix> names an EXISTING enroot container on that node built from
# docker://lmsysorg/sglang:qwen38flashnext (e.g. tp4-b200-2 / tp4b-b200-2, one
# per port), so no image is pulled; the two patched files are bind-mounted over
# the container's copies, the rest of the image is untouched.
set +u
TAG="${1:?s164|s329}"
PORT="${2:?port}"
CNAME="${3:?existing container suffix, e.g. tp4-b200-2}"
REPO=/home/zhihan/delta-engram-automodel
SERV=$REPO/experiments/delta_engram/serving
SGL=/sgl-workspace/sglang/python/sglang/srt
RUN_DIR=/mnt/data/zhihan/delta-engram/serving
mkdir -p "$RUN_DIR/logs"
echo "=== $(date -u +%F' '%T' '%Z) job=$SLURM_JOB_ID node=$SLURM_JOB_NODELIST tag=$TAG port=$PORT cname=$CNAME repo=$(git -C $REPO rev-parse --short HEAD) ==="
# Cluster rule (platform-nebius-b200-slurm, memory pyxis-container-name-must-be-constant): only the
# fixed container names may be used -- a new name unpacks another 20-40GB image into node-local
# /mnt/image-storage and, once that disk is full, every docker sandbox on the node fails.
case "$CNAME" in tp4-b200-[0-9]|tp4b-b200-[0-9]|tp4c|tp4|tp4b) ;; *) echo "FATAL: container suffix '$CNAME' is not in the fixed set (tp4-b200-N, tp4b-b200-N, tp4c, tp4, tp4b)"; exit 3;; esac
USE=$(df --output=pcent /mnt/image-storage 2>/dev/null | tail -1 | tr -dc 0-9)
if [ -n "$USE" ] && [ "$USE" -ge 90 ]; then echo "FATAL: /mnt/image-storage at ${USE}% on $(hostname); free space (enroot remove MY stale pyxis_* / docker volume prune) before serving"; df -h /mnt/image-storage; exit 3; fi

srun --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --container-image=docker://lmsysorg/sglang:qwen38flashnext \
  --container-name="sglang-qwen38flashnext-$CNAME" \
  --container-mounts=/mnt/data:/mnt/data,$REPO:$REPO,$SERV/sglang_patch/models_qwen4_exp.py:$SGL/models/qwen4_exp.py,$SERV/sglang_patch/configs_qwen4_exp.py:$SGL/configs/qwen4_exp.py \
  bash "$SERV/serve_delta_inner.sh" "$TAG" "$PORT"
rc=$?
echo "[$(date -u +%F' '%T' '%Z)] srun exited rc=$rc" >> "$RUN_DIR/EXITED_${TAG}_${SLURM_JOB_NODELIST}_${PORT}"
exit $rc
