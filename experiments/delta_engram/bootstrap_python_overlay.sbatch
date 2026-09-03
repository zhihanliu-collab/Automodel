#!/bin/bash
#SBATCH --job-name=q38-ple-deps
#SBATCH --partition=b200
#SBATCH --nodes=1
#SBATCH --nodelist=b200-2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/data/zhihan/delta-engram/logs/%x-%j.log

set +u
set -eo pipefail

OVERLAY=/mnt/data/zhihan/delta-engram/python-overlay
mkdir -p "$OVERLAY"

srun --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=4 --mem=16G \
  --container-name=miles-q38next-standing \
  --container-mounts=/mnt/data:/mnt/data \
  --container-workdir=/tmp \
  bash -lc '
    set +u
    set -eo pipefail
    python -m pip install --no-deps --upgrade \
      --target=/mnt/data/zhihan/delta-engram/python-overlay \
      fla-core==0.4.2 \
      flash-linear-attention==0.4.2 \
      torchao==0.14.0 \
      mlflow==3.15.1 \
      mlflow-skinny==3.15.1 \
      mlflow-tracing==3.15.1 \
      alembic==1.17.1 \
      docker==7.1.0 \
      flask==3.1.2 \
      flask-cors==6.0.1 \
      graphene==3.4.3 \
      gunicorn==23.0.0 \
      huey==2.6.0 \
      scikit-learn==1.7.2 \
      scipy==1.15.3 \
      skops==0.13.0 \
      sqlalchemy==2.0.44 \
      sqlparse==0.5.3 \
      Mako==1.3.12 \
      greenlet==3.2.4 \
      graphql-core==3.2.7 \
      graphql-relay==3.2.0 \
      joblib==1.5.2 \
      threadpoolctl==3.6.0 \
      prettytable==3.17.0 \
      databricks-sdk==0.73.0

    PYTHONPATH=/mnt/data/zhihan/delta-engram/python-overlay \
      python -c "from fla.modules.convolution import causal_conv1d; from fla.ops.cp import build_cp_context; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(\"FLA CP imports OK\")"

    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=10.0 \
      python -m pip install --no-deps --no-build-isolation --upgrade \
        --target=/mnt/data/zhihan/delta-engram/python-overlay \
        causal-conv1d==1.6.0

    PYTHONPATH=/mnt/data/zhihan/delta-engram/python-overlay \
      python -c "import torch; import causal_conv1d_cuda; from fla.modules.convolution import causal_conv1d; print(\"causal-conv1d CUDA fast path OK\")"
  '
