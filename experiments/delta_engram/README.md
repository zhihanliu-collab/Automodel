# Delta-Engram experiments

This directory tracks the staged Qwen3.8-Flash-Next experiments used to establish an
upstream NeMo AutoModel baseline before introducing Delta-Engram.

## Stage 1: upstream PLE baseline

The baseline intentionally runs the unmodified upstream model and the shipped
`qwen3_8_flash_next_180b_hellaswag_ep64.yaml` recipe. Runtime overrides reduce EP64 to
EP32 on four 8-GPU B200 nodes and shorten the run without copying the upstream recipe.

Cluster code is updated only through Git. Logs and checkpoints live under
`/mnt/data/zhihan/delta-engram`; the checkout lives under
`/home/zhihan/delta-engram-automodel`.

The Slurm jobs reuse compatible node-local Pyxis environments and mount this checkout
over `/workspace`, so all NeMo AutoModel Python code comes from the recorded Git commit.
The four baseline nodes use the same Torch 2.10 / Transformer Engine 2.12 stack even
though one node's existing container has a different name. This avoids unpacking a
second large framework image into the constrained node-local `/mnt/image-storage`
volume.

The checkout's lock file requires packages newer than those cached containers. The
minimal missing set is installed with exact lock-file pins into
`/mnt/data/zhihan/delta-engram/python-overlay` and added to `PYTHONPATH`; the run disables
TorchAO's incompatible optional C++ extension because this baseline does not enable QAT.

The cached container's HybridEP extension was built without multinode support. The
baseline therefore overrides only the MoE communication dispatcher to NeMo's portable
`torch` implementation, which uses autograd-aware NCCL collectives. The Qwen model,
FSDP2 wrapping, expert implementation, PLE/Engram path, optimizer groups, and weights
remain those of the upstream recipe.

1. Bootstrap the shared Python overlay (safe to rerun):

   ```bash
   sbatch experiments/delta_engram/bootstrap_python_overlay.sbatch
   ```

2. Run the one-GPU environment and dataset probe:

   ```bash
   sbatch experiments/delta_engram/nebius_env_probe.sbatch
   ```

3. After the probe succeeds, run a two-step EP32 baseline without checkpoint I/O:

   ```bash
   sbatch experiments/delta_engram/nebius_ple_baseline.sbatch 2 false
   ```

4. A later acceptance run enables native DCP after the optimizer-step baseline is
   stable:

   ```bash
   sbatch experiments/delta_engram/nebius_ple_baseline.sbatch 4 true
   ```

The B200 and H200 partitions use different InfiniBand fabrics. These scripts always
select the `b200` partition and never use `main`.
