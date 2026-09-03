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

The Slurm jobs reuse the node-local `miles-dev` Pyxis environment and mount this
checkout over `/workspace`, so all NeMo AutoModel Python code comes from the recorded
Git commit. This avoids unpacking a second large framework image into the constrained
node-local `/mnt/image-storage` volume.

1. Run the one-GPU environment and dataset probe:

   ```bash
   sbatch experiments/delta_engram/nebius_env_probe.sbatch
   ```

2. After the probe succeeds, run a two-step EP32 baseline without checkpoint I/O:

   ```bash
   sbatch experiments/delta_engram/nebius_ple_baseline.sbatch 2 false
   ```

3. A later acceptance run enables native DCP after the optimizer-step baseline is
   stable:

   ```bash
   sbatch experiments/delta_engram/nebius_ple_baseline.sbatch 4 true
   ```

The B200 and H200 partitions use different InfiniBand fabrics. These scripts always
select the `b200` partition and never use `main`.
