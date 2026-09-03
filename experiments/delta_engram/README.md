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

   Acceptance run `4565` completed on `b200-[2-5]` from Git commit
   `37136f24995ac1705fbd495be96f1b84f71cb40d`: both optimizer steps and
   validation exited successfully. Step losses were `3.2910` and `3.3383`;
   the steady-state second step reported `131.02 tokens/s` globally and
   `90.02 GiB` peak allocated memory. The first step took 75.52 seconds
   including compilation, while the second took about 9.4 seconds.

4. A later acceptance run enables native DCP after the optimizer-step baseline is
   stable:

   ```bash
   sbatch experiments/delta_engram/nebius_ple_baseline.sbatch 4 true
   ```

The B200 and H200 partitions use different InfiniBand fabrics. These scripts always
select the `b200` partition and never use `main`.

## Stage 2: Delta-Engram smoke

`nebius_delta_smoke.sbatch` enables the append-only branch with the proposal's
default one-million nominal rows per head. It freezes the complete base model
and unfreezes only the Delta table plus Delta-PLE `key_proj`/`value_proj`.
The reader uses `1e-4` peak LR and the table uses a 10x multiplier with no
weight decay.

```bash
sbatch experiments/delta_engram/nebius_delta_smoke.sbatch 2 false 1000000
```

For fast architecture debugging, the third argument may reduce the nominal
rows per head. Acceptance and Odoo runs must use `1000000`.
