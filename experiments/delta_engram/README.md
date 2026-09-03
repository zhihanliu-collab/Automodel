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
The reader uses a configurable peak LR and the table uses a configurable multiplier with no
weight decay.

```bash
sbatch experiments/delta_engram/nebius_delta_smoke.sbatch 2 false 1000000
```

Acceptance run `4575` completed on `b200-[2-5]` from Git commit
`684bf33daef2d025b08797a1b0f9be29b109f136`. FSDP2 reported two
model-owned sharded parameters, confirming that both the Base and Delta
Engram tables retained their owner-sharded layouts instead of being wrapped a
second time. The model contained `179,537,046,400` parameters, of which only
`2,593,075,200` (`1.44%`) were trainable. Both optimizer steps and validation
completed successfully. Step losses were `3.2852` and `3.3289`; the
steady-state second step reported `141.27 tokens/s` globally and `13.86 GiB`
peak allocated memory. The run used two optimizer groups: Delta-PLE K/V at the
base LR and the Delta table at a 10x LR multiplier with zero weight decay.

For fast architecture debugging, the third argument may reduce the nominal
rows per head. Acceptance and Odoo runs must use `1000000`.

To exercise a trainable-only DCP save and resume without serializing the frozen
base model, use a stable checkpoint tag across both jobs:

```bash
sbatch experiments/delta_engram/nebius_delta_smoke.sbatch 2 true 1000000 delta-roundtrip
sbatch experiments/delta_engram/nebius_delta_smoke.sbatch 3 true 1000000 delta-roundtrip LATEST
```

`checkpoint.trainable_only=true` still loads the complete Hugging Face base
checkpoint during initialization. Training checkpoints contain only the Delta
table and Delta-PLE K/V model state, plus their optimizer and scheduler state.

The 32-GPU round-trip acceptance used jobs `4578` and `4579` from commit
`25998ad8dc5677252837455e97ad416dcef45c72`. Job `4578` wrote a `4.9 GB`
model directory and a `25 GB` optimizer directory after step 1; the frozen
`237.20 GB` base checkpoint was not duplicated. Job `4579` loaded `4.83 GB`
of Delta model state, restored optimizer/scheduler progress, resumed directly
at step 2, and completed another validation and checkpoint with exit code 0.

For the repeated-sample learnability diagnostic, the sixth and seventh Slurm
arguments control the epoch count and dataset limit. This keeps the base model
frozen while revisiting a small fixed corpus often enough to distinguish an
actual Delta optimization signal from per-batch loss noise:

```bash
sbatch experiments/delta_engram/nebius_delta_smoke.sbatch \
  40 false 1000000 delta-learnability-low-lr '' 5 256 8 1 0.00001 10 2 0.000001
```

Arguments 8 and 9 select expert and context parallelism; arguments 10--13 select
the Delta-PLE reader LR, Delta-table LR multiplier, warmup steps, and minimum LR.
Argument 14 optionally sets the validation interval. Job `4583`
verified that EP8 + CP8 (DP4) can execute complete optimizer steps, but its first
two eight-step epoch means rose from `4.1616` to `5.2330` at reader/table peak LRs
of `1e-4`/`1e-3`, so that unstable run was stopped. The lower-LR diagnostic uses
EP8 + CP1 to avoid short-sequence CP overhead. Pipeline and tensor parallelism
remain unsupported by the native Qwen3.8-Flash-Next implementation.

The lower-LR learnability run `4598` completed all 40 optimizer steps on 32 B200s
from commit `2ed0aeeda32423555cf467cc1e41692b1e66d88d`. Only the Delta table and
Delta-PLE K/V were trainable (`2,593,075,200` parameters, or `1.44%` of the
`179,537,046,400`-parameter model). Its eight-step epoch mean losses were
`3.5039`, `5.0852`, `3.9320`, `3.2251`, and `2.5751`: after a transient second-epoch
overshoot, the final epoch was 26.5% below the first. Steps 2--39 averaged
`412.74` label tokens/s globally (range `183.45`--`513.28`) with EP8 + CP1 and
gradient accumulation 1. This establishes a finite, decreasing-loss Delta-only
optimization signal; the HellaSwag repetition is a mechanism diagnostic rather
than an Odoo knowledge result.

For this short-sequence diagnostic, an eight-step topology benchmark (`4607`)
showed that EP16 + CP1 averaged `430.89` global label tokens/s over steps 2--7,
versus `396.02` tokens/s over the matching steps of EP8 + CP1: an 8.8% gain.
Peak allocated memory reported by the trainer also decreased from `14.28 GiB`
to `13.99 GiB`. Delta smoke jobs therefore default to EP16 + CP1; EP32 is not
used.

EP16 + CP2 job `4608` was rejected for this workload. It initialized all 32
ranks and entered training with gradient accumulation 2, but did not complete
step 0 after more than seven minutes in the step while most GPUs were idle. The
job was cancelled after 11 minutes rather than spending more cluster time on a
clearly inferior short-sequence configuration. This does not mean context
parallelism is unsupported: job `4583` completed 16 optimizer steps with EP8 +
CP8. It means CP should be selected from a separate long-sequence benchmark;
these HellaSwag measurements cannot determine the best 256K topology.

An attempted optional `causal-conv1d==1.6.0` CUDA fast-path install built a valid
extension but changed import order such that NeMo's custom Qwen3.8 config was not
registered with `AutoModelForCausalLM`. The extension is quarantined outside the
active overlay while that independent throughput optimization is investigated;
the accepted learnability run uses FLA's PyTorch convolution fallback.
