# Delta-Engram Tulu chat-format learnability probe

## Run

- Date: 2026-09-03 UTC
- Slurm job: `4634`
- Git revision: `38f0a29711e25fc2b0a50bfa0e571662d0a80cf1`
- Hardware: 32 B200 GPUs (`b200-[2-5]`)
- Topology: EP16, CP1 (no EP32)
- Model: `Qwen/Qwen3.8-Flash-Next`, language-only
- Data: `allenai/tulu-3-sft-mixture`, 256 shuffled training examples and
  64 held-out validation examples
- Formatting: the checkpoint's native chat template, with only the final
  assistant turn supervised (`mask_history: true`)
- Sequence length: up to 4096 tokens; local/global batch sizes 1/32
- Trainable parameters: 2,593,075,200 (1.44% of 179,537,046,400 total)
- Trainable tensors: Delta N-gram embeddings and Delta-PLE `key_proj` /
  `value_proj` only
- LR: Delta-PLE K/V `3e-6`; Delta table `3e-5`; 8-step warmup followed by
  cosine decay to `3e-7` (reader LR)
- Schedule: 20 epochs, 8 steps/epoch, 160 total steps; validation every epoch

The preceding one-step format probe (job `4633`) measured an initial train
loss of 1.1206 and a post-update validation loss of 0.6224. This replaced the
raw HellaSwag completion probe, whose 3.x loss is not comparable to native
chat SFT loss.

## Results

| Epoch | Mean train loss | Validation loss |
|---:|---:|---:|
| 0 | 0.740226 | 0.517516 |
| 1 | 0.583846 | 0.474459 |
| 2 | 0.602719 | 0.462824 |
| 3 | 0.478101 | 0.455623 |
| **4** | **0.375625** | **0.454940** |
| 5 | 0.266141 | 0.469188 |
| 6 | 0.181419 | 0.478204 |
| 7 | 0.115317 | 0.497494 |
| 8 | 0.075290 | 0.520534 |
| 9 | 0.049583 | 0.519846 |
| 10 | 0.029167 | 0.539581 |
| 11 | 0.018562 | 0.539587 |
| 12 | 0.010950 | 0.548542 |
| 13 | 0.007743 | 0.561629 |
| 14 | 0.005283 | 0.575336 |
| 15 | 0.003067 | 0.575546 |
| 16 | 0.002794 | 0.576074 |
| 17 | 0.001527 | 0.578715 |
| 18 | 0.001258 | 0.582002 |
| 19 | 0.000692 | 0.585634 |

The first train step was 1.0930 and the final train step was 0.000305. The
best held-out loss was 0.454940 after epoch 4 (step 39), a 12.1% reduction
from the first epoch-end validation loss. Validation then rose while training
loss continued toward zero, showing ordinary small-dataset overfitting rather
than an inability to optimize the Delta path.

After excluding the compile-heavy first step, mean/median throughput was
4542/4856 input tokens/s globally, with a measured maximum of 7169 input
tokens/s. The run processed 3,693,551 input tokens after step 0 and completed
successfully in 22m38s including cold initialization, graph compilation,
validation, and final checkpointing.

## Artifacts

- Log: `/mnt/data/zhihan/delta-engram/logs/q38-delta-4634.log`
- Metrics: `/mnt/data/zhihan/delta-engram/checkpoints/tulu-chat-20ep/training.jsonl`
  and `validation.jsonl`
- Final trainable-only checkpoint:
  `/mnt/data/zhihan/delta-engram/checkpoints/tulu-chat-20ep/epoch_19_step_159`
- Checkpoint size: 4.9 GB model shards and 25 GB optimizer state

Only the final checkpoint was saved in this probe. Consequently NeMo's
`LOWEST_VAL` symlink points at the final checkpoint even though the metric
history shows that epoch 4 was best. Future learning runs should save at every
validation boundary or implement metric-aware retention.

## Conclusion

This run establishes the narrow mechanism claim: with the base Engram, base
PLE, backbone, MoE, router, token embeddings, and LM head frozen, the Delta
N-gram table plus Delta-PLE K/V can receive gradients and drive native
chat-format causal loss sharply downward under FSDP2/owner sharding.

It does not establish Odoo knowledge acquisition or capability retention.
Those require source-level Odoo train/validation/test splits, knowledge QA
evaluation, Base/Delta-off/Delta-on comparisons, and agent/tool-call retention
tests. The 256-example probe also demonstrates that epoch count is not a safe
budgeting rule for the full experiment; use sampled token budget and held-out
early stopping instead.
