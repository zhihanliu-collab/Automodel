# Serving a Delta-Engram checkpoint with SGLang

The training checkpoint is trainable-only (Delta table + Delta PLE K/V, 32 DCP
shards). Stock SGLang (`lmsysorg/sglang:qwen38flashnext`, model
`Qwen4ExpForConditionalGeneration`) only knows the base PLE, so serving needs:

1. **`sglang_patch/`** — `models_qwen4_exp.py` and `configs_qwen4_exp.py`,
   patched copies of the image's files (upstream sglang `d91c368`; the
   `upstream_*` copies are byte-identical to the container's, see
   `delta_engram_sglang.diff`). The patch builds a second `Qwen4ExpPLELayer`
   (`delta_ple`) beside the base one when `text_config.delta_engram_enabled`,
   reads both from the same pre-injection state and adds both (as training
   did), gives the Delta branch its own short-conv state slot (mirror layer
   index, see `delta_short_conv_layer_id`), keeps the Delta hash on the plain
   torch path (int64 wrap + `torch.remainder`, matching training), and loads
   `delta_ple.*` weights/buffers through the existing PLE loader, refusing a
   layout that differs from the constructed one.
2. **`export_delta_serving_dir.py`** — reassembles the DCP shards, copies the
   frozen reader parts (norms, conv1d) from the base PLE, writes the Delta hash
   layout buffers and 128 table shards, symlinks the base snapshot, and writes
   a patched `config.json` + merged index. Prints table nonzero fraction and
   K/V relative change to compare with the training handoff.
3. **`serve_delta.sbatch` / `serve_delta_inner.sh`** — 4×B200 TP4 SGLang with
   the same flags as the base replicas; the two patched files are bind-mounted
   over the container's copies (no image rebuild).

```bash
# export (CPU step inside the sglang container; ~5 GiB output per checkpoint)
srun -p b200 -w b200-2 --cpus-per-task=8 --mem=64G --time=01:00:00 \
  --container-name=sglang-qwen38flashnext-tp4-b200-2 --container-mounts=/mnt/data:/mnt/data,$PWD:$PWD \
  python experiments/delta_engram/serving/export_delta_serving_dir.py \
    --ckpt-model-dir /mnt/data/zhihan/delta-engram/checkpoints/odoo-delta-formal-4ep-v5/epoch_0_step_329/model \
    --base-snapshot /mnt/data/zhihan/hf_cache/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540 \
    --out /mnt/data/zhihan/delta-engram/serving/s329
# serve (one job per 4-GPU endpoint; container suffix = existing enroot container on that node)
sbatch -w b200-4 experiments/delta_engram/serving/serve_delta.sbatch s329 30000 tp4-b200-4
```

Served model name: `Qwen/Qwen3.8-Flash-Next-Delta-<tag>`.
