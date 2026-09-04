# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Delta-Engram training recipe with scale and overfitting diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import math
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from experiments.delta_engram.generation_probe import _difference_stats, _local_tensor, _tensor_stats
from experiments.delta_engram.odoo_corpus import ALL_SOURCES
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.shared.import_utils import safe_import

_, wandb = safe_import("wandb")


class DeltaGateFailed(RuntimeError):
    """Raised when Delta-on validation loss exceeds Delta-off on the gate bucket."""


class DeltaDiagnosticsTrainingRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Add low-frequency Delta scale diagnostics to the standard train recipe.

    Parameter scans and activation reductions run only when validation runs.
    Ordinary steps retain the upstream recipe's logging and compute path.

    Delta-on/off gate: the validation loader named by ``DELTA_GATE_VAL_NAME``
    (default ``offline_docs``) is scored twice at every validation step, once
    normally and once with the Delta branch switched off. From step
    ``DELTA_GATE_MIN_STEP`` (default 100) on, the first run whose on-minus-off
    loss exceeds ``DELTA_GATE_TOLERANCE`` (default 0.03 nats) logs the gap and
    stops training with :class:`DeltaGateFailed`: a Delta that makes
    text it was trained on *less* likely than the frozen base is the failure
    mode that cost the first checkpoint 8 of 100 tasks.
    """

    train_loss_ema_decay = 0.9

    def setup(self) -> None:
        super().setup()
        self._checkpoint_lora_like_delta()
        self._check_trainable_order_across_ranks()
        model = self.model_parts[0]
        language_model = model.model.language_model
        ple_layer_index = str(int(model.config.text_config.ple_layer_ids[0]) - 1)
        layer = language_model.layers[ple_layer_index]
        self._diag_base_ple = layer.ple
        self._diag_delta_ple = layer.delta_ple
        if self._diag_base_ple is None or self._diag_delta_ple is None:
            raise RuntimeError("Delta diagnostics require both Base PLE and Delta-PLE")

        self._diag_capture_enabled = False
        self._diag_activation_sums: dict[str, torch.Tensor] = {}
        self._diag_activation_maxima: dict[str, torch.Tensor] = {}
        self._diag_activation_counts: dict[str, int] = defaultdict(int)
        self._diag_train_loss_ema: float | None = None
        self._diag_source_loss_ema: dict[str, float] = {}
        self._diag_source_loss_numerators: torch.Tensor | None = None
        self._diag_source_label_tokens: torch.Tensor | None = None
        self._diag_source_samples: torch.Tensor | None = None
        self._diag_latest_val_losses: dict[str, float] = {}
        self._diag_latest_val_token_counts: dict[str, int] = {}
        self._diag_latest_val_step = -1
        self._diag_best_val_losses: dict[str, float] = {}
        self._diag_best_val_steps: dict[str, int] = {}
        self._diag_parameter_metrics_step = -1
        self._diag_parameter_metrics: dict[str, float] = {}
        self._diag_table_module = self._diag_delta_ple.ple_embedding.ngram_embedding
        self._diag_delta_layer = layer
        self._gate_val_name = os.environ.get("DELTA_GATE_VAL_NAME", "offline_docs")
        self._gate_tolerance = float(os.environ.get("DELTA_GATE_TOLERANCE", "0.03"))
        # A freshly initialized Delta (zero table, copied reader) adds a small
        # random perturbation before it has learned anything: the 3-step smoke
        # measured +0.014 nats at step 1. The gate is armed only from this step on.
        self._gate_min_step = int(os.environ.get("DELTA_GATE_MIN_STEP", "100"))
        self._gate_pending: dict[str, float] | None = None
        self._gate_failed_message: str | None = None
        ngram_reader = self._diag_delta_ple.ple_embedding
        self._diag_head_sizes = [int(value) for value in ngram_reader.ngram_heads_vocab_sizes.tolist()]
        self._diag_head_offsets = [int(value) for value in ngram_reader.ngram_heads_offsets.tolist()]
        self._diag_step0_touched_bitmap: torch.Tensor | None = None
        self._diag_step0_grad_bitmap: torch.Tensor | None = None
        access_path = os.environ.get("DELTA_ACCESS_COUNTS_PATH")
        self._diag_access_counts = (
            np.memmap(access_path, mode="r", dtype=np.uint32) if access_path and os.path.exists(access_path) else None
        )
        self._diag_hook_handles = [
            self._diag_base_ple.register_forward_hook(self._make_activation_hook("base_output")),
            self._diag_delta_ple.register_forward_hook(self._make_activation_hook("delta_output", capture_input=True)),
            self._diag_table_module.register_forward_pre_hook(self._capture_step0_touched_rows),
        ]
        if self._diag_table_module.weight.requires_grad:  # frozen in LoRA-only controls (FREEZE_CONFIG=lora)
            self._diag_hook_handles.append(self._diag_table_module.weight.register_hook(self._capture_step0_row_grads))

    def _checkpoint_lora_like_delta(self) -> None:
        """Save Delta+LoRA runs the way the Delta-only v2 run saved (trainable-only DCP with FQN keys).

        With a ``peft:`` section the framework marks the checkpoint ``is_peft`` and, under expert
        parallelism, switches the optimizer state to the native ``optimizer.state_dict()`` whose
        entries are keyed by *position* in the param groups. Job 5068 (2026-09-04) died at the
        first save with DCP "Failed to validate global plan": the same position held a different
        LoRA tensor on ranks 24-31 than on the other ranks. The FQN-keyed path used by the v2 run
        (is_peft=False, checkpoint.trainable_only=True) does not depend on parameter order, and the
        export merges lora_A/lora_B from those DCP shards directly, so the HF PEFT adapter layout
        is not needed. DELTA_CKPT_KEEP_PEFT=1 keeps the framework default.
        """
        if os.environ.get("DELTA_CKPT_KEEP_PEFT", "0") == "1" or self.peft_config is None:
            return
        config = self.checkpointer.config
        if getattr(config, "is_peft", False):
            config.is_peft = False
            if dist.get_rank() == 0:
                print("[DIAG] checkpoint.is_peft forced False: Delta+LoRA saved as trainable-only DCP (FQN keys)", flush=True)

    def _check_trainable_order_across_ranks(self) -> None:
        """Log whether every rank sees the same trainable-parameter FQN sequence (order and set)."""
        import hashlib

        names = [name for name, param in self.model_parts[0].named_parameters() if param.requires_grad]
        digest = hashlib.sha1("\n".join(names).encode()).hexdigest()[:12]
        gathered: list[tuple[str, int, list[str]]] = [None] * dist.get_world_size()  # type: ignore[list-item]
        dist.all_gather_object(gathered, (digest, len(names), names if dist.get_rank() % 8 == 0 else []))
        if dist.get_rank() != 0:
            return
        digests = sorted({d for d, _, _ in gathered})
        print(f"[DIAG] trainable params per rank: n={len(names)} order_digests={digests}", flush=True)
        if len(digests) > 1:
            reference = gathered[0][2]
            for rank, (d, n, other) in enumerate(gathered):
                if d == gathered[0][0] or not other:
                    continue
                first = next((i for i, (a, b) in enumerate(zip(reference, other)) if a != b), min(len(reference), len(other)))
                print(f"[DIAG] rank {rank} trainable order differs from rank 0 at index {first}: "
                      f"rank0={reference[first] if first < len(reference) else None} rank{rank}={other[first] if first < len(other) else None} (n={n} vs {len(reference)})", flush=True)

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_label_tokens,
        num_batches,
        is_train: bool = True,
    ):
        """Capture exact token-weighted source losses without changing backward."""
        source_ids = batch.get("source_id")
        label_count = None
        if is_train and source_ids is not None:
            unique_sources = torch.unique(source_ids)
            if unique_sources.numel() != 1:
                raise ValueError("Per-source diagnostics require each local microbatch to contain one source")
            source_id = int(unique_sources.item())
            if source_id < 0 or source_id >= len(ALL_SOURCES):
                raise ValueError(f"Unknown source_id={source_id}")
            label_count = int(torch.count_nonzero(batch["labels"] != -100).item())
        before = len(loss_buffer)
        result = super()._forward_backward_step(
            idx,
            batch,
            loss_buffer=loss_buffer,
            num_label_tokens=num_label_tokens,
            num_batches=num_batches,
            is_train=is_train,
        )
        if is_train and source_ids is not None and label_count is not None:
            device = loss_buffer[-1].device
            if self._diag_source_loss_numerators is None:
                self._diag_source_loss_numerators = torch.zeros(len(ALL_SOURCES), dtype=torch.float64, device=device)
                self._diag_source_label_tokens = torch.zeros(len(ALL_SOURCES), dtype=torch.float64, device=device)
                self._diag_source_samples = torch.zeros(len(ALL_SOURCES), dtype=torch.float64, device=device)
            local_normalized_loss = torch.stack(loss_buffer[before:]).sum().detach().double()
            self._diag_source_loss_numerators[source_id] += local_normalized_loss * float(num_label_tokens)
            # The unsharded batch is replicated over CP. Count its labels and
            # sample only once, while loss numerators retain every CP shard.
            cp_rank = self.device_mesh["cp"].get_local_rank() if self._get_cp_group_size() > 1 else 0
            if cp_rank == 0:
                self._diag_source_label_tokens[source_id] += label_count
                self._diag_source_samples[source_id] += int(source_ids.numel())
        return result

    def _source_train_metrics(self) -> dict[str, float]:
        if self._diag_source_loss_numerators is None:
            return {}
        numerators = self._diag_source_loss_numerators
        label_tokens = self._diag_source_label_tokens
        samples = self._diag_source_samples
        assert label_tokens is not None and samples is not None
        group = self._get_dp_group(include_cp=True)
        dist.all_reduce(numerators, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(label_tokens, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(samples, op=dist.ReduceOp.SUM, group=group)
        metrics: dict[str, float] = {}
        for source_id, source in enumerate(ALL_SOURCES):
            tokens = float(label_tokens[source_id].item())
            if tokens <= 0:
                continue
            loss = float(numerators[source_id].item()) / tokens
            previous = self._diag_source_loss_ema.get(source)
            ema = loss if previous is None else self.train_loss_ema_decay * previous + (1 - self.train_loss_ema_decay) * loss
            self._diag_source_loss_ema[source] = ema
            prefix = f"train_source/{source}"
            metrics[f"{prefix}/loss"] = loss
            metrics[f"{prefix}/loss_ema"] = ema
            metrics[f"{prefix}/num_label_tokens"] = tokens
            metrics[f"{prefix}/num_samples"] = float(samples[source_id].item())
        self._diag_source_loss_numerators = None
        self._diag_source_label_tokens = None
        self._diag_source_samples = None
        return metrics

    def _capture_step0_touched_rows(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if int(self.step_scheduler.step) != 0 or not inputs:
            return
        row_ids = inputs[0].detach().reshape(-1)
        if self._diag_step0_touched_bitmap is None:
            self._diag_step0_touched_bitmap = torch.zeros(
                int(self._diag_table_module.num_embeddings), dtype=torch.uint8, device=row_ids.device
            )
        self._diag_step0_touched_bitmap[row_ids.unique()] = 1

    def _capture_step0_row_grads(self, gradient: torch.Tensor) -> torch.Tensor:
        if int(self.step_scheduler.step) != 0:
            return gradient
        local = _local_tensor(gradient).detach().float()
        row_norms = local.square().sum(dim=-1).sqrt()
        bitmap = torch.zeros(
            int(self._diag_table_module.num_embeddings), dtype=torch.uint8, device=row_norms.device
        )
        global_start = int(self._diag_table_module.global_row_start)
        global_end = int(self._diag_table_module.global_row_end)
        for head, (offset, size) in enumerate(zip(self._diag_head_offsets, self._diag_head_sizes, strict=True)):
            start = max(offset, global_start)
            end = min(offset + size, global_end)
            if end > start:
                local_slice = row_norms[start - global_start : end - global_start]
                local_nonzero = torch.nonzero(local_slice, as_tuple=False).flatten()
                bitmap[start + local_nonzero] = 1
        self._diag_step0_grad_bitmap = bitmap
        return gradient

    def _step0_gradient_metrics(self) -> dict[str, float]:
        bitmap = self._diag_step0_touched_bitmap
        gradient_bitmap = self._diag_step0_grad_bitmap
        if bitmap is None or gradient_bitmap is None:
            return {}
        dist.all_reduce(bitmap, op=dist.ReduceOp.MAX)
        dist.all_reduce(gradient_bitmap, op=dist.ReduceOp.MAX)
        metrics: dict[str, float] = {}
        for head, (offset, size) in enumerate(zip(self._diag_head_offsets, self._diag_head_sizes, strict=True)):
            touched = float(torch.count_nonzero(bitmap[offset : offset + size]).item())
            nonzero_grad = float(torch.count_nonzero(gradient_bitmap[offset : offset + size]).item())
            prefix = f"delta/step0/head_{head:02d}"
            metrics[f"{prefix}/touched_rows"] = touched
            metrics[f"{prefix}/grad_nonzero_rows"] = nonzero_grad
            metrics[f"{prefix}/grad_to_touched"] = nonzero_grad / max(touched, 1.0)
        self._diag_step0_touched_bitmap = None
        self._diag_step0_grad_bitmap = None
        return metrics

    def _make_activation_hook(
        self, name: str, *, capture_input: bool = False
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Build a hook that accumulates scalar activation statistics.

        Args:
            name: Metric name for the module output.
            capture_input: Whether to also capture the first module input.

        Returns:
            Hook accepting an output tensor of shape [batch, sequence,
            hyperconnection_hidden] and inputs whose first tensor has that shape.
        """

        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if not self._diag_capture_enabled:
                return
            tensors = {name: output.detach()}
            if capture_input:
                tensors["ple_input_hidden"] = inputs[0].detach()
            for tensor_name, value in tensors.items():
                # Accumulate scalar sums on device without synchronizing every
                # validation microbatch back to the host.
                sum_sq = value.float().square().sum()
                if tensor_name in self._diag_activation_sums:
                    self._diag_activation_sums[tensor_name] += sum_sq
                else:
                    self._diag_activation_sums[tensor_name] = sum_sq
                maximum = value.detach().abs().max().float()
                if tensor_name in self._diag_activation_maxima:
                    self._diag_activation_maxima[tensor_name] = torch.maximum(
                        self._diag_activation_maxima[tensor_name], maximum
                    )
                else:
                    self._diag_activation_maxima[tensor_name] = maximum
                self._diag_activation_counts[tensor_name] += value.numel()

        return hook

    def log_train_metrics(self, log_data: MetricsSample) -> None:
        log_data.metrics.update(self._source_train_metrics())
        if int(log_data.step) == 0:
            log_data.metrics.update(self._step0_gradient_metrics())
        loss = float(log_data.metrics["loss"])
        if self._diag_train_loss_ema is None:
            self._diag_train_loss_ema = loss
        else:
            decay = self.train_loss_ema_decay
            self._diag_train_loss_ema = decay * self._diag_train_loss_ema + (1.0 - decay) * loss
        log_data.metrics["overfit/train_loss_ema"] = self._diag_train_loss_ema
        for optimizer_index, optimizer in enumerate(self.optimizer):
            for group_index, group in enumerate(optimizer.param_groups):
                prefix = f"optimizer/opt_{optimizer_index}_group_{group_index}"
                log_data.metrics[f"{prefix}/lr"] = float(group["lr"])
                log_data.metrics[f"{prefix}/lr_mult"] = float(group.get("lr_mult", 1.0))
        super().log_train_metrics(log_data)

    def _run_validation_epoch(self, val_dataloader: Any) -> MetricsSample:
        self._diag_activation_sums = {}
        self._diag_activation_maxima = {}
        self._diag_activation_counts = defaultdict(int)
        self._diag_capture_enabled = True
        try:
            sample = super()._run_validation_epoch(val_dataloader)
        finally:
            self._diag_capture_enabled = False
        gate_loader = self.val_dataloaders.get(self._gate_val_name)
        if gate_loader is not None and val_dataloader is gate_loader:
            # Second pass with the Delta branch switched off (parameters untouched).
            self._diag_delta_layer.delta_enabled = False
            try:
                off_sample = super()._run_validation_epoch(val_dataloader)
            finally:
                self._diag_delta_layer.delta_enabled = True
            on_loss = float(sample.metrics["val_loss"])
            off_loss = float(off_sample.metrics["val_loss"])
            self._gate_pending = {
                "gate/delta_on_val_loss": on_loss,
                "gate/delta_off_val_loss": off_loss,
                "gate/delta_on_minus_off": on_loss - off_loss,
                "gate/tolerance": self._gate_tolerance,
            }
            if int(sample.step) >= self._gate_min_step and on_loss - off_loss > self._gate_tolerance:
                self._gate_failed_message = (
                    f"Delta gate failed on {self._gate_val_name!r} at step {int(sample.step)}: "
                    f"delta_on={on_loss:.4f} delta_off={off_loss:.4f} gap={on_loss - off_loss:+.4f} > {self._gate_tolerance}"
                )
        return sample

    def _activation_rms(self, name: str) -> float:
        value = self._diag_activation_sums.get(name)
        count = self._diag_activation_counts.get(name, 0)
        if value is None or count == 0:
            return float("nan")
        reduced_sum = value.detach().clone().to(torch.float64)
        reduced_count = torch.tensor(float(count), dtype=torch.float64, device=reduced_sum.device)
        dist.all_reduce(reduced_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(reduced_count, op=dist.ReduceOp.SUM)
        return float((reduced_sum / reduced_count.clamp_min(1.0)).sqrt().item())

    def _activation_max(self, name: str) -> float:
        value = self._diag_activation_maxima.get(name)
        if value is None:
            return float("nan")
        reduced = value.detach().clone().to(torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
        return float(reduced.item())

    def _collect_parameter_metrics(self, step: int) -> dict[str, float]:
        if self._diag_parameter_metrics_step == step:
            return self._diag_parameter_metrics
        table = _tensor_stats(self._diag_delta_ple.ple_embedding.ngram_embedding.weight)
        key = _difference_stats(self._diag_delta_ple.key_proj.weight, self._diag_base_ple.key_proj.weight)
        value = _difference_stats(self._diag_delta_ple.value_proj.weight, self._diag_base_ple.value_proj.weight)
        self._diag_parameter_metrics_step = step
        self._diag_parameter_metrics = {
            "delta/table/rms": table["rms"],
            "delta/table/mean_abs": table["mean_abs"],
            "delta/table/max_abs": table["max_abs"],
            "delta/table/nonzero_fraction": table["nonzero_fraction"],
            "delta/table/above_1e-4_fraction": table["above_1e-4_fraction"],
            "delta/key_proj/relative_l2_change": key["relative_l2_change"],
            "delta/key_proj/cosine_to_initial": key["cosine_to_initial"],
            "delta/key_proj/delta_rms": key["delta_rms"],
            "delta/value_proj/relative_l2_change": value["relative_l2_change"],
            "delta/value_proj/cosine_to_initial": value["cosine_to_initial"],
            "delta/value_proj/delta_rms": value["delta_rms"],
        }
        self._diag_parameter_metrics.update(self._collect_row_metrics())
        return self._diag_parameter_metrics

    def _collect_row_metrics(self) -> dict[str, float]:
        local_weight = _local_tensor(self._diag_table_module.weight).detach()
        global_start = int(self._diag_table_module.global_row_start)
        global_end = int(self._diag_table_module.global_row_end)
        device = local_weight.device
        metrics: dict[str, float] = {}
        histogram_bins = 104
        for head, (offset, size) in enumerate(zip(self._diag_head_offsets, self._diag_head_sizes, strict=True)):
            start = max(offset, global_start)
            end = min(offset + size, global_end)
            if end > start:
                rows = local_weight[start - global_start : end - global_start].float()
                norms = rows.square().sum(dim=-1).sqrt()
            else:
                norms = local_weight.new_empty((0,), dtype=torch.float32)
            packed = torch.tensor(
                [
                    norms.numel(),
                    torch.count_nonzero(norms).item(),
                    torch.count_nonzero(norms > 1e-8).item(),
                    torch.count_nonzero(norms > 1e-6).item(),
                    torch.count_nonzero(norms > 1e-4).item(),
                    norms.sum().item(),
                    norms.square().sum().item(),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            local_max = norms.max().double() if norms.numel() else torch.zeros((), dtype=torch.float64, device=device)
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            log_norms = norms.clamp_min(1e-12).log10()
            histogram = torch.histc(log_norms, bins=histogram_bins, min=-12.0, max=1.0).double()
            dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
            prefix = f"delta/table/head_{head:02d}"
            count = float(packed[0].item())
            metrics[f"{prefix}/row_nonzero_fraction"] = float(packed[1].item()) / max(count, 1.0)
            metrics[f"{prefix}/row_above_1e-8_fraction"] = float(packed[2].item()) / max(count, 1.0)
            metrics[f"{prefix}/row_above_1e-6_fraction"] = float(packed[3].item()) / max(count, 1.0)
            metrics[f"{prefix}/row_above_1e-4_fraction"] = float(packed[4].item()) / max(count, 1.0)
            metrics[f"{prefix}/row_norm_mean"] = float(packed[5].item()) / max(count, 1.0)
            metrics[f"{prefix}/row_norm_rms"] = math.sqrt(float(packed[6].item()) / max(count, 1.0))
            metrics[f"{prefix}/row_norm_max"] = float(local_max.item())
            cumulative = histogram.cumsum(0)
            for quantile in (0.5, 0.9, 0.99):
                target = quantile * max(float(cumulative[-1].item()), 1.0)
                bin_index = int(
                    torch.searchsorted(
                        cumulative, torch.tensor(target, dtype=torch.float64, device=device)
                    ).item()
                )
                log_value = -12.0 + (min(bin_index, histogram_bins - 1) + 0.5) * 13.0 / histogram_bins
                metrics[f"{prefix}/row_norm_p{int(quantile * 100)}_approx"] = 10.0**log_value

            if self._diag_access_counts is not None and end > start:
                access = torch.from_numpy(
                    np.asarray(self._diag_access_counts[start:end], dtype=np.float64)
                ).to(device=device)
                y = norms.double()
                corr_sums = torch.stack(
                    [
                        torch.tensor(float(len(y)), dtype=torch.float64, device=device),
                        access.sum(),
                        y.sum(),
                        access.square().sum(),
                        y.square().sum(),
                        (access * y).sum(),
                    ]
                )
            else:
                corr_sums = torch.zeros(6, dtype=torch.float64, device=device)
            dist.all_reduce(corr_sums, op=dist.ReduceOp.SUM)
            n, sum_x, sum_y, sum_xx, sum_yy, sum_xy = corr_sums.tolist()
            covariance = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt(
                max(n * sum_xx - sum_x * sum_x, 0.0) * max(n * sum_yy - sum_y * sum_y, 0.0)
            )
            metrics[f"{prefix}/access_count_row_norm_pearson"] = covariance / max(denominator, 1e-30)
        return metrics

    def _collect_delta_metrics(self, step: int) -> dict[str, float]:
        metrics = self._collect_parameter_metrics(step).copy()
        base_rms = self._activation_rms("base_output")
        delta_rms = self._activation_rms("delta_output")
        hidden_rms = self._activation_rms("ple_input_hidden")
        base_max = self._activation_max("base_output")
        delta_max = self._activation_max("delta_output")
        hidden_max = self._activation_max("ple_input_hidden")
        metrics.update(
            {
                "delta/activation/base_ple_rms": base_rms,
                "delta/activation/delta_ple_rms": delta_rms,
                "delta/activation/ple_input_hidden_rms": hidden_rms,
                "delta/activation/base_ple_max_abs": base_max,
                "delta/activation/delta_ple_max_abs": delta_max,
                "delta/activation/ple_input_hidden_max_abs": hidden_max,
                "delta/activation/delta_to_base_rms": delta_rms / max(base_rms, 1e-30),
                "delta/activation/delta_to_hidden_rms": delta_rms / max(hidden_rms, 1e-30),
            }
        )
        return metrics

    def log_val_metrics(self, val_name: str, log_data: MetricsSample, metric_logger: Any | None = None) -> None:
        # This method is called on every rank. Collectives must happen before
        # the upstream main-rank-only logging guard.
        delta_metrics = self._collect_delta_metrics(int(log_data.step))
        log_data.metrics.update(delta_metrics)
        val_loss = float(log_data.metrics["val_loss"])
        if self._diag_latest_val_step != int(log_data.step):
            self._diag_latest_val_step = int(log_data.step)
            self._diag_latest_val_losses = {}
            self._diag_latest_val_token_counts = {}
        self._diag_latest_val_losses[val_name] = val_loss
        self._diag_latest_val_token_counts[val_name] = int(log_data.metrics["num_label_tokens"])
        previous_best = self._diag_best_val_losses.get(val_name, float("inf"))
        best_loss = min(val_loss, previous_best)
        if val_loss < previous_best:
            self._diag_best_val_steps[val_name] = int(log_data.step)
        self._diag_best_val_losses[val_name] = best_loss
        log_data.metrics["overfit/val_loss_best"] = best_loss
        log_data.metrics["overfit/val_loss_above_best"] = val_loss - best_loss
        log_data.metrics["overfit/steps_since_best_val"] = int(log_data.step) - self._diag_best_val_steps[val_name]
        if self._diag_train_loss_ema is not None:
            train_ema = self._diag_train_loss_ema
            log_data.metrics["overfit/val_minus_train_ema"] = val_loss - train_ema
            log_data.metrics["overfit/val_over_train_ema"] = val_loss / max(train_ema, 1e-30)

        if len(self._diag_latest_val_losses) > 1:
            minimum = min(self._diag_latest_val_losses.values())
            maximum = max(self._diag_latest_val_losses.values())
            log_data.metrics["retention/validation_bucket_max_minus_min"] = maximum - minimum

        if val_name == self._gate_val_name and self._gate_pending is not None:
            log_data.metrics.update(self._gate_pending)
            self._gate_pending = None
        super().log_val_metrics(val_name, log_data, metric_logger)
        if val_name == self._gate_val_name and self._gate_failed_message is not None:
            # Metrics for this step are already logged; stop before the next step.
            raise DeltaGateFailed(self._gate_failed_message)

        # The upstream call already sends log_data.metrics to W&B. Add a
        # compact per-bucket view when several validation suites are present.
        if self.dist_env.is_main and wandb.run is not None and len(self._diag_latest_val_losses) > 1:
            retention_metrics = {
                f"retention/val_loss/{name}": loss for name, loss in self._diag_latest_val_losses.items()
            }
            if len(self._diag_latest_val_losses) == len(self.val_dataloaders):
                total_tokens = sum(self._diag_latest_val_token_counts.values())
                aggregate = sum(
                    self._diag_latest_val_losses[name] * self._diag_latest_val_token_counts[name]
                    for name in self._diag_latest_val_losses
                ) / max(total_tokens, 1)
                retention_metrics["retention/val_loss/aggregate"] = aggregate
                retention_metrics["retention/num_label_tokens/aggregate"] = total_tokens
            wandb.log(retention_metrics, step=log_data.step)

    def run_train_validation_loop(self) -> Any:
        try:
            return super().run_train_validation_loop()
        finally:
            for handle in getattr(self, "_diag_hook_handles", ()):  # safe during partial setup failures
                handle.remove()
