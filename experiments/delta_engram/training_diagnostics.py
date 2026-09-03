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
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.shared.import_utils import safe_import

_, wandb = safe_import("wandb")


class DeltaDiagnosticsTrainingRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Add low-frequency Delta scale diagnostics to the standard train recipe.

    Parameter scans and activation reductions run only when validation runs.
    Ordinary steps retain the upstream recipe's logging and compute path.
    """

    train_loss_ema_decay = 0.9

    def setup(self) -> None:
        super().setup()
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
        self._diag_latest_val_losses: dict[str, float] = {}
        self._diag_latest_val_token_counts: dict[str, int] = {}
        self._diag_latest_val_step = -1
        self._diag_best_val_losses: dict[str, float] = {}
        self._diag_best_val_steps: dict[str, int] = {}
        self._diag_parameter_metrics_step = -1
        self._diag_parameter_metrics: dict[str, float] = {}
        self._diag_table_module = self._diag_delta_ple.ple_embedding.ngram_embedding
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
            self._diag_table_module.weight.register_hook(self._capture_step0_row_grads),
        ]

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
            return super()._run_validation_epoch(val_dataloader)
        finally:
            self._diag_capture_enabled = False

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

        super().log_val_metrics(val_name, log_data, metric_logger)

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
