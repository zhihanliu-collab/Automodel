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
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from experiments.delta_engram.generation_probe import _difference_stats, _tensor_stats
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
        self._diag_latest_val_step = -1
        self._diag_best_val_losses: dict[str, float] = {}
        self._diag_best_val_steps: dict[str, int] = {}
        self._diag_parameter_metrics_step = -1
        self._diag_parameter_metrics: dict[str, float] = {}
        self._diag_hook_handles = [
            self._diag_base_ple.register_forward_hook(self._make_activation_hook("base_output")),
            self._diag_delta_ple.register_forward_hook(self._make_activation_hook("delta_output", capture_input=True)),
        ]

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
        return self._diag_parameter_metrics

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
        self._diag_latest_val_losses[val_name] = val_loss
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
            wandb.log(
                {f"retention/val_loss/{name}": loss for name, loss in self._diag_latest_val_losses.items()},
                step=log_data.step,
            )

    def run_train_validation_loop(self) -> Any:
        try:
            return super().run_train_validation_loop()
        finally:
            for handle in getattr(self, "_diag_hook_handles", ()):  # safe during partial setup failures
                handle.remove()
