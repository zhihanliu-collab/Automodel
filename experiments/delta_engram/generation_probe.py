"""Offline generation and scale diagnostics for a restored Delta-Engram checkpoint."""

from __future__ import annotations

import json
import math
import time
import types
from typing import Any

import torch
import torch.distributed as dist

from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _all_reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_max(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _tensor_stats(value: torch.Tensor, *, chunk_elements: int = 4_000_000) -> dict[str, float]:
    """Return global moments; ratios remain correct if a DTensor dimension is replicated."""
    local = _local_tensor(value).detach().reshape(-1)
    device = local.device
    count = float(local.numel())
    sum_abs = 0.0
    sum_sq = 0.0
    maximum = 0.0
    nonzero = 0.0
    above_1e6 = 0.0
    above_1e4 = 0.0
    for start in range(0, local.numel(), chunk_elements):
        chunk = local[start : start + chunk_elements].float()
        absolute = chunk.abs()
        sum_abs += float(absolute.sum().item())
        sum_sq += float(chunk.square().sum().item())
        maximum = max(maximum, float(absolute.max().item()) if chunk.numel() else 0.0)
        nonzero += float(torch.count_nonzero(chunk).item())
        above_1e6 += float((absolute > 1e-6).sum().item())
        above_1e4 += float((absolute > 1e-4).sum().item())

    count = _all_reduce_sum(count, device)
    sum_abs = _all_reduce_sum(sum_abs, device)
    sum_sq = _all_reduce_sum(sum_sq, device)
    nonzero = _all_reduce_sum(nonzero, device)
    above_1e6 = _all_reduce_sum(above_1e6, device)
    above_1e4 = _all_reduce_sum(above_1e4, device)
    maximum = _all_reduce_max(maximum, device)
    return {
        "elements": count,
        "rms": math.sqrt(sum_sq / max(count, 1.0)),
        "mean_abs": sum_abs / max(count, 1.0),
        "max_abs": maximum,
        "nonzero_fraction": nonzero / max(count, 1.0),
        "above_1e-6_fraction": above_1e6 / max(count, 1.0),
        "above_1e-4_fraction": above_1e4 / max(count, 1.0),
    }


def _difference_stats(final: torch.Tensor, initial: torch.Tensor) -> dict[str, float]:
    final_local = _local_tensor(final).detach().reshape(-1)
    initial_local = _local_tensor(initial).detach().reshape(-1)
    if final_local.shape != initial_local.shape:
        raise RuntimeError(f"Reader shard mismatch: {final_local.shape} != {initial_local.shape}")
    device = final_local.device
    count = float(final_local.numel())
    delta_sq = 0.0
    initial_sq = 0.0
    final_sq = 0.0
    dot = 0.0
    delta_abs = 0.0
    delta_max = 0.0
    changed_1e6 = 0.0
    changed_1e4 = 0.0
    chunk_elements = 4_000_000
    for start in range(0, final_local.numel(), chunk_elements):
        current = final_local[start : start + chunk_elements].float()
        base = initial_local[start : start + chunk_elements].float()
        delta = current - base
        absolute = delta.abs()
        delta_sq += float(delta.square().sum().item())
        initial_sq += float(base.square().sum().item())
        final_sq += float(current.square().sum().item())
        dot += float((current * base).sum().item())
        delta_abs += float(absolute.sum().item())
        delta_max = max(delta_max, float(absolute.max().item()) if delta.numel() else 0.0)
        changed_1e6 += float((absolute > 1e-6).sum().item())
        changed_1e4 += float((absolute > 1e-4).sum().item())

    count = _all_reduce_sum(count, device)
    delta_sq = _all_reduce_sum(delta_sq, device)
    initial_sq = _all_reduce_sum(initial_sq, device)
    final_sq = _all_reduce_sum(final_sq, device)
    dot = _all_reduce_sum(dot, device)
    delta_abs = _all_reduce_sum(delta_abs, device)
    changed_1e6 = _all_reduce_sum(changed_1e6, device)
    changed_1e4 = _all_reduce_sum(changed_1e4, device)
    delta_max = _all_reduce_max(delta_max, device)
    return {
        "elements": count,
        "initial_rms": math.sqrt(initial_sq / max(count, 1.0)),
        "final_rms": math.sqrt(final_sq / max(count, 1.0)),
        "delta_rms": math.sqrt(delta_sq / max(count, 1.0)),
        "relative_l2_change": math.sqrt(delta_sq / max(initial_sq, 1e-30)),
        "cosine_to_initial": dot / max(math.sqrt(initial_sq * final_sq), 1e-30),
        "delta_mean_abs": delta_abs / max(count, 1.0),
        "delta_max_abs": delta_max,
        "changed_above_1e-6_fraction": changed_1e6 / max(count, 1.0),
        "changed_above_1e-4_fraction": changed_1e4 / max(count, 1.0),
    }


def _activation_stats(value: torch.Tensor) -> dict[str, float]:
    tensor = _local_tensor(value).detach().float()
    token_rms = tensor.square().mean(dim=-1).sqrt()
    return {
        "rms": float(tensor.square().mean().sqrt().item()),
        "mean_abs": float(tensor.abs().mean().item()),
        "max_abs": float(tensor.abs().max().item()),
        "token_rms_mean": float(token_rms.mean().item()),
        "token_rms_max": float(token_rms.max().item()),
    }


class DeltaGenerationProbeRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Restore one checkpoint, inspect Delta scale, and greedily decode prompts."""

    prompts = (
        "你好，请用两句话介绍一下你自己，并说明你能如何帮助用户。",
        "What is 17 times 23? Give the answer and explain the calculation briefly.",
        "写一个 Python 函数，输入整数列表并返回其中所有偶数的平方。请给出简短说明。",
    )
    max_new_tokens = 32

    def _log(self, marker: str, payload: dict[str, Any]) -> None:
        if dist.get_rank() == 0:
            print(f"[{marker}] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)

    def _render(self, prompt: str) -> torch.Tensor:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        return encoded["input_ids"].to(self.dist_env.device)

    # FSDP2's all-gather path temporarily preserves parameter version counters,
    # which is incompatible with tensors created by inference_mode().  no_grad
    # still avoids autograd storage while retaining those counters.
    @torch.no_grad()
    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model_parts[0](
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            logits_to_keep=1,
            output_hidden_states=False,
        )
        logits = output.logits
        full_tensor = getattr(logits, "full_tensor", None)
        return full_tensor() if callable(full_tensor) else logits

    @torch.no_grad()
    def _generate(self, prompt: str, mode: str) -> dict[str, Any]:
        input_ids = self._render(prompt)
        prompt_length = int(input_ids.shape[1])
        generated: list[int] = []
        start = time.perf_counter()
        for _ in range(self.max_new_tokens):
            logits = self._forward_logits(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            # Force every FSDP/EP rank down the identical decode path.
            dist.broadcast(next_token, src=0)
            token_id = int(next_token.item())
            generated.append(token_id)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if self.tokenizer.eos_token_id is not None and token_id == self.tokenizer.eos_token_id:
                break
        torch.cuda.synchronize()
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        elapsed = time.perf_counter() - start
        result = {
            "mode": mode,
            "prompt": prompt,
            "prompt_tokens": prompt_length,
            "generated_tokens": len(generated),
            "seconds": elapsed,
            "replacement_chars": text.count("\ufffd"),
            "text": text,
        }
        self._log("GENERATION", result)
        return result

    def run_train_validation_loop(self):
        model = self.model_parts[0]
        model.eval()
        language_model = model.model.language_model
        ple_layer_index = str(int(model.config.text_config.ple_layer_ids[0]) - 1)
        layer = language_model.layers[ple_layer_index]
        base_ple = layer.ple
        delta_ple = layer.delta_ple
        if base_ple is None or delta_ple is None:
            raise RuntimeError("Expected both Base PLE and Delta-PLE after checkpoint restore")

        self._log("TABLE_STATS", _tensor_stats(delta_ple.ple_embedding.ngram_embedding.weight))
        self._log("KEY_PROJ_CHANGE", _difference_stats(delta_ple.key_proj.weight, base_ple.key_proj.weight))
        self._log("VALUE_PROJ_CHANGE", _difference_stats(delta_ple.value_proj.weight, base_ple.value_proj.weight))

        captures: dict[str, torch.Tensor] = {}

        def capture(name: str):
            def hook(_module, inputs, output):
                captures[f"{name}_input"] = inputs[0].detach()
                captures[f"{name}_output"] = output.detach()

            return hook

        handles = [
            base_ple.register_forward_hook(capture("base")),
            delta_ple.register_forward_hook(capture("delta")),
        ]
        diagnostic_ids = self._render(self.prompts[0])
        logits_on = self._forward_logits(diagnostic_ids).float()
        for handle in handles:
            handle.remove()

        base_stats = _activation_stats(captures["base_output"])
        delta_stats = _activation_stats(captures["delta_output"])
        hidden_stats = _activation_stats(captures["delta_input"])
        self._log(
            "ACTIVATION_SCALE",
            {
                "base_ple": base_stats,
                "delta_ple": delta_stats,
                "ple_input_hidden": hidden_stats,
                "delta_to_base_rms": delta_stats["rms"] / max(base_stats["rms"], 1e-30),
                "delta_to_hidden_rms": delta_stats["rms"] / max(hidden_stats["rms"], 1e-30),
            },
        )

        original_forward = delta_ple.forward

        def zero_delta(_self, hidden_states, input_ids, *, cp_context=None):
            del input_ids, cp_context
            return torch.zeros_like(hidden_states)

        delta_ple.forward = types.MethodType(zero_delta, delta_ple)
        logits_off = self._forward_logits(diagnostic_ids).float()
        off_log_probs = torch.log_softmax(logits_off[:, -1, :], dim=-1)
        on_log_probs = torch.log_softmax(logits_on[:, -1, :], dim=-1)
        off_probs = off_log_probs.exp()
        kl_off_to_on = float((off_probs * (off_log_probs - on_log_probs)).sum().item())
        self._log(
            "NEXT_TOKEN_SHIFT",
            {
                "kl_off_to_on": kl_off_to_on,
                "delta_on_token": self.tokenizer.decode([int(logits_on[:, -1, :].argmax().item())]),
                "delta_off_token": self.tokenizer.decode([int(logits_off[:, -1, :].argmax().item())]),
            },
        )

        for prompt in self.prompts:
            self._generate(prompt, "delta_off")
        delta_ple.forward = original_forward
        for prompt in self.prompts:
            self._generate(prompt, "delta_on")

        dist.barrier()
        return 0
