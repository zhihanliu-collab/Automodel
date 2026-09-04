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

"""Per-token Delta injection ratio r_t = ||alpha * Delta_t|| / ||H_t|| on four text families.

Restores a Delta-Engram checkpoint (restore-only, no optimizer step) and, for
every token of each probe sequence, records the norm of the scaled Delta PLE
output relative to the norm of the HyperConnection state it is added to, plus
the same ratio for the frozen base PLE and whether the token's bigram/trigram
exists in the exact Delta table. Distributions are logged per family:

  A  new_knowledge : handbook + MCP tutorial + memory-edit samples (the knowledge the Delta was trained to hold)
  B  original_corpus: v2 chatter threads + the v1 offline bill JSON records
  C  general_text  : repository prose (handoff, README) - text the Delta never saw
  D  agent_tool    : agent trajectory samples (chat template with tool calls)

Every rank runs the same sequences (FSDP/EP collectives), rank 0 logs
``[RATIO] {...}`` per family and writes one JSON file with the histograms.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from experiments.delta_engram.odoo_corpus import OdooCorpusDataset
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

MAX_TOKENS_PER_SEQUENCE = int(os.environ.get("RATIO_PROBE_MAX_TOKENS", "16384"))
SAMPLES_PER_SOURCE = int(os.environ.get("RATIO_PROBE_SAMPLES", "6"))
V2_CACHE = os.environ.get("RATIO_PROBE_V2_CACHE", "/mnt/data/zhihan/delta-engram/corpus/qwen38-131k-v2")
V1_CACHE = os.environ.get("RATIO_PROBE_V1_CACHE", "/mnt/data/zhihan/delta-engram/corpus/qwen38-131k-v1")
GENERAL_TEXT_FILES = os.environ.get(
    "RATIO_PROBE_GENERAL_FILES",
    "/workspace/experiments/delta_engram/HANDOFF_2026-09-03.md,/workspace/README.md,/workspace/CLAUDE.md",
).split(",")
OUTPUT_DIR = os.environ.get("RATIO_PROBE_OUTPUT_DIR", "/mnt/data/zhihan/delta-engram/probes")
HIST_EDGES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, float("inf")]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "frac_gt_0.3": float((values > 0.3).mean()),
        "frac_gt_0.5": float((values > 0.5).mean()),
        "frac_gt_1.0": float((values > 1.0).mean()),
    }


def _histogram(values: np.ndarray) -> dict[str, float]:
    counts, _ = np.histogram(values, bins=HIST_EDGES)
    labels = [f"[{HIST_EDGES[i]},{HIST_EDGES[i+1]})" for i in range(len(HIST_EDGES) - 1)]
    total = max(int(values.size), 1)
    return {label: round(float(count) / total, 4) for label, count in zip(labels, counts)}


class DeltaRatioProbeRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Restore a checkpoint and log the per-token Delta/hidden norm ratio by text family."""

    def _log(self, marker: str, payload: dict[str, Any]) -> None:
        if dist.get_rank() == 0:
            print(f"[{marker}] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)

    def _sequences(self) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Token sequences per family, identical on every rank (built on rank 0, broadcast).

        Returns:
            family -> list of (label, input_ids of shape [1, sequence]).
        """
        device = self.dist_env.device
        payload: list[tuple[str, str, list[int]]] = []
        if dist.get_rank() == 0:
            tok = self.tokenizer

            def add_text(family: str, label: str, text: str) -> None:
                ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"][:MAX_TOKENS_PER_SEQUENCE]
                payload.append((family, label, [int(x) for x in ids]))

            def add_cache(family: str, cache: str, split: str, source: str, n: int) -> None:
                try:
                    ds = OdooCorpusDataset(cache_dir=cache, split=split, sources=[source], unique_groups=True)
                except Exception as error:  # noqa: BLE001 - a missing bucket must not kill the probe
                    self._log("WARN", {"skip": f"{cache}:{split}:{source}", "error": str(error)[:200]})
                    return
                for index in range(min(n, len(ds))):
                    ids = ds[index]["input_ids"].tolist()[:MAX_TOKENS_PER_SEQUENCE]
                    payload.append((family, f"{source}:{ds.records[index]['sample_id']}", [int(x) for x in ids]))

            add_cache("A_new_knowledge", V2_CACHE, "train", "offline_docs", 2)
            add_cache("A_new_knowledge", V2_CACHE, "validation", "memory_edits", SAMPLES_PER_SOURCE)
            add_cache("B_original_corpus", V2_CACHE, "validation", "offline_messages", SAMPLES_PER_SOURCE)
            add_cache("B_original_corpus", V1_CACHE, "validation", "offline_bills_messages", SAMPLES_PER_SOURCE)
            for path in GENERAL_TEXT_FILES:
                if Path(path).exists():
                    add_text("C_general_text", Path(path).name, Path(path).read_text(encoding="utf-8"))
            add_cache("D_agent_tool", V2_CACHE, "validation", "agent_trajectories", SAMPLES_PER_SOURCE)
        holder = [payload]
        dist.broadcast_object_list(holder, src=0)
        out: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for family, label, ids in holder[0]:
            out.setdefault(family, []).append((label, torch.tensor([ids], dtype=torch.long, device=device)))
        return out

    @torch.no_grad()
    def _forward(self, input_ids: torch.Tensor) -> None:
        self.model_parts[0](
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            logits_to_keep=1,
            output_hidden_states=False,
        )

    def run_train_validation_loop(self):
        model = self.model_parts[0]
        model.eval()
        language_model = model.model.language_model
        ple_layer_index = str(int(model.config.text_config.ple_layer_ids[0]) - 1)
        layer = language_model.layers[ple_layer_index]
        base_ple, delta_ple = layer.ple, layer.delta_ple
        if base_ple is None or delta_ple is None:
            raise RuntimeError("Expected both Base PLE and Delta-PLE after checkpoint restore")
        alpha = float(getattr(layer, "delta_alpha", 1.0))
        exact = hasattr(delta_ple.ple_embedding, "_rows_and_mask")

        captures: dict[str, torch.Tensor] = {}

        def capture(name: str):
            def hook(_module, inputs, output):
                captures[f"{name}_input"] = inputs[0].detach()
                captures[f"{name}_output"] = output.detach()

            return hook

        handles = [base_ple.register_forward_hook(capture("base")), delta_ple.register_forward_hook(capture("delta"))]
        report: dict[str, Any] = {"alpha": alpha, "exact_table": exact, "max_tokens_per_sequence": MAX_TOKENS_PER_SEQUENCE, "families": {}}
        for family, items in self._sequences().items():
            ratios, base_ratios, found_any, tokens = [], [], [], 0
            per_seq = []
            for label, ids in items:
                captures.clear()
                self._forward(ids)
                hidden = captures["delta_input"].float()[0]  # [seq, hc_hidden]
                delta = captures["delta_output"].float()[0] * alpha
                base = captures["base_output"].float()[0]
                h_norm = hidden.norm(dim=-1).clamp_min(1e-12)
                r = (delta.norm(dim=-1) / h_norm).cpu().numpy()
                b = (base.norm(dim=-1) / h_norm).cpu().numpy()
                if exact:
                    _, found = delta_ple.ple_embedding._rows_and_mask(ids)
                    f = found[0].any(dim=-1).cpu().numpy()
                else:
                    f = np.ones_like(r, dtype=bool)
                ratios.append(r)
                base_ratios.append(b)
                found_any.append(f)
                tokens += int(r.size)
                per_seq.append({"label": label, "tokens": int(r.size), "r_mean": float(r.mean()), "r_p90": float(np.percentile(r, 90)), "found_frac": float(f.mean())})
            r_all = np.concatenate(ratios)
            b_all = np.concatenate(base_ratios)
            f_all = np.concatenate(found_any)
            family_report = {
                "sequences": len(items),
                "tokens": tokens,
                "found_fraction": float(f_all.mean()),
                "r_delta": _percentiles(r_all),
                "r_delta_on_found_tokens": _percentiles(r_all[f_all]) if f_all.any() else {},
                "r_base_ple": _percentiles(b_all),
                "r_delta_hist": _histogram(r_all),
                "sequences_detail": per_seq,
            }
            report["families"][family] = family_report
            self._log("RATIO", {"family": family, **{k: v for k, v in family_report.items() if k != "sequences_detail"}})
        for handle in handles:
            handle.remove()
        if dist.get_rank() == 0:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            tag = os.environ.get("RATIO_PROBE_TAG", "ratio")
            out_path = os.path.join(OUTPUT_DIR, f"{tag}.json")
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1, ensure_ascii=False)
            print(f"[RATIO_DONE] {out_path}", flush=True)
        dist.barrier()
        return 0
