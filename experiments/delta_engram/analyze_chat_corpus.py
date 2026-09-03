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

"""Measure rendered and supervised token lengths for a local chat corpus."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import TypedDict

from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset


logger = logging.getLogger(__name__)


class ThresholdStats(TypedDict):
    """Counts and token mass exceeding one candidate context length."""

    max_context: int
    samples_over: int
    sample_fraction_over: float
    tokens_over: int
    token_fraction_over: float


def _nearest_rank(values: list[int], quantile: float) -> int:
    """Return the nearest-rank quantile from non-empty integer observations."""
    if not values:
        raise ValueError("Cannot calculate a quantile from no observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _threshold_stats(lengths: list[int], max_context: int) -> ThresholdStats:
    """Summarize samples and token mass that exceed ``max_context``."""
    total_tokens = sum(lengths)
    tokens_over = sum(max(0, length - max_context) for length in lengths)
    samples_over = sum(length > max_context for length in lengths)
    return {
        "max_context": max_context,
        "samples_over": samples_over,
        "sample_fraction_over": samples_over / len(lengths),
        "tokens_over": tokens_over,
        "token_fraction_over": tokens_over / total_tokens,
    }


def analyze(*, dataset_path: Path, model_id: str, progress_every: int) -> dict[str, object]:
    """Render a chat JSONL with the model template and calculate length statistics.

    Args:
        dataset_path: Local OpenAI-format JSON or JSONL dataset.
        model_id: Hugging Face tokenizer identifier or local tokenizer path.
        progress_every: Log progress after this many samples; zero disables it.

    Returns:
        JSON-serializable corpus length and supervision statistics.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError(f"{model_id} tokenizer has no chat template")

    dataset = ChatDataset(
        path_or_dataset_id=str(dataset_path),
        tokenizer=tokenizer,
        seq_length=None,
        truncation=False,
        padding="do_not_pad",
        mask_history=False,
        mask_reasoning_content=False,
    )
    sequence_lengths: list[int] = []
    supervised_lengths: list[int] = []
    supervised_runs: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample["labels"]
        supervised = [label != -100 for label in labels]
        run_count = sum(
            value and (position == 0 or not supervised[position - 1])
            for position, value in enumerate(supervised)
        )
        sequence_lengths.append(len(sample["input_ids"]))
        supervised_lengths.append(sum(supervised))
        supervised_runs.append(run_count)
        if progress_every > 0 and (index + 1) % progress_every == 0:
            logger.info("Rendered %d/%d samples", index + 1, len(dataset))

    if not sequence_lengths:
        raise RuntimeError(f"Dataset is empty: {dataset_path}")
    if any(length == 0 for length in supervised_lengths):
        empty_indices = [index for index, length in enumerate(supervised_lengths) if length == 0]
        raise RuntimeError(f"Samples have no supervised assistant tokens: {empty_indices[:20]}")

    quantiles = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
    total_tokens = sum(sequence_lengths)
    total_supervised = sum(supervised_lengths)
    return {
        "dataset_path": str(dataset_path),
        "model_id": model_id,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "chat_template_has_generation_blocks": "generation" in tokenizer.chat_template,
        "samples": len(sequence_lengths),
        "sequence_tokens": {
            "total": total_tokens,
            "mean": total_tokens / len(sequence_lengths),
            "min": min(sequence_lengths),
            "quantiles": {str(q): _nearest_rank(sequence_lengths, q) for q in quantiles},
        },
        "supervised_tokens": {
            "total": total_supervised,
            "mean": total_supervised / len(supervised_lengths),
            "fraction_of_sequence": total_supervised / total_tokens,
            "min": min(supervised_lengths),
            "quantiles": {str(q): _nearest_rank(supervised_lengths, q) for q in quantiles},
        },
        "assistant_supervised_runs": {
            "total": sum(supervised_runs),
            "mean": sum(supervised_runs) / len(supervised_runs),
            "min": min(supervised_runs),
            "max": max(supervised_runs),
        },
        "context_thresholds": [
            _threshold_stats(sequence_lengths, max_context)
            for max_context in (32768, 65536, 98304, 131072, 196608, 262144)
        ],
    }


def main() -> None:
    """Parse CLI arguments and emit corpus statistics as JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = analyze(dataset_path=args.dataset_path, model_id=args.model_id, progress_every=args.progress_every)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
