"""Validate Qwen chat rendering and assistant-only masks before GPU training."""

from __future__ import annotations

import argparse
import statistics

from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--inspect", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()

    model_id = "Qwen/Qwen3.8-Flash-Next"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError(f"{model_id} tokenizer has no chat template")

    dataset = ChatDataset(
        path_or_dataset_id="allenai/tulu-3-sft-mixture",
        tokenizer=tokenizer,
        split=f"train[:{args.samples}]",
        shuffle_seed=42,
        truncation=True,
        seq_length=args.sequence_length,
        padding="do_not_pad",
        mask_history=True,
    )

    inspected = min(args.inspect, len(dataset))
    sequence_lengths: list[int] = []
    supervised_lengths: list[int] = []
    for index in range(inspected):
        sample = dataset[index]
        labels = sample["labels"]
        supervised_positions = [i for i, token in enumerate(labels) if token != -100]
        if not supervised_positions:
            raise RuntimeError(f"sample {index} has no supervised assistant tokens")
        if supervised_positions != list(range(supervised_positions[0], supervised_positions[-1] + 1)):
            raise RuntimeError(f"sample {index} assistant mask is not one contiguous final-turn suffix")

        target_ids = [token for token in labels if token != -100]
        preview = tokenizer.decode(target_ids, skip_special_tokens=False).replace("\n", "\\n")[:240]
        sequence_lengths.append(len(sample["input_ids"]))
        supervised_lengths.append(len(target_ids))
        print(
            f"sample={index} sequence_tokens={len(sample['input_ids'])} "
            f"supervised_tokens={len(target_ids)} target={preview!r}"
        )

    print(
        f"dataset_samples={len(dataset)} inspected={inspected} "
        f"sequence_tokens_mean={statistics.mean(sequence_lengths):.1f} "
        f"supervised_tokens_mean={statistics.mean(supervised_lengths):.1f} "
        f"chat_template_has_generation={'generation' in str(tokenizer.chat_template)}"
    )


if __name__ == "__main__":
    main()

