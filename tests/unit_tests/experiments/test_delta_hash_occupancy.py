import json

import numpy as np
import torch

from experiments.delta_engram.analyze_delta_hash_occupancy import _mixed, _ngram_keys
from experiments.delta_engram.odoo_corpus import OdooCorpusDatasetConfig
from nemo_automodel.components.datasets.loader import TokenizerDatasetConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    QWEN3_8_FLASH_NEXT_DELTA_LAYER_MULTIPLIERS,
    Qwen3_8_FlashNextNGramEmbedding,
    build_delta_ngram_layout,
)


def test_numpy_delta_hash_matches_model_with_eos_resets():
    eos = 248044
    tokens = np.asarray([10, 11, eos, 20, 30, 31], dtype=np.uint64)
    bigrams, trigrams = _ngram_keys(tokens)
    sizes, offsets, _ = build_delta_ngram_layout(101, 16, alignment=1)

    module = Qwen3_8_FlashNextNGramEmbedding(
        torch.nn.Embedding(sum(sizes), 1),
        eos_token_id=eos,
        layer_multipliers=QWEN3_8_FLASH_NEXT_DELTA_LAYER_MULTIPLIERS,
        ngram_heads_vocab_sizes=sizes,
        ngram_heads_offsets=offsets,
    )
    expected = module._hash_input_ids(torch.tensor(tokens.astype(np.int64)).unsqueeze(0))[0].numpy()

    for head in range(8):
        actual = np.remainder(_mixed(bigrams, 2).view(np.int64), sizes[head]) + offsets[head]
        np.testing.assert_array_equal(actual, expected[:, head])
    for head in range(8, 16):
        actual = np.remainder(_mixed(trigrams, 3).view(np.int64), sizes[head]) + offsets[head]
        np.testing.assert_array_equal(actual, expected[:, head])


def test_odoo_cache_config_builds_plain_filtered_dataset(tmp_path):
    np.asarray([1, 2, 3, 4], dtype=np.int32).tofile(tmp_path / "input_ids.i32")
    np.asarray([2, 3, 4, -100], dtype=np.int32).tofile(tmp_path / "labels.i32")
    manifest = {
        "format_version": 1,
        "records": [
            {
                "offset": 0,
                "length": 4,
                "split": "train",
                "source": "offline_bills_messages",
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = OdooCorpusDatasetConfig(
        cache_dir=str(tmp_path), split="train", sources=["offline_bills_messages"]
    )

    assert not isinstance(config, TokenizerDatasetConfig)
    dataset = config.build()
    assert dataset.lengths == [4]
    sample = dataset[0]
    assert sample["input_ids"].tolist() == [1, 2, 3, 4]
    assert sample["labels"].tolist() == [2, 3, 4, -100]
    assert sample["attention_mask"].tolist() == [1, 1, 1, 1]
    assert sample["source_id"].tolist() == [0]
