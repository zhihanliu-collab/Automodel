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

"""CPU tests for the exact-dictionary Delta n-gram embedding."""

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    EXACT_NGRAM_PACK_BASE,
    EXACT_NGRAM_ZERO_ROW,
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextExactNGramEmbedding,
    exact_ngram_table_rows,
    load_exact_ngram_keys,
)

EOS = 248044


def _keys_from_sequence(ids: list[int]) -> dict[str, torch.Tensor | int]:
    """Bigram/trigram keys of one EOS-free sequence, using the EOS padding rule for the first positions."""
    base = EXACT_NGRAM_PACK_BASE
    padded = [EOS, EOS] + ids
    bigrams = {padded[i - 1] * base + padded[i] for i in range(2, len(padded))}
    trigrams = {(padded[i - 2] * base + padded[i - 1]) * base + padded[i] for i in range(2, len(padded))}
    return {
        "bigram": torch.tensor(sorted(bigrams), dtype=torch.long),
        "trigram": torch.tensor(sorted(trigrams), dtype=torch.long),
        "pack_base": base,
    }


def _build(keys: dict[str, torch.Tensor | int], head_dim: int = 4) -> Qwen3_8_FlashNextExactNGramEmbedding:
    rows = exact_ngram_table_rows(int(keys["bigram"].numel()), int(keys["trigram"].numel()), alignment=8)
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=rows, embedding_dim=head_dim, initializer_range=0.0).build(
        process_group=None, dtype=torch.float32
    )
    with torch.no_grad():
        table.weight.copy_(torch.arange(rows, dtype=torch.float32).unsqueeze(-1).expand(rows, head_dim) + 1.0)
        table.weight[EXACT_NGRAM_ZERO_ROW].zero_()
    return Qwen3_8_FlashNextExactNGramEmbedding(table, keys=keys, ngram_size=3, eos_token_id=EOS)


def test_seen_ngrams_read_their_own_rows_and_unseen_read_zero() -> None:
    train = [11, 12, 13, 14]
    module = _build(_keys_from_sequence(train))
    seen = torch.tensor([train])
    out = module(seen)  # [1, 4, 8]
    assert out.shape == (1, 4, 8)
    rows, found = module._rows_and_mask(seen)
    assert bool(found.all())
    # Every seen position reads a distinct non-zero row per head; the value encodes the row id.
    assert rows[..., 0].unique().numel() == 4 and rows[..., 1].unique().numel() == 4
    assert torch.equal(out[0, :, 0], (rows[0, :, 0] + 1).float())
    assert torch.equal(out[0, :, 4], (rows[0, :, 1] + 1).float())

    novel = torch.tensor([[11, 99, 13, 14]])  # (11,99) and (99,13) bigrams, all trigrams after them, are unseen
    novel_out = module(novel)
    novel_rows, novel_found = module._rows_and_mask(novel)
    assert not bool(novel_found[0, 1].any())  # position of 99: neither bigram nor trigram known
    assert torch.equal(novel_out[0, 1], torch.zeros(8))
    assert bool((novel_rows[0, 1] == EXACT_NGRAM_ZERO_ROW).all())
    assert bool(novel_found[0, 0].all())  # the first position (EOS,EOS,11) is unchanged


def test_unseen_ngrams_receive_zero_gradient() -> None:
    train = [5, 6, 7]
    module = _build(_keys_from_sequence(train))
    novel = torch.tensor([[5, 6, 8]])  # last position: bigram (6,8) and trigram (5,6,8) unseen
    out = module(novel)
    out.square().sum().backward()
    grad = module.ngram_embedding.weight.grad
    assert grad is not None
    assert torch.equal(grad[EXACT_NGRAM_ZERO_ROW], torch.zeros(4))
    rows, found = module._rows_and_mask(novel)
    seen_rows = rows[found].unique()
    assert seen_rows.numel() == 4  # positions 0 and 1, two heads each
    assert bool((grad[seen_rows] != 0).all())
    touched = grad.abs().sum(dim=-1).nonzero().flatten()
    assert set(touched.tolist()) == set(seen_rows.tolist())


def test_eos_resets_context_and_global_slice_matches_full_forward() -> None:
    train = [1, 2, 3, 4, 5, 6]
    module = _build(_keys_from_sequence(train))
    ids = torch.tensor([[1, 2, 3, EOS, 1, 2, 3, 4]])
    full = module(ids)
    sliced = module._forward_global_slice(ids, sequence_start=3, sequence_end=7)
    assert torch.equal(sliced, full[:, 3:7])
    # After EOS the context restarts: (EOS,1) is a training bigram (sequence start), so it is found again.
    _, found = module._rows_and_mask(ids)
    assert bool(found[0, 4, 0]) and bool(found[0, 4, 1])


def test_keys_file_round_trip(tmp_path) -> None:
    keys = _keys_from_sequence([3, 1, 2])
    path = tmp_path / "keys.pt"
    torch.save(keys, path)
    loaded = load_exact_ngram_keys(str(path))
    assert torch.equal(loaded["bigram"], keys["bigram"]) and loaded["pack_base"] == EXACT_NGRAM_PACK_BASE
    bad = dict(keys)
    bad["bigram"] = torch.tensor([5, 4], dtype=torch.long)
    torch.save(bad, path)
    with pytest.raises(ValueError, match="strictly increasing"):
        load_exact_ngram_keys(str(path))
