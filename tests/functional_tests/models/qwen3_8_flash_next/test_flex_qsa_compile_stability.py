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

"""GPU regression test for Qwen3.8-Flash-Next FlexAttention compilation."""

import warnings

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next import flex_qsa


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_flex_qsa_reuses_compiled_graph_across_sequence_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise local-query/global-KV shapes without a Dynamo recompile."""
    torch._dynamo.reset()
    flex_qsa._compiled_flex.cache_clear()
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    generator = torch.Generator(device="cuda").manual_seed(1234)

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="flex_attention called without torch.compile")
        for query_length in (129, 257, 385):
            kv_length = query_length * 8
            query = torch.randn(
                1,
                query_length,
                4,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            )
            key = torch.randn(
                1,
                kv_length,
                2,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            )
            value = torch.randn(
                1,
                kv_length,
                2,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
                requires_grad=True,
            )
            selected = torch.randint(
                kv_length,
                (1, query_length, 16),
                device="cuda",
                dtype=torch.int32,
                generator=generator,
            )

            output = flex_qsa.flex_sparse_gqa_attention(query, key, value, selected)
            output.float().square().mean().backward()

            assert output.shape == query.shape
            assert torch.isfinite(output).all()
            assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (query, key, value))

    flex_qsa._compiled_flex.cache_clear()
    torch._dynamo.reset()
