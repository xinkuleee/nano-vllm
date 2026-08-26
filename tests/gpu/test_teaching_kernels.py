from __future__ import annotations

import math

import pytest


pytestmark = pytest.mark.gpu


def _torch_and_triton():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


def _reference_varlen(torch, query, key, value, lengths, scale):
    outputs = []
    start = 0
    repeat = query.shape[1] // key.shape[1]
    for length in lengths:
        q = query[start : start + length].transpose(0, 1).float()
        k = key[start : start + length].transpose(0, 1).float()
        v = value[start : start + length].transpose(0, 1).float()
        k = k.repeat_interleave(repeat, dim=0)
        v = v.repeat_interleave(repeat, dim=0)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        mask = torch.ones(length, length, device=query.device, dtype=torch.bool).tril()
        probabilities = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        outputs.append(torch.matmul(probabilities, v).transpose(0, 1))
        start += length
    return torch.cat(outputs).to(query.dtype)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_triton_flash_attention_matches_torch_varlen_gqa(dtype_name):
    torch = _torch_and_triton()
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")
    from nanovllm.layers.triton_flash_attention import triton_flash_attention

    dtype = getattr(torch, dtype_name)
    lengths = [17, 41]
    total = sum(lengths)
    torch.manual_seed(0)
    query = torch.randn(total, 8, 128, device="cuda", dtype=dtype)
    key = torch.randn(total, 2, 128, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    cumulative = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(128)

    actual = triton_flash_attention(
        query, key, value, cumulative, cumulative, max(lengths), max(lengths), scale
    )
    expected = _reference_varlen(torch, query, key, value, lengths, scale)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_flash_attention_matches_chunked_query_suffix():
    torch = _torch_and_triton()
    from nanovllm.layers.triton_flash_attention import triton_flash_attention

    torch.manual_seed(11)
    query_length, key_length = 17, 83
    query = torch.randn(17, 8, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(83, 2, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    cumulative_query = torch.tensor([0, query_length], device="cuda", dtype=torch.int32)
    cumulative_key = torch.tensor([0, key_length], device="cuda", dtype=torch.int32)
    scale = 1 / math.sqrt(128)

    actual = triton_flash_attention(
        query, key, value, cumulative_query, cumulative_key,
        query_length, key_length, scale,
    )
    repeat = query.shape[1] // key.shape[1]
    repeated_key = key.repeat_interleave(repeat, dim=1)
    repeated_value = value.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query.float(), repeated_key.float()) * scale
    query_positions = torch.arange(query_length, device="cuda") + key_length - query_length
    key_positions = torch.arange(key_length, device="cuda")
    causal = key_positions[None, None, :] <= query_positions[None, :, None]
    probabilities = torch.softmax(scores.masked_fill(~causal, -math.inf), dim=-1)
    expected = torch.einsum(
        "hqk,khd->qhd", probabilities, repeated_value.float()
    ).to(query.dtype)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_attention_backend_is_registered():
    _torch_and_triton()
    pytest.importorskip("flash_attn")
    from nanovllm.layers.backends import create_attention_backend
    from nanovllm.layers.triton_attention_backend import TritonFlashAttentionBackend

    assert isinstance(
        create_attention_backend("triton_flash_attention"),
        TritonFlashAttentionBackend,
    )


def test_mini_moe_dense_and_sparse_paths_match():
    torch = _torch_and_triton()
    from torch import nn
    from nanovllm.layers.mini_moe import MiniMoE, routing_stats

    torch.manual_seed(1)
    experts = [nn.Linear(16, 16, bias=False, device="cuda") for _ in range(4)]
    dense = MiniMoE(16, experts, top_k=2, implementation="dense_masked").cuda()
    sparse = MiniMoE(
        16, [nn.Linear(16, 16, bias=False, device="cuda") for _ in range(4)],
        top_k=2,
        implementation="sparse_reference",
    ).cuda()
    sparse.load_state_dict(dense.state_dict())
    hidden = torch.randn(23, 16, device="cuda")

    dense_output, weights, indices = dense(hidden, return_routing=True)
    sparse_output = sparse(hidden)
    stats = routing_stats(indices, 4)

    torch.testing.assert_close(sparse_output, dense_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(23, device="cuda"))
    assert sum(stats.assignments_per_expert) == 23 * 2
