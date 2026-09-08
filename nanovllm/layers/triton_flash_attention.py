"""Tiled causal FlashAttention prefill written in Triton.

The kernel is intentionally forward-only and inference-oriented.  Each Triton
program owns a tile of query rows for one sequence and one query head.  It
streams over K/V tiles while maintaining the online-softmax state ``(m, l, o)``
in registers, so the quadratic score matrix is never written to HBM.

Supported by the teaching implementation:

* packed variable-length batches;
* causal self-attention and chunked query suffixes over contiguous K/V;
* MHA/GQA with FP16 or BF16 inputs; and
* head dimensions up to 128 (including Qwen3-0.6B's 128).

Paged prefix-cache prefill is intentionally handled by the production
FlashAttention fallback in ``TritonFlashAttentionBackend``.  A paged KV cache
changes the memory traversal and belongs to the paged-attention lesson rather
than this contiguous FlashAttention kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _flash_attention_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    output_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    query_stride_token: tl.constexpr,
    query_stride_head: tl.constexpr,
    key_stride_token: tl.constexpr,
    key_stride_head: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_head: tl.constexpr,
    output_stride_token: tl.constexpr,
    output_stride_head: tl.constexpr,
    softmax_scale,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_SEQLEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_IS_BF16: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_head = tl.program_id(1)
    sequence = tl.program_id(2)

    query_start = tl.load(cu_seqlens_q_ptr + sequence)
    query_end = tl.load(cu_seqlens_q_ptr + sequence + 1)
    key_start = tl.load(cu_seqlens_k_ptr + sequence)
    key_end = tl.load(cu_seqlens_k_ptr + sequence + 1)
    query_length = query_end - query_start
    key_length = key_end - key_start

    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dimension_offsets = tl.arange(0, BLOCK_D)
    valid_query = query_offsets < query_length
    valid_dimension = dimension_offsets < HEAD_DIM
    query_token_offsets = query_start + query_offsets

    query_ptrs = (
        query_ptr
        + query_token_offsets[:, None] * query_stride_token
        + query_head * query_stride_head
        + dimension_offsets[None, :]
    )
    query = tl.load(
        query_ptrs,
        mask=valid_query[:, None] & valid_dimension[None, :],
        other=0.0,
    )

    # Grouped-query attention maps a group of query heads to one KV head.
    kv_head = query_head // (NUM_QUERY_HEADS // NUM_KV_HEADS)

    # Online softmax state.  `maximum` and `denominator` correspond to m and l
    # in the FlashAttention paper; `accumulator` is the unnormalised output.
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # If Q is a suffix chunk, row zero is located at key_length-query_length
    # in the complete causal sequence.
    absolute_query_positions = key_length - query_length + query_offsets

    for key_block_start in range(0, MAX_SEQLEN_K, BLOCK_N):
        key_offsets = key_block_start + tl.arange(0, BLOCK_N)
        valid_key = key_offsets < key_length
        key_token_offsets = key_start + key_offsets

        key_ptrs = (
            key_ptr
            + key_token_offsets[:, None] * key_stride_token
            + kv_head * key_stride_head
            + dimension_offsets[None, :]
        )
        key = tl.load(
            key_ptrs,
            mask=valid_key[:, None] & valid_dimension[None, :],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(key))
        scores *= softmax_scale * _LOG2_E

        causal = key_offsets[None, :] <= absolute_query_positions[:, None]
        score_mask = valid_query[:, None] & valid_key[None, :] & causal
        scores = tl.where(score_mask, scores, -float("inf"))

        tile_maximum = tl.max(scores, axis=1)
        new_maximum = tl.maximum(maximum, tile_maximum)
        # Invalid rows are never stored, but keeping their state finite avoids
        # producing NaNs that complicate kernel debugging.
        new_maximum = tl.where(valid_query, new_maximum, 0.0)
        correction = tl.exp2(maximum - new_maximum)
        correction = tl.where(valid_query, correction, 0.0)
        probabilities = tl.exp2(scores - new_maximum[:, None])
        probabilities = tl.where(score_mask, probabilities, 0.0)

        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator *= correction[:, None]

        value_ptrs = (
            value_ptr
            + key_token_offsets[:, None] * value_stride_token
            + kv_head * value_stride_head
            + dimension_offsets[None, :]
        )
        value = tl.load(
            value_ptrs,
            mask=valid_key[:, None] & valid_dimension[None, :],
            other=0.0,
        )
        if INPUT_IS_BF16:
            probabilities_for_dot = probabilities.to(tl.bfloat16)
        else:
            probabilities_for_dot = probabilities.to(tl.float16)
        accumulator += tl.dot(probabilities_for_dot, value)
        maximum = new_maximum

    output = accumulator / denominator[:, None]
    output_ptrs = (
        output_ptr
        + query_token_offsets[:, None] * output_stride_token
        + query_head * output_stride_head
        + dimension_offsets[None, :]
    )
    tl.store(
        output_ptrs,
        output,
        mask=valid_query[:, None] & valid_dimension[None, :],
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> None:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Q/K/V must have shape [tokens, heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[1] < 1 or key.shape[1] < 1:
        raise ValueError("Q/K/V must contain at least one attention head")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if query.shape[-1] < 16:
        raise ValueError("teaching FlashAttention requires head_dim >= 16")
    if query.shape[1] % key.shape[1]:
        raise ValueError("query head count must be divisible by KV head count")
    if query.shape[-1] > 128:
        raise ValueError("teaching FlashAttention supports head_dim <= 128")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("teaching FlashAttention supports FP16 and BF16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V must use the same dtype")
    tensors = (query, key, value, cu_seqlens_q, cu_seqlens_k)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("teaching FlashAttention requires CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if not all(tensor.stride(-1) == 1 for tensor in (query, key, value)):
        raise ValueError("the head dimension must be contiguous")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cumulative sequence lengths must be one-dimensional")
    if cu_seqlens_q.shape != cu_seqlens_k.shape:
        raise ValueError("query/key cumulative length arrays must have equal size")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cumulative sequence lengths must use int32")
    if cu_seqlens_q.device != query.device or cu_seqlens_k.device != query.device:
        raise ValueError("cumulative sequence lengths must be on the Q/K/V device")
    if query.shape[0] == 0 or key.shape[0] == 0 or cu_seqlens_q.numel() < 2:
        raise ValueError("attention requires a non-empty packed batch")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    if max_seqlen_q > max_seqlen_k:
        raise ValueError("causal self-attention requires max_seqlen_q <= max_seqlen_k")
    if max_seqlen_q > query.shape[0] or max_seqlen_k > key.shape[0]:
        raise ValueError("maximum sequence length exceeds packed tensor storage")
    if triton.next_power_of_2(max_seqlen_k) > 65536:
        raise ValueError("teaching FlashAttention supports max_seqlen_k <= 65536")
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")


def triton_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run forward-only packed causal attention without materialising scores."""

    _validate_inputs(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
    )
    output = torch.empty_like(query)
    block_m = 32
    block_n = 32 if query.shape[-1] > 64 else 64
    block_d = triton.next_power_of_2(query.shape[-1])
    batch_size = cu_seqlens_q.numel() - 1
    grid = (triton.cdiv(max_seqlen_q, block_m), query.shape[1], batch_size)
    _flash_attention_forward_kernel[grid](
        query,
        key,
        value,
        output,
        cu_seqlens_q,
        cu_seqlens_k,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        output.stride(0),
        output.stride(1),
        softmax_scale,
        query.shape[1],
        key.shape[1],
        query.shape[-1],
        triton.next_power_of_2(max_seqlen_k),
        block_m,
        block_n,
        block_d,
        query.dtype == torch.bfloat16,
        num_warps=4,
        num_stages=2,
    )
    return output
