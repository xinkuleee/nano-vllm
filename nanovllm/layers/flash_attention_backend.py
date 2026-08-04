import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import triton
import triton.language as tl

from nanovllm.layers.backends import AttentionBackend
from nanovllm.utils.context import AttentionMetadata


@triton.jit
def store_kv_cache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    width: tl.constexpr,
):
    token_index = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_index)
    if slot == -1:
        return
    offsets = tl.arange(0, width)
    key = tl.load(key_ptr + token_index * key_stride + offsets)
    value = tl.load(value_ptr + token_index * value_stride + offsets)
    cache_offsets = slot * width + offsets
    tl.store(key_cache_ptr + cache_offsets, key)
    tl.store(value_cache_ptr + cache_offsets, value)


def store_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_heads, head_dim = key.shape
    width = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert key_cache.stride(1) == width and value_cache.stride(1) == width
    assert slot_mapping.numel() == num_tokens
    store_kv_cache_kernel[(num_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        key_cache,
        value_cache,
        slot_mapping,
        width,
    )


class FlashAttentionBackend(AttentionBackend):
    """Default CUDA path: FlashAttention plus a Triton KV-cache write."""

    def write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        store_kv_cache(key, value, key_cache, value_cache, slot_mapping)

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return flash_attn_varlen_func(
            query,
            key,
            value,
            max_seqlen_q=metadata.max_seqlen_q,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_k=metadata.max_seqlen_k,
            cu_seqlens_k=metadata.cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=metadata.block_tables,
        )

    def decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale: float,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return flash_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            cache_seqlens=metadata.context_lens,
            block_table=metadata.block_tables,
            softmax_scale=scale,
            causal=True,
        )
