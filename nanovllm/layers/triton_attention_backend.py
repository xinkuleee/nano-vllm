"""End-to-end backend using the teaching Triton FlashAttention prefill.

Only contiguous prefill is custom.  KV writes, paged decode, and paged-prefix
prefill deliberately remain on the external FlashAttention baseline.  This is
an honest boundary: decode is paged attention, not the prefill FlashAttention
algorithm implemented in ``triton_flash_attention.py``.
"""

from __future__ import annotations

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.backends import AttentionBackend
from nanovllm.layers.flash_attention_backend import store_kv_cache
from nanovllm.layers.triton_flash_attention import triton_flash_attention
from nanovllm.utils.context import AttentionMetadata


class TritonFlashAttentionBackend(AttentionBackend):
    """Teaching prefill kernel plus production paged-attention operations."""

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
        if metadata.cu_seqlens_q is None or metadata.cu_seqlens_k is None:
            raise ValueError("prefill requires cumulative sequence lengths")
        if metadata.block_tables is not None:
            # Prefix-cache K/V live in non-contiguous physical pages.
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
        return triton_flash_attention(
            query,
            key,
            value,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            metadata.max_seqlen_q,
            metadata.max_seqlen_k,
            scale,
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
