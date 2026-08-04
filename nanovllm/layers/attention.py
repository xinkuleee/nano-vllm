import torch
from torch import nn

from nanovllm.layers.backends import AttentionBackend, create_attention_backend
from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        backend: AttentionBackend | str | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.backend = (
            backend
            if isinstance(backend, AttentionBackend)
            else create_attention_backend(backend or "flash_attention")
        )
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self.backend.write_kv_cache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = self.backend.prefill(q, k, v, self.scale, context)
        else:    # decode
            o = self.backend.decode(q, k_cache, v_cache, self.scale, context)
        return o
