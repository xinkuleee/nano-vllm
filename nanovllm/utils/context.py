from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch

from nanovllm.engine.contracts import ExecutionMode


@dataclass(frozen=True, slots=True)
class AttentionMetadata:
    """Per-batch tensors consumed by attention and the LM head.

    Keeping this contract independent from ``Sequence`` lets schedulers and
    future serving frontends evolve without leaking their mutable state into
    kernels.  Tensor fields are intentionally optional because prefill and
    decode require different metadata.
    """

    mode: ExecutionMode
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    @property
    def is_prefill(self) -> bool:
        return self.mode is ExecutionMode.PREFILL

    def validate(self) -> None:
        if self.slot_mapping is None:
            raise ValueError("attention metadata requires a slot mapping")
        if self.is_prefill:
            if self.cu_seqlens_q is None or self.cu_seqlens_k is None:
                raise ValueError("prefill metadata requires cumulative sequence lengths")
        elif self.context_lens is None or self.block_tables is None:
            raise ValueError("decode metadata requires context lengths and block tables")


_METADATA: ContextVar[AttentionMetadata | None] = ContextVar(
    "nanovllm_attention_metadata",
    default=None,
)


def get_context() -> AttentionMetadata:
    metadata = _METADATA.get()
    if metadata is None:
        raise RuntimeError("attention metadata is only available during model execution")
    return metadata


@contextmanager
def forward_context(metadata: AttentionMetadata) -> Iterator[AttentionMetadata]:
    """Install metadata for exactly one forward/capture scope."""

    metadata.validate()
    token = _METADATA.set(metadata)
    try:
        yield metadata
    finally:
        _METADATA.reset(token)
