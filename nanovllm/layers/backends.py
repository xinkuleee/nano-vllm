from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from nanovllm.utils.context import AttentionMetadata


class AttentionBackend(ABC):
    """Kernel boundary for paged attention and KV-cache writes."""

    @abstractmethod
    def write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        scale: float,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError


AttentionBackendFactory = Callable[[], AttentionBackend]


def _create_flash_attention_backend() -> AttentionBackend:
    # Keep optional/device-specific dependencies out of the registry import path.
    from nanovllm.layers.flash_attention_backend import FlashAttentionBackend

    return FlashAttentionBackend()


_BACKENDS: dict[str, AttentionBackendFactory] = {
    "flash_attention": _create_flash_attention_backend,
}


def register_attention_backend(
    name: str,
    factory: AttentionBackendFactory,
) -> None:
    """Register a backend factory for later engine construction.

    Factories, rather than shared backend instances, keep any future tuning
    caches or workspaces isolated between engines.
    """

    if name in _BACKENDS:
        raise ValueError(f"attention backend is already registered: {name}")
    _BACKENDS[name] = factory


def create_attention_backend(name: str = "flash_attention") -> AttentionBackend:
    try:
        factory = _BACKENDS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"unknown attention backend {name!r}; available: {supported}") from exc
    backend = factory()
    if not isinstance(backend, AttentionBackend):
        raise TypeError(f"attention backend factory {name!r} returned an invalid object")
    return backend
