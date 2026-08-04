from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence


@dataclass(frozen=True, slots=True)
class AllocationPlan:
    """Read-only admission decision for a sequence's logical KV blocks."""

    num_cached_blocks: int
    num_free_blocks_required: int
    block_size: int

    def __post_init__(self) -> None:
        if self.num_cached_blocks < 0 or self.num_free_blocks_required < 0:
            raise ValueError("allocation block counts must be non-negative")
        if self.block_size <= 0:
            raise ValueError("allocation block size must be positive")

    @property
    def num_cached_tokens(self) -> int:
        return self.num_cached_blocks * self.block_size


@dataclass(frozen=True, slots=True)
class KVCacheStats:
    num_total_blocks: int
    num_used_blocks: int
    num_free_blocks: int
    num_prefix_cache_entries: int

    @property
    def utilization(self) -> float:
        if self.num_total_blocks == 0:
            return 0.0
        return self.num_used_blocks / self.num_total_blocks


class KVCacheManager(Protocol):
    """Logical KV-block lifecycle consumed by the scheduler.

    This policy interface is deliberately independent of CUDA/ROCm tensors.
    Physical cache allocation and cache-write kernels remain runner/backend
    responsibilities.
    """

    block_size: int

    @property
    def stats(self) -> KVCacheStats:
        ...

    def plan_allocation(self, sequence: Sequence) -> AllocationPlan | None:
        ...

    def allocate(self, sequence: Sequence, plan: AllocationPlan) -> None:
        ...

    def free(self, sequence: Sequence) -> None:
        ...

    def can_append(self, sequence: Sequence) -> bool:
        ...

    def append_slot(self, sequence: Sequence) -> None:
        ...

    def commit(self, sequence: Sequence, token_count: int) -> None:
        ...
