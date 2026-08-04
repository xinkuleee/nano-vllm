"""Typed contracts shared by scheduling and model execution.

These objects are deliberately device-agnostic.  A scheduler decides *what*
work should run; a runner/backend decides *how* to materialise and execute it.
Keeping that boundary explicit makes later CUDA/ROCm kernels and MoE-specific
execution plans possible without teaching the scheduler about tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from nanovllm.engine.kv_cache import KVCacheStats

if TYPE_CHECKING:
    from nanovllm.engine.sequence import Sequence


class ExecutionMode(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"

    @property
    def is_prefill(self) -> bool:
        return self is ExecutionMode.PREFILL


@dataclass(frozen=True, slots=True)
class SequenceSchedule:
    """The immutable token range selected for one sequence this step."""

    sequence: Sequence
    token_start: int
    token_count: int

    def __post_init__(self) -> None:
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.token_end > len(self.sequence):
            raise ValueError("scheduled token range exceeds sequence length")

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    """A device-independent execution plan produced by the scheduler."""

    mode: ExecutionMode
    entries: tuple[SequenceSchedule, ...]
    token_budget: int
    preempted_seq_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a scheduled batch cannot be empty")
        if len({entry.sequence.seq_id for entry in self.entries}) != len(self.entries):
            raise ValueError("a sequence cannot be scheduled twice in one batch")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.num_scheduled_tokens > self.token_budget:
            raise ValueError("scheduled work exceeds the token budget")
        if self.mode is ExecutionMode.DECODE and any(
            entry.token_count != 1 for entry in self.entries
        ):
            raise ValueError("decode schedules exactly one token per sequence")
        if self.mode is ExecutionMode.PREFILL and any(
            entry.token_start != entry.sequence.num_cached_tokens
            for entry in self.entries
        ):
            raise ValueError("prefill must start at each sequence's cache frontier")
        if self.mode is ExecutionMode.DECODE and any(
            entry.token_start != len(entry.sequence) - 1
            for entry in self.entries
        ):
            raise ValueError("decode must schedule each sequence's last token")
        if len(set(self.preempted_seq_ids)) != len(self.preempted_seq_ids):
            raise ValueError("a sequence cannot be preempted twice in one batch")
        if set(self.preempted_seq_ids) & {
            entry.sequence.seq_id for entry in self.entries
        }:
            raise ValueError("a sequence cannot be both scheduled and preempted")

    @property
    def sequences(self) -> tuple[Sequence, ...]:
        return tuple(entry.sequence for entry in self.entries)

    @property
    def num_scheduled_tokens(self) -> int:
        return sum(entry.token_count for entry in self.entries)

    @property
    def is_prefill(self) -> bool:
        return self.mode.is_prefill


@dataclass(frozen=True, slots=True)
class RequestOutput:
    request_id: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EngineStepResult:
    """Observable result of one scheduling and execution iteration."""

    outputs: tuple[RequestOutput, ...]
    mode: ExecutionMode
    num_scheduled_tokens: int
    preempted_seq_ids: tuple[int, ...]
    cache_stats: KVCacheStats


@dataclass(frozen=True, slots=True)
class RunnerSequenceInput:
    """Immutable per-request snapshot transferred to model workers."""

    request_id: int
    token_ids: tuple[int, ...]
    token_start: int
    context_len: int
    block_table: tuple[int, ...]
    block_size: int
    temperature: float

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("runner input must contain at least one token")
        if self.token_start < 0 or self.context_len <= 0:
            raise ValueError("runner token positions must be positive and in range")
        if self.token_end != self.context_len:
            raise ValueError("runner tokens must end at the declared context length")
        if self.block_size <= 0:
            raise ValueError("runner block size must be positive")
        if self.temperature <= 0:
            raise ValueError("runner temperature must be positive")
        if self.block_table and len(self.block_table) < self.num_blocks:
            raise ValueError("runner block table does not cover the context")

    @classmethod
    def from_schedule(cls, entry: SequenceSchedule) -> RunnerSequenceInput:
        sequence = entry.sequence
        return cls(
            request_id=sequence.seq_id,
            token_ids=tuple(sequence[entry.token_start:entry.token_end]),
            token_start=entry.token_start,
            context_len=entry.token_end,
            block_table=tuple(sequence.block_table),
            block_size=sequence.block_size,
            temperature=sequence.temperature,
        )

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def last_block_num_tokens(self) -> int:
        return self.context_len - (self.num_blocks - 1) * self.block_size

    @property
    def num_blocks(self) -> int:
        return (self.context_len + self.block_size - 1) // self.block_size


@dataclass(frozen=True, slots=True)
class RunnerBatch:
    """Pickle-friendly runner input derived from a scheduler decision."""

    mode: ExecutionMode
    entries: tuple[RunnerSequenceInput, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("a runner batch cannot be empty")
        if len({entry.request_id for entry in self.entries}) != len(self.entries):
            raise ValueError("a sequence cannot appear twice in one runner batch")
        if len({entry.block_size for entry in self.entries}) != 1:
            raise ValueError("all runner entries must use the same block size")
        if self.mode is ExecutionMode.DECODE and any(
            entry.token_count != 1 for entry in self.entries
        ):
            raise ValueError("decode runner batches contain one token per sequence")
        if self.mode is ExecutionMode.DECODE and any(
            not entry.block_table for entry in self.entries
        ):
            raise ValueError("decode runner entries require logical block tables")
        if self.mode is ExecutionMode.DECODE and any(
            entry.token_start != entry.context_len - 1
            for entry in self.entries
        ):
            raise ValueError("decode runner entries must target the last token")

    @classmethod
    def from_schedule(cls, batch: ScheduledBatch) -> RunnerBatch:
        return cls(
            batch.mode,
            tuple(
                RunnerSequenceInput.from_schedule(entry)
                for entry in batch.entries
            ),
        )


@dataclass(frozen=True, slots=True)
class SequenceUpdate:
    """One sampled token returned by the rank-zero runner."""

    request_id: int
    token_id: int


@dataclass(frozen=True, slots=True)
class RunnerOutput:
    updates: tuple[SequenceUpdate, ...]
