from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_cache import KVCacheManager
from nanovllm.engine.contracts import (
    ExecutionMode,
    ScheduledBatch,
    SequenceSchedule,
    RunnerOutput,
)


class Scheduler:

    def __init__(
        self,
        config: Config,
        cache_manager: KVCacheManager | None = None,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.cache_manager = (
            cache_manager
            if cache_manager is not None
            else BlockManager(
                config.num_kvcache_blocks,
                config.kvcache_block_size,
            )
        )
        if self.cache_manager.block_size != self.block_size:
            raise ValueError("scheduler and KV cache block sizes must match")
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> ScheduledBatch:
        scheduled: list[SequenceSchedule] = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                allocation = self.cache_manager.plan_allocation(seq)
                if allocation is None:
                    break
                num_tokens = seq.num_tokens - allocation.num_cached_tokens
            else:
                allocation = None
                num_tokens = seq.num_pending_tokens
            if remaining < num_tokens and scheduled:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                assert allocation is not None
                self.cache_manager.allocate(seq, allocation)
            num_scheduled_tokens = min(num_tokens, remaining)
            scheduled.append(SequenceSchedule(
                sequence=seq,
                token_start=seq.num_cached_tokens,
                token_count=num_scheduled_tokens,
            ))
            num_batched_tokens += num_scheduled_tokens
            if seq.num_cached_tokens + num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

        if scheduled:
            return ScheduledBatch(
                ExecutionMode.PREFILL,
                tuple(scheduled),
                self.max_num_batched_tokens,
            )
        if self.waiting and not self.running:
            seq = self.waiting[0]
            raise RuntimeError(
                f"request {seq.seq_id} requires {seq.num_blocks} KV-cache blocks, "
                f"but only {self.cache_manager.stats.num_total_blocks} exist"
            )

        # decode
        preempted_seq_ids = []
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.cache_manager.can_append(seq):
                if self.running:
                    preempted_seq_ids.append(self.preempt(self.running.pop()))
                else:
                    preempted_seq_ids.append(self.preempt(seq))
                    break
            else:
                seq.is_prefill = False
                self.cache_manager.append_slot(seq)
                scheduled.append(SequenceSchedule(
                    sequence=seq,
                    token_start=len(seq) - 1,
                    token_count=1,
                ))
        if not scheduled:
            raise RuntimeError(
                "no running request can reserve its next KV-cache block; "
                "the active context exceeds cache capacity"
            )
        self.running.extendleft(
            entry.sequence for entry in reversed(scheduled)
        )
        return ScheduledBatch(
            ExecutionMode.DECODE,
            tuple(scheduled),
            self.max_num_seqs,
            tuple(preempted_seq_ids),
        )

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.cache_manager.free(seq)
        self.waiting.appendleft(seq)
        return seq.seq_id

    def postprocess(self, batch: ScheduledBatch, output: RunnerOutput):
        if len(output.updates) != len(batch.entries):
            raise ValueError("runner output count does not match scheduled batch")
        if any(
            update.request_id != entry.sequence.seq_id
            for entry, update in zip(batch.entries, output.updates)
        ):
            raise ValueError("runner output order does not match scheduled batch")
        for entry in batch.entries:
            seq = entry.sequence
            if entry.token_start != seq.num_cached_tokens and batch.is_prefill:
                raise ValueError("sequence cache frontier changed after scheduling")
            if seq.status is SequenceStatus.FINISHED:
                raise ValueError("cannot postprocess an already finished sequence")

        for entry, update in zip(batch.entries, output.updates):
            seq = entry.sequence
            self.cache_manager.commit(seq, entry.token_count)
            seq.num_cached_tokens += entry.token_count
            if batch.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(update.token_id)
            if (
                (not seq.ignore_eos and update.token_id == self.eos)
                or seq.num_completion_tokens >= seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.cache_manager.free(seq)
                self.running.remove(seq)
