from nanovllm.engine.contracts import (
    ExecutionMode,
    RunnerOutput,
    SequenceUpdate,
)
import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_sequence(tokens, block_size=4, max_tokens=8, ignore_eos=False):
    return Sequence(
        tokens,
        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
        block_size=block_size,
    )


def finish_step(scheduler, batch, token_ids):
    scheduler.postprocess(
        batch,
        RunnerOutput(tuple(
            SequenceUpdate(entry.sequence.seq_id, token_id)
            for entry, token_id in zip(batch.entries, token_ids)
        )),
    )


def test_chunked_prefill_advances_cache_without_appending_token(config_factory):
    scheduler = Scheduler(config_factory(max_num_batched_tokens=4))
    sequence = make_sequence(list(range(6)))
    scheduler.add(sequence)

    first = scheduler.schedule()
    assert first.mode is ExecutionMode.PREFILL
    assert first.entries[0].token_start == 0
    assert first.entries[0].token_count == 4
    assert sequence.status is SequenceStatus.WAITING

    finish_step(scheduler, first, [99])
    assert sequence.num_cached_tokens == 4
    assert sequence.token_ids == list(range(6))

    second = scheduler.schedule()
    assert second.entries[0].token_start == 4
    assert second.entries[0].token_count == 2
    assert sequence.status is SequenceStatus.RUNNING

    finish_step(scheduler, second, [99])
    assert sequence.num_cached_tokens == 6
    assert sequence.completion_token_ids == [99]


def test_scheduler_uses_prefix_cache_for_a_later_request(config_factory):
    scheduler = Scheduler(config_factory(max_num_batched_tokens=16))
    first = make_sequence(list(range(10)), max_tokens=1)
    scheduler.add(first)
    first_batch = scheduler.schedule()
    finish_step(scheduler, first_batch, [0])
    assert first.is_finished

    second = make_sequence(list(range(8)) + [50, 51])
    scheduler.add(second)
    second_batch = scheduler.schedule()

    assert second.num_cached_tokens == 8
    assert second_batch.entries[0].token_start == 8
    assert second_batch.entries[0].token_count == 2


def test_decode_schedules_one_token_and_allocates_boundary_block(config_factory):
    scheduler = Scheduler(config_factory())
    sequence = make_sequence([1, 2, 3, 4])
    scheduler.add(sequence)
    prefill = scheduler.schedule()
    finish_step(scheduler, prefill, [5])

    assert len(sequence) == 5
    decode = scheduler.schedule()

    assert decode.mode is ExecutionMode.DECODE
    assert decode.num_scheduled_tokens == 1
    assert decode.entries[0].token_start == 4
    assert len(sequence.block_table) == 2

    finish_step(scheduler, decode, [6])
    assert sequence.completion_token_ids == [5, 6]


def test_decode_preempts_tail_sequence_when_cache_is_full(config_factory):
    scheduler = Scheduler(
        config_factory(
            max_num_batched_tokens=8,
            max_num_seqs=2,
            num_kvcache_blocks=2,
        )
    )
    first = make_sequence([1, 2, 3, 4])
    second = make_sequence([5, 6, 7, 8])
    scheduler.add(first)
    scheduler.add(second)
    prefill = scheduler.schedule()
    finish_step(scheduler, prefill, [9, 10])

    decode = scheduler.schedule()

    assert decode.mode is ExecutionMode.DECODE
    assert decode.preempted_seq_ids == (second.seq_id,)
    assert [entry.sequence.seq_id for entry in decode.entries] == [first.seq_id]
    assert second.status is SequenceStatus.WAITING
    assert second.block_table == []
    assert scheduler.waiting[0] is second


def test_eos_finishes_and_frees_sequence(config_factory):
    scheduler = Scheduler(config_factory(eos=7))
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    batch = scheduler.schedule()

    finish_step(scheduler, batch, [7])

    assert sequence.is_finished
    assert sequence.block_table == []
    assert scheduler.is_finished()
    assert scheduler.cache_manager.stats.num_used_blocks == 0


def test_max_tokens_finishes_even_when_eos_is_ignored(config_factory):
    scheduler = Scheduler(config_factory(eos=7))
    sequence = make_sequence([1, 2], max_tokens=1, ignore_eos=True)
    scheduler.add(sequence)
    batch = scheduler.schedule()

    finish_step(scheduler, batch, [99])

    assert sequence.is_finished
    assert sequence.completion_token_ids == [99]


def test_postprocess_validates_full_output_before_mutating_state(config_factory):
    scheduler = Scheduler(config_factory())
    sequence = make_sequence([1, 2])
    scheduler.add(sequence)
    batch = scheduler.schedule()

    with pytest.raises(ValueError, match="order does not match"):
        scheduler.postprocess(
            batch,
            RunnerOutput((SequenceUpdate(sequence.seq_id + 1, 9),)),
        )

    assert sequence.num_cached_tokens == 0
    assert sequence.token_ids == [1, 2]


def test_scheduler_rejects_cache_manager_with_different_block_size(config_factory):
    with pytest.raises(ValueError, match="block sizes must match"):
        Scheduler(
            config_factory(kvcache_block_size=4),
            cache_manager=BlockManager(num_blocks=4, block_size=8),
        )
