from dataclasses import FrozenInstanceError
import pickle

import pytest

from nanovllm.engine.contracts import (
    ExecutionMode,
    RunnerBatch,
    ScheduledBatch,
    SequenceSchedule,
)
from nanovllm.engine.sequence import Sequence


def make_prefill_batch(tokens=(1, 2, 3, 4), block_size=4):
    sequence = Sequence(list(tokens), block_size=block_size)
    sequence.block_table = [3]
    entry = SequenceSchedule(sequence, 0, len(tokens))
    return sequence, ScheduledBatch(ExecutionMode.PREFILL, (entry,), 8)


def test_runner_batch_is_an_immutable_minimal_pickle_snapshot():
    sequence, scheduled = make_prefill_batch()
    runner_batch = RunnerBatch.from_schedule(scheduled)
    restored = pickle.loads(pickle.dumps(runner_batch))
    entry = restored.entries[0]

    assert entry.request_id == sequence.seq_id
    assert entry.token_ids == (1, 2, 3, 4)
    assert entry.block_table == (3,)
    assert entry.context_len == 4
    assert not hasattr(entry, "sequence")


def test_runner_snapshot_is_not_changed_by_later_sequence_mutation():
    sequence, scheduled = make_prefill_batch()
    runner_batch = RunnerBatch.from_schedule(scheduled)

    sequence.token_ids[0] = 99
    sequence.block_table[0] = 10

    assert runner_batch.entries[0].token_ids == (1, 2, 3, 4)
    assert runner_batch.entries[0].block_table == (3,)


def test_scheduled_batch_rejects_duplicate_sequence():
    sequence = Sequence([1, 2], block_size=4)
    entry = SequenceSchedule(sequence, 0, 2)

    with pytest.raises(ValueError, match="scheduled twice"):
        ScheduledBatch(ExecutionMode.PREFILL, (entry, entry), 4)


def test_scheduled_batch_rejects_work_over_budget():
    sequence = Sequence([1, 2, 3], block_size=4)
    entry = SequenceSchedule(sequence, 0, 3)

    with pytest.raises(ValueError, match="exceeds the token budget"):
        ScheduledBatch(ExecutionMode.PREFILL, (entry,), 2)


def test_decode_runner_batch_requires_a_block_table():
    sequence = Sequence([1, 2], block_size=4)
    entry = SequenceSchedule(sequence, 1, 1)
    scheduled = ScheduledBatch(ExecutionMode.DECODE, (entry,), 1)

    with pytest.raises(ValueError, match="require logical block tables"):
        RunnerBatch.from_schedule(scheduled)


def test_contracts_are_frozen():
    _, scheduled = make_prefill_batch()

    with pytest.raises(FrozenInstanceError):
        scheduled.token_budget = 1


@pytest.mark.parametrize(
    ("token_start", "token_count"),
    [(-1, 1), (0, 0), (2, 2)],
)
def test_sequence_schedule_rejects_invalid_ranges(token_start, token_count):
    sequence = Sequence([1, 2, 3], block_size=4)

    with pytest.raises(ValueError):
        SequenceSchedule(sequence, token_start, token_count)


def test_prefill_must_start_at_committed_cache_frontier():
    sequence = Sequence([1, 2, 3], block_size=4)
    sequence.num_cached_tokens = 1
    entry = SequenceSchedule(sequence, 0, 1)

    with pytest.raises(ValueError, match="cache frontier"):
        ScheduledBatch(ExecutionMode.PREFILL, (entry,), 1)
