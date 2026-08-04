import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.kv_cache import AllocationPlan
from nanovllm.engine.sequence import Sequence


def make_sequence(tokens, block_size=4):
    return Sequence(tokens, block_size=block_size)


def commit(manager, sequence, token_count):
    manager.commit(sequence, token_count)
    sequence.num_cached_tokens += token_count


def test_allocation_plan_is_read_only_and_reports_capacity():
    manager = BlockManager(num_blocks=3, block_size=4)
    sequence = make_sequence(list(range(9)))

    before = manager.stats
    plan = manager.plan_allocation(sequence)

    assert plan is not None
    assert plan.num_cached_blocks == 0
    assert plan.num_free_blocks_required == 3
    assert plan.num_cached_tokens == 0
    assert manager.stats == before
    assert sequence.block_table == []


def test_allocation_rejects_insufficient_capacity():
    manager = BlockManager(num_blocks=2, block_size=4)
    sequence = make_sequence(list(range(9)))

    assert manager.plan_allocation(sequence) is None


def test_prefix_cache_reuses_complete_blocks_and_reference_counts():
    manager = BlockManager(num_blocks=6, block_size=4)
    first = make_sequence(list(range(10)))
    first_plan = manager.plan_allocation(first)
    manager.allocate(first, first_plan)
    commit(manager, first, len(first))

    shared_ids = tuple(first.block_table[:2])
    manager.free(first)
    assert manager.stats.num_prefix_cache_entries == 2

    second = make_sequence(list(range(8)) + [99, 100])
    second_plan = manager.plan_allocation(second)

    assert second_plan is not None
    assert second_plan.num_cached_blocks == 2
    assert second_plan.num_free_blocks_required == 3

    manager.allocate(second, second_plan)
    assert tuple(second.block_table[:2]) == shared_ids
    assert second.num_cached_tokens == 8
    assert all(manager.blocks[block_id].ref_count == 1 for block_id in shared_ids)


def test_active_prefix_sharing_only_needs_new_unshared_blocks():
    manager = BlockManager(num_blocks=6, block_size=4)
    first = make_sequence(list(range(10)))
    first_plan = manager.plan_allocation(first)
    manager.allocate(first, first_plan)
    commit(manager, first, len(first))

    second = make_sequence(list(range(8)) + [99, 100])
    second_plan = manager.plan_allocation(second)

    assert second_plan is not None
    assert second_plan.num_cached_blocks == 2
    assert second_plan.num_free_blocks_required == 1
    manager.allocate(second, second_plan)
    assert all(
        manager.blocks[block_id].ref_count == 2
        for block_id in first.block_table[:2]
    )


def test_commit_indexes_only_full_blocks():
    manager = BlockManager(num_blocks=3, block_size=4)
    sequence = make_sequence(list(range(6)))
    plan = manager.plan_allocation(sequence)
    manager.allocate(sequence, plan)

    commit(manager, sequence, 3)
    assert manager.stats.num_prefix_cache_entries == 0

    commit(manager, sequence, 3)
    assert manager.stats.num_prefix_cache_entries == 1
    assert manager.blocks[sequence.block_table[0]].token_ids == [0, 1, 2, 3]
    assert manager.blocks[sequence.block_table[1]].token_ids == []


def test_append_slot_reserves_a_new_block_at_boundary():
    manager = BlockManager(num_blocks=3, block_size=4)
    sequence = make_sequence([1, 2, 3, 4])
    manager.allocate(sequence, manager.plan_allocation(sequence))
    commit(manager, sequence, 4)

    sequence.append_token(5)
    assert manager.can_append(sequence)
    manager.append_slot(sequence)

    assert len(sequence.block_table) == 2
    assert manager.stats.num_used_blocks == 2


def test_block_size_mismatch_is_rejected():
    manager = BlockManager(num_blocks=2, block_size=4)
    sequence = make_sequence([1, 2], block_size=8)

    with pytest.raises(ValueError, match="block sizes must match"):
        manager.plan_allocation(sequence)


def test_allocation_rejects_a_plan_for_another_cache_manager():
    manager = BlockManager(num_blocks=2, block_size=4)
    sequence = make_sequence([1, 2])

    with pytest.raises(ValueError, match="different cache manager"):
        manager.allocate(
            sequence,
            AllocationPlan(0, 1, block_size=8),
        )
