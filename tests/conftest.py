from dataclasses import dataclass

import pytest

from nanovllm.engine.sequence import Sequence


@dataclass(slots=True)
class SchedulerConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 16
    eos: int = 0
    kvcache_block_size: int = 4
    num_kvcache_blocks: int = 16


@pytest.fixture(autouse=True)
def reset_sequence_ids():
    Sequence.counter = iter(range(1_000_000))


@pytest.fixture
def config_factory():
    def make_config(**overrides):
        values = {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 16,
            "eos": 0,
            "kvcache_block_size": 4,
            "num_kvcache_blocks": 16,
        }
        values.update(overrides)
        return SchedulerConfig(**values)

    return make_config
