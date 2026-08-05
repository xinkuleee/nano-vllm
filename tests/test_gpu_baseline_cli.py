from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu_baseline.py"
SPEC = spec_from_file_location("gpu_baseline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gpu_baseline = module_from_spec(SPEC)
sys.modules[SPEC.name] = gpu_baseline
SPEC.loader.exec_module(gpu_baseline)


def make_args(**overrides):
    values = {
        "workload": "quick",
        "prompt_length": None,
        "output_length": None,
        "concurrency": None,
        "warmups": None,
        "repetitions": None,
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.75,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_workload_uses_quick_preset():
    workload = gpu_baseline.resolve_workload(make_args())

    assert workload.prompt_length == 128
    assert workload.output_length == 32
    assert workload.concurrency == 4
    assert workload.warmups == 1
    assert workload.repetitions == 3


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"prompt_length": 1000, "output_length": 100}, "max_model_len"),
        ({"prompt_length": 200, "concurrency": 8}, "token budget"),
        ({"concurrency": 9}, "max_num_seqs"),
        ({"gpu_memory_utilization": 1.0}, "between 0 and 1"),
        ({"warmups": -1}, "warmups"),
    ],
)
def test_resolve_workload_rejects_unsafe_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        gpu_baseline.resolve_workload(make_args(**overrides))


def test_aggregate_reports_latency_throughput_and_memory():
    summary = gpu_baseline._aggregate([
        {
            "wall_time_seconds": 2.0,
            "output_tokens_per_second": 10.0,
            "peak_allocated_memory_gib": 2.0,
            "peak_reserved_memory_gib": 3.0,
            "ttft_seconds": 0.2,
            "tpot_seconds": 0.02,
        },
        {
            "wall_time_seconds": 4.0,
            "output_tokens_per_second": 20.0,
            "peak_allocated_memory_gib": 2.5,
            "peak_reserved_memory_gib": 3.5,
            "ttft_seconds": 0.4,
            "tpot_seconds": 0.04,
        },
    ])

    assert summary["wall_time_mean_seconds"] == 3.0
    assert summary["output_throughput_mean_tokens_per_second"] == 15.0
    assert summary["peak_reserved_memory_gib_max"] == 3.5
    assert summary["ttft_mean_seconds"] == pytest.approx(0.3)
    assert summary["tpot_mean_seconds"] == pytest.approx(0.03)


def test_generate_measured_records_engine_metrics():
    class FakeCacheStats:
        utilization = 0.5

    class FakeMode:
        def __init__(self, is_prefill):
            self.is_prefill = is_prefill

    class FakeOutput:
        request_id = 0
        token_ids = (7, 8)

    class FakeResult:
        def __init__(self, *, prefill):
            self.mode = FakeMode(prefill)
            self.cache_stats = FakeCacheStats()
            self.preempted_seq_ids = (99,) if not prefill else ()
            self.num_scheduled_tokens = 4 if prefill else 1
            self.outputs = () if prefill else (FakeOutput(),)

    class FakeTokenizer:
        @staticmethod
        def decode(token_ids):
            return "decoded"

    class FakeLLM:
        tokenizer = FakeTokenizer()

        def __init__(self):
            self.results = [FakeResult(prefill=True), FakeResult(prefill=False)]
            self.requests = []

        def add_request(self, prompt, sampling):
            self.requests.append((prompt, sampling))

        def is_finished(self):
            return not self.results

        def step(self):
            return self.results.pop(0)

    class FakeCuda:
        @staticmethod
        def synchronize():
            pass

    class FakeTorch:
        cuda = FakeCuda()

    llm = FakeLLM()
    prompts = [[1, 2, 3, 4]]

    times = iter((0.0, 0.2, 1.0, 1.04))
    original_perf_counter = gpu_baseline.time.perf_counter
    gpu_baseline.time.perf_counter = lambda: next(times)
    try:
        outputs, metrics = gpu_baseline._generate_measured(
            llm, prompts, object(), FakeTorch()
        )
    finally:
        gpu_baseline.time.perf_counter = original_perf_counter

    assert outputs == [{"token_ids": [7, 8], "text": "decoded"}]
    assert metrics["ttft_seconds"] == pytest.approx(0.2)
    assert metrics["tpot_seconds"] == pytest.approx(0.04)
    assert {key: value for key, value in metrics.items() if key not in {"ttft_seconds", "tpot_seconds"}} == {
        "decode_step_count": 1,
        "scheduled_prefill_tokens": 4,
        "scheduled_decode_tokens": 1,
        "preemptions": 1,
        "peak_logical_kv_cache_utilization": 0.5,
    }


def test_compare_token_outputs_checks_every_request_and_iteration():
    shared = {"model": "model", "workload": {"name": "quick"}, "engine": {"max_model_len": 1024}}
    left = {**shared, "measurements": [
        {"output_token_ids": [[1, 2], [3, 4]]},
        {"output_token_ids": [[5, 6], [7, 8]]},
    ]}
    right = {**shared, "measurements": [
        {"output_token_ids": [[1, 2], [3, 4]]},
        {"output_token_ids": [[5, 6], [7, 9]]},
    ]}

    comparison = gpu_baseline.compare_token_outputs(left, right)

    assert comparison["comparable"] is True
    assert comparison["matching_iterations"] == 1
    assert comparison["all_token_ids_match"] is False
