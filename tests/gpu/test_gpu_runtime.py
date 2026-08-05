from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu


def _require_gpu_stack():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    pytest.importorskip("flash_attn")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


def _model_path() -> Path:
    value = os.environ.get("NANOVLLM_TEST_MODEL")
    if not value:
        pytest.skip("set NANOVLLM_TEST_MODEL to a local Qwen3 model directory")
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.fail(f"NANOVLLM_TEST_MODEL is not a directory: {path}")
    return path


def _run_once(enforce_eager: bool, seed: int = 0) -> list[list[int]]:
    torch = _require_gpu_stack()
    model = _model_path()
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    llm = LLM(
        str(model),
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.75,
    )
    try:
        outputs = llm.generate(
            ["Hello from nano-vLLM.", "Count from one to five."],
            SamplingParams(temperature=1e-5, max_tokens=8, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        llm.exit()
    return [output["token_ids"] for output in outputs]


def test_eager_smoke():
    outputs = _run_once(enforce_eager=True)

    assert len(outputs) == 2
    assert all(len(token_ids) == 8 for token_ids in outputs)


def test_cuda_graph_smoke():
    outputs = _run_once(enforce_eager=False)

    assert len(outputs) == 2
    assert all(len(token_ids) == 8 for token_ids in outputs)


def test_eager_cuda_graph_token_parity():
    eager_outputs = _run_once(enforce_eager=True, seed=0)
    graph_outputs = _run_once(enforce_eager=False, seed=0)

    assert graph_outputs == eager_outputs
