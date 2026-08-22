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


def _model_path(variable: str = "NANOVLLM_TEST_MODEL") -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to a local Qwen3 model directory")
    path = Path(value).expanduser()
    if not path.is_dir():
        pytest.fail(f"{variable} is not a directory: {path}")
    return path


def _run_once(
    enforce_eager: bool,
    seed: int = 0,
    *,
    model: Path | None = None,
    attention_backend: str = "flash_attention",
) -> list[list[int]]:
    torch = _require_gpu_stack()
    model = model or _model_path()
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
        attention_backend=attention_backend,
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


def test_triton_flash_attention_engine_token_parity():
    flash_outputs = _run_once(
        enforce_eager=True, seed=0, attention_backend="flash_attention"
    )
    triton_outputs = _run_once(
        enforce_eager=True, seed=0, attention_backend="triton_flash_attention"
    )

    assert triton_outputs == flash_outputs


def test_cloned_expert_mini_moe_model_token_parity():
    dense_model = _model_path()
    mini_moe_model = _model_path("NANOVLLM_TEST_MINI_MOE_MODEL")

    dense_outputs = _run_once(enforce_eager=True, seed=0, model=dense_model)
    mini_moe_outputs = _run_once(
        enforce_eager=True, seed=0, model=mini_moe_model
    )

    assert mini_moe_outputs == dense_outputs
