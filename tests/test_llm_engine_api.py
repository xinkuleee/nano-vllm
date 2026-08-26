from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams


def bare_engine(*, max_model_len=8):
    engine = object.__new__(LLMEngine)
    engine._closed = False
    engine.config = SimpleNamespace(max_model_len=max_model_len, kvcache_block_size=4)
    engine.tokenizer = SimpleNamespace(encode=lambda text: [1] * len(text))
    engine.scheduler = SimpleNamespace(add=lambda sequence: None)
    return engine


def test_add_request_rejects_context_that_cannot_fit_requested_output():
    engine = bare_engine(max_model_len=8)

    with pytest.raises(ValueError, match="max_model_len"):
        engine.add_request([1, 2, 3, 4, 5], SamplingParams(max_tokens=5))


def test_add_request_allows_completion_that_exactly_fills_context():
    engine = bare_engine(max_model_len=8)

    request_id = engine.add_request(
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=4),
    )

    assert isinstance(request_id, int)


def test_add_request_allows_one_token_after_full_length_prompt():
    engine = bare_engine(max_model_len=8)

    request_id = engine.add_request(
        [1, 2, 3, 4, 5, 6, 7, 8],
        SamplingParams(max_tokens=1),
    )

    assert isinstance(request_id, int)


def test_add_request_rejects_closed_engine():
    engine = bare_engine()
    engine._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        engine.add_request([1], SamplingParams(max_tokens=1))


@pytest.mark.parametrize(
    ("prompt", "error", "message"),
    [
        ([], ValueError, "at least one token"),
        ([1, "two"], TypeError, "must be integers"),
        ([True], TypeError, "must be integers"),
    ],
)
def test_add_request_validates_token_id_prompts(prompt, error, message):
    engine = bare_engine()

    with pytest.raises(error, match=message):
        engine.add_request(prompt, SamplingParams(max_tokens=1))


def test_generate_rejects_sampling_parameter_count_mismatch_before_mutation():
    engine = bare_engine()

    with pytest.raises(ValueError, match="one entry per prompt"):
        engine.generate(
            [[1], [2]],
            [SamplingParams(max_tokens=1)],
            use_tqdm=False,
        )


def test_generate_accepts_empty_prompt_batch_without_runtime_imports():
    engine = bare_engine()

    assert engine.generate([], SamplingParams(max_tokens=1), use_tqdm=False) == []


def test_generate_validates_every_prompt_before_enqueuing_any_request():
    added = []
    engine = bare_engine(max_model_len=4)
    engine.scheduler = SimpleNamespace(add=added.append)

    with pytest.raises(ValueError, match="max_model_len"):
        engine.generate(
            [[1], [1, 2, 3, 4]],
            [SamplingParams(max_tokens=1), SamplingParams(max_tokens=2)],
            use_tqdm=False,
        )

    assert added == []
