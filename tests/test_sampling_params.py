from math import inf, nan

import pytest

from nanovllm.sampling_params import SamplingParams


@pytest.mark.parametrize("temperature", [0.0, -1.0, inf, nan, "warm", True])
def test_sampling_params_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=temperature)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_sampling_params_require_positive_output_length(max_tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=max_tokens)


@pytest.mark.parametrize("max_tokens", [1.5, "2", True])
def test_sampling_params_require_integer_output_length(max_tokens):
    with pytest.raises(TypeError, match="integer"):
        SamplingParams(max_tokens=max_tokens)


def test_sampling_params_require_boolean_ignore_eos():
    with pytest.raises(TypeError, match="boolean"):
        SamplingParams(ignore_eos=1)
