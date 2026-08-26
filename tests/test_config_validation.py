import sys
import types

import pytest

from nanovllm.config import Config


def install_fake_transformers(monkeypatch, **config_values):
    config = types.SimpleNamespace(
        max_position_embeddings=4096,
        **config_values,
    )

    class AutoConfig:
        @staticmethod
        def from_pretrained(path):
            return config

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoConfig=AutoConfig),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"max_model_len": 0}, "max_model_len"),
        ({"gpu_memory_utilization": 1.0}, "between 0 and 1"),
        ({"kvcache_block_size": 128}, "multiple of 256"),
        ({"tensor_parallel_size": 0}, "between 1 and 8"),
    ],
)
def test_config_rejects_invalid_runtime_values_before_loading_transformers(
    tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        Config(str(tmp_path), **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_num_batched_tokens": 1.5},
        {"max_num_seqs": True},
        {"max_model_len": "1024"},
        {"kvcache_block_size": 256.0},
        {"tensor_parallel_size": 1.0},
    ],
)
def test_config_rejects_non_integer_counts_before_loading_transformers(
    tmp_path, overrides
):
    with pytest.raises(TypeError, match="must be an integer"):
        Config(str(tmp_path), **overrides)


def test_triton_moe_rejects_tensor_parallel_runtime(monkeypatch, tmp_path):
    install_fake_transformers(
        monkeypatch,
        mini_moe_implementation="sparse_dispatch",
    )

    with pytest.raises(ValueError, match="tensor_parallel_size=1"):
        Config(
            str(tmp_path),
            enforce_eager=True,
            tensor_parallel_size=2,
        )


def test_sparse_moe_rejects_cuda_graph_mode(monkeypatch, tmp_path):
    install_fake_transformers(
        monkeypatch,
        mini_moe_implementation="sparse_reference",
    )

    with pytest.raises(ValueError, match="enforce_eager=True"):
        Config(str(tmp_path), enforce_eager=False)
