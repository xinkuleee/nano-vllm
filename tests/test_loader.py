from types import SimpleNamespace

import pytest

from nanovllm.utils.loader import (
    _load_complete_or_sharded_weight,
    _map_packed_parameter,
    _replace_path_component,
    load_model,
)


class CopyTarget:
    def __init__(self):
        self.copied = None

    def copy_(self, value):
        self.copied = value


def test_replace_path_component_matches_complete_module_names_only():
    assert (
        _replace_path_component(
            "model.layers.0.mlp.up_proj.weight",
            "up_proj",
            "gate_up_proj",
        )
        == "model.layers.0.mlp.gate_up_proj.weight"
    )
    assert (
        _replace_path_component(
            "model.layers.0.mlp.gate_up_proj.weight",
            "up_proj",
            "gate_up_proj",
        )
        is None
    )


def test_replace_path_component_does_not_rewrite_packed_qkv_name():
    assert (
        _replace_path_component(
            "model.layers.0.self_attn.qkv_proj.weight",
            "v_proj",
            "qkv_proj",
        )
        is None
    )


def test_map_packed_parameter_distinguishes_source_and_runtime_names():
    mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    assert _map_packed_parameter(
        "model.layers.0.mlp.up_proj.weight", mapping
    ) == ("model.layers.0.mlp.gate_up_proj.weight", 1)
    assert (
        _map_packed_parameter(
            "model.layers.0.mlp.gate_up_proj.weight", mapping
        )
        is None
    )
    assert (
        _map_packed_parameter(
            "model.layers.0.self_attn.qkv_proj.weight", mapping
        )
        is None
    )


def test_complete_runtime_weights_copy_without_source_shard_loader():
    target = CopyTarget()
    shard_calls = []
    parameter = SimpleNamespace(
        shape=(4, 8),
        data=target,
        weight_loader=lambda *_: shard_calls.append(True),
    )
    loaded = SimpleNamespace(shape=(4, 8))

    _load_complete_or_sharded_weight(parameter, loaded)

    assert target.copied is loaded
    assert shard_calls == []


def test_full_source_weights_still_use_tensor_parallel_shard_loader():
    target = CopyTarget()
    shard_calls = []
    parameter = SimpleNamespace(
        shape=(2, 8),
        data=target,
        weight_loader=lambda param, value: shard_calls.append((param, value)),
    )
    loaded = SimpleNamespace(shape=(4, 8))

    _load_complete_or_sharded_weight(parameter, loaded)

    assert target.copied is None
    assert shard_calls == [(parameter, loaded)]


def test_load_model_rejects_directory_without_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        load_model(object(), str(tmp_path))
