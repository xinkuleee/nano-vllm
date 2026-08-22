import json
from pathlib import Path

import pytest

from scripts.convert_qwen3_to_mini_moe import convert, parse_layer_ids


def make_checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 4,
    }), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_converter_creates_config_overlay_and_relative_weight_links(tmp_path):
    source = tmp_path / "qwen"
    output = tmp_path / "moe"
    make_checkpoint(source)

    convert(source, output, [0, 2], 2, 2, "dense_masked")

    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3MiniMoEForCausalLM"]
    assert config["mini_moe_layer_ids"] == [0, 2]
    assert config["mini_moe_num_experts"] == 2
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").resolve() == source / "model.safetensors"


@pytest.mark.parametrize(
    "value, expected",
    [("0", [0]), ("0, 2,3", [0, 2, 3])],
)
def test_parse_layer_ids(value, expected):
    assert parse_layer_ids(value) == expected


def test_converter_rejects_invalid_layer(tmp_path):
    source = tmp_path / "qwen"
    make_checkpoint(source)

    with pytest.raises(ValueError, match="layer IDs"):
        convert(source, tmp_path / "moe", [4], 2, 2, "dense_masked")


def test_converter_rejects_non_directory_output(tmp_path):
    source = tmp_path / "qwen"
    make_checkpoint(source)
    output = tmp_path / "occupied"
    output.write_text("file", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        convert(source, output, [0], 2, 2, "dense_masked")


def test_converter_rejects_output_inside_source_checkpoint(tmp_path):
    source = tmp_path / "qwen"
    make_checkpoint(source)

    with pytest.raises(ValueError, match="must not be inside"):
        convert(source, source / "mini-moe", [0], 2, 2, "dense_masked")


@pytest.mark.parametrize(
    ("layer_ids", "num_experts", "top_k", "implementation", "message"),
    [
        ([], 2, 2, "dense_masked", "at least one"),
        ([0], 1, 1, "dense_masked", "num_experts"),
        ([0], 2, 3, "dense_masked", "top_k"),
        ([0], 2, 2, "unknown", "unknown"),
    ],
)
def test_converter_rejects_invalid_moe_configuration(
    tmp_path, layer_ids, num_experts, top_k, implementation, message
):
    source = tmp_path / "qwen"
    make_checkpoint(source)

    with pytest.raises(ValueError, match=message):
        convert(
            source,
            tmp_path / "mini-moe",
            layer_ids,
            num_experts,
            top_k,
            implementation,
        )
