#!/usr/bin/env python3
"""Create a lightweight config overlay for the Qwen3 Mini-MoE experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_layer_ids(value: str) -> list[int]:
    try:
        layer_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layer_ids:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return layer_ids


def relative_symlink(target: Path, link: Path) -> None:
    link.symlink_to(os.path.relpath(target, start=link.parent))


def convert(
    source: Path,
    output: Path,
    layer_ids: list[int],
    num_experts: int,
    top_k: int,
    implementation: str,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not (source / "config.json").is_file():
        raise ValueError(f"missing source config.json: {source}")
    if output.is_relative_to(source):
        raise ValueError("output must not be inside the source checkpoint directory")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if "Qwen3ForCausalLM" not in config.get("architectures", []):
        raise ValueError("source checkpoint must use Qwen3ForCausalLM")
    num_layers = int(config["num_hidden_layers"])
    if not layer_ids:
        raise ValueError("at least one Mini-MoE layer is required")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layer list contains duplicates")
    if any(layer_id < 0 or layer_id >= num_layers for layer_id in layer_ids):
        raise ValueError(f"layer IDs must be between 0 and {num_layers - 1}")
    if num_experts < 2 or not 1 <= top_k <= num_experts:
        raise ValueError("require num_experts >= 2 and 1 <= top_k <= num_experts")
    if implementation not in {
        "dense_masked",
        "sparse_reference",
        "sparse_dispatch",
        "triton_grouped",
    }:
        raise ValueError(f"unknown Mini-MoE implementation: {implementation}")

    config["architectures"] = ["Qwen3MiniMoEForCausalLM"]
    config["mini_moe_base_model"] = str(source)
    config["mini_moe_layer_ids"] = layer_ids
    config["mini_moe_num_experts"] = num_experts
    config["mini_moe_top_k"] = top_k
    config["mini_moe_implementation"] = implementation

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for path in source.iterdir():
        if path.name == "config.json":
            continue
        relative_symlink(path, output / path.name)
    print(f"Mini-MoE overlay: {output}")
    print(f"layers={layer_ids}, experts={num_experts}, top_k={top_k}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert selected Qwen3 MLP layers into checkpoint-compatible Mini-MoE layers."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layer_ids, default=[0])
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--implementation",
        choices=(
            "dense_masked",
            "sparse_reference",
            "sparse_dispatch",
            "triton_grouped",
        ),
        default="sparse_dispatch",
    )
    args = parser.parse_args()
    convert(
        args.source,
        args.output,
        args.layers,
        args.num_experts,
        args.top_k,
        args.implementation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
