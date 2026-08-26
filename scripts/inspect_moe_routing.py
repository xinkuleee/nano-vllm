#!/usr/bin/env python3
"""Run MiniMoE on synthetic hidden states and print router-load statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Mini-MoE top-k routing on CUDA.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--implementation",
        choices=("sparse_reference", "sparse_dispatch", "triton_grouped"),
        default="sparse_dispatch",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch import nn
    from nanovllm.layers.mini_moe import MiniMoE, routing_stats

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)

    class SwiGLUExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(
                args.hidden_size, args.hidden_size * 4, bias=False
            )
            self.down_proj = nn.Linear(
                args.hidden_size * 2, args.hidden_size, bias=False
            )

        def forward(self, values):
            gate, up = self.gate_up_proj(values).chunk(2, dim=-1)
            return self.down_proj(torch.nn.functional.silu(gate) * up)

    experts = [SwiGLUExpert() for _ in range(args.experts)]
    layer = MiniMoE(
        args.hidden_size,
        experts,
        top_k=args.top_k,
        implementation=args.implementation,
    ).cuda().half()
    hidden_states = torch.randn(
        args.tokens, args.hidden_size, device="cuda", dtype=torch.float16
    )
    output, weights, indices = layer(hidden_states, return_routing=True)
    stats = routing_stats(indices, args.experts)
    print(json.dumps({
        "output_shape": list(output.shape),
        "mean_gate_sum": weights.float().sum(-1).mean().item(),
        "assignments_per_expert": stats.assignments_per_expert,
        "assignment_fraction_per_expert": stats.assignment_fraction_per_expert,
        "max_to_mean_load": stats.max_to_mean_load,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
