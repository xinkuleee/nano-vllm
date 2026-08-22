"""Small, auditable mixture-of-experts feed-forward layer.

This is an inference teaching implementation, not a grouped-GEMM performance
kernel.  It exposes the complete MoE algorithm: router logits, top-k selection,
renormalised gates, expert computation, weighted combine, and load statistics.

``implementation="dense_masked"`` evaluates every expert and masks its output.
It is wasteful but static-shape and CUDA-Graph friendly.
``implementation="sparse_dispatch"`` sends only selected tokens to each expert,
which demonstrates real sparse execution but uses a Python expert loop and is
therefore eager-only.  A production next step replaces that loop with a sorted
token permutation and grouped GEMM.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class MoERoutingStats:
    token_count: int
    assignments_per_expert: tuple[int, ...]
    assignment_fraction_per_expert: tuple[float, ...]
    max_to_mean_load: float


def routing_stats(expert_indices: torch.Tensor, num_experts: int) -> MoERoutingStats:
    """Copy small aggregate routing counts to CPU for diagnostics."""

    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if expert_indices.ndim < 1 or expert_indices.shape[-1] < 1:
        raise ValueError("expert_indices must include an assignment dimension")
    if expert_indices.numel() and (
        torch.any(expert_indices < 0) or torch.any(expert_indices >= num_experts)
    ):
        raise ValueError("expert index is outside the configured expert range")
    counts = torch.bincount(expert_indices.reshape(-1), minlength=num_experts)
    values = tuple(int(value) for value in counts.detach().cpu().tolist())
    total = sum(values)
    fractions = tuple(value / total if total else 0.0 for value in values)
    mean = total / num_experts
    token_count = expert_indices.numel() // expert_indices.shape[-1]
    return MoERoutingStats(
        token_count=token_count,
        assignments_per_expert=values,
        assignment_fraction_per_expert=fractions,
        max_to_mean_load=max(values) / mean if mean else 0.0,
    )


class MiniMoE(nn.Module):
    """Top-k router over independently supplied expert modules."""

    def __init__(
        self,
        hidden_size: int,
        experts: list[nn.Module] | nn.ModuleList,
        top_k: int = 2,
        implementation: str = "dense_masked",
    ) -> None:
        super().__init__()
        if len(experts) < 2:
            raise ValueError("MiniMoE requires at least two experts")
        if not 1 <= top_k <= len(experts):
            raise ValueError("top_k must be between one and num_experts")
        if implementation not in {"dense_masked", "sparse_dispatch"}:
            raise ValueError("unknown MiniMoE implementation")
        self.hidden_size = hidden_size
        self.num_experts = len(experts)
        self.top_k = top_k
        self.implementation = implementation
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(experts)

    def route(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Model weights follow the checkpoint dtype; promote only the small
        # router logits for stable top-k softmax.
        router_logits = self.router(hidden_states).float()
        top_logits, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = torch.softmax(top_logits, dim=-1).to(hidden_states.dtype)
        return routing_weights, expert_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_routing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        if original_shape[-1] != self.hidden_size:
            raise ValueError("hidden state width does not match the router")
        flat = hidden_states.reshape(-1, self.hidden_size)
        weights, indices = self.route(flat)
        if self.implementation == "dense_masked":
            output = self._dense_masked(flat, weights, indices)
        else:
            output = self._sparse_dispatch(flat, weights, indices)
        output = output.reshape(original_shape)
        if return_routing:
            return output, weights.reshape(*original_shape[:-1], self.top_k), indices.reshape(
                *original_shape[:-1], self.top_k
            )
        return output

    def _dense_masked(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            expert_weight = torch.sum(
                torch.where(
                    indices == expert_id,
                    weights,
                    torch.zeros_like(weights),
                ),
                dim=-1,
            )
            output += expert_output * expert_weight.unsqueeze(-1)
        return output

    def _sparse_dispatch(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert_id, expert in enumerate(self.experts):
            token_indices, topk_slots = torch.where(indices == expert_id)
            if token_indices.numel() == 0:
                continue
            expert_output = expert(hidden_states[token_indices])
            weighted = expert_output * weights[token_indices, topk_slots].unsqueeze(-1)
            output.index_add_(0, token_indices, weighted)
        return output
