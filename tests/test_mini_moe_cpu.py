from __future__ import annotations

import pytest


try:
    import torch
except (ImportError, OSError):
    pytest.skip("a working PyTorch installation is required", allow_module_level=True)
from torch import nn

from nanovllm.layers.mini_moe import MiniMoE, routing_stats


def _make_pair(hidden_size: int = 8) -> tuple[MiniMoE, MiniMoE]:
    torch.manual_seed(7)
    dense = MiniMoE(
        hidden_size,
        [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(4)],
        top_k=2,
        implementation="dense_masked",
    )
    sparse = MiniMoE(
        hidden_size,
        [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(4)],
        top_k=2,
        implementation="sparse_dispatch",
    )
    sparse.load_state_dict(dense.state_dict())
    return dense, sparse


def test_dense_and_sparse_dispatch_match_and_preserve_leading_dimensions():
    dense, sparse = _make_pair()
    hidden_states = torch.randn(2, 5, 8)

    dense_output, weights, indices = dense(hidden_states, return_routing=True)
    sparse_output = sparse(hidden_states)

    torch.testing.assert_close(dense_output, sparse_output)
    assert dense_output.shape == hidden_states.shape
    assert weights.shape == (2, 5, 2)
    assert indices.shape == (2, 5, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))


def test_identical_experts_reproduce_the_dense_expert_output():
    expert = nn.Linear(8, 8, bias=False)
    experts = [nn.Linear(8, 8, bias=False) for _ in range(3)]
    for clone in experts:
        clone.load_state_dict(expert.state_dict())
    layer = MiniMoE(8, experts, top_k=2, implementation="sparse_dispatch")
    hidden_states = torch.randn(11, 8)

    torch.testing.assert_close(layer(hidden_states), expert(hidden_states))


def test_router_gemm_accepts_half_precision_model_weights():
    dense, _ = _make_pair()
    dense = dense.half()
    hidden_states = torch.randn(9, 8, dtype=torch.float16)

    output, weights, _ = dense(hidden_states, return_routing=True)

    assert output.dtype == torch.float16
    assert weights.dtype == torch.float16
    torch.testing.assert_close(
        weights.float().sum(dim=-1), torch.ones(9), atol=1e-3, rtol=1e-3
    )


def test_routing_stats_reports_imbalance_and_counts_batched_tokens():
    indices = torch.tensor([[[0, 1], [1, 1]], [[0, 0], [1, 0]]])
    stats = routing_stats(indices, num_experts=2)

    assert stats.token_count == 4
    assert stats.assignments_per_expert == (4, 4)
    assert stats.assignment_fraction_per_expert == (0.5, 0.5)
    assert stats.max_to_mean_load == 1.0


def test_routing_stats_rejects_invalid_ids():
    with pytest.raises(ValueError, match="outside"):
        routing_stats(torch.tensor([[0, 2]]), num_experts=2)

    one_token = routing_stats(torch.tensor([0, 1]), num_experts=2)
    assert one_token.token_count == 1
    assert one_token.assignments_per_expert == (1, 1)


@pytest.mark.parametrize(
    ("top_k", "implementation"),
    [(0, "dense_masked"), (3, "dense_masked"), (1, "unknown")],
)
def test_invalid_configuration_is_rejected(top_k, implementation):
    experts = [nn.Identity(), nn.Identity()]

    with pytest.raises(ValueError):
        MiniMoE(8, experts, top_k=top_k, implementation=implementation)
