from __future__ import annotations

import pytest


pytestmark = pytest.mark.gpu


def _torch_and_triton():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch


def _reference_swiglu_moe(
    torch,
    hidden_states,
    route_weights,
    expert_indices,
    gate_up_weights,
    down_weights,
):
    """Readable assignment-by-assignment reference for the Triton kernels."""

    output = torch.zeros_like(hidden_states)
    intermediate_size = down_weights.shape[-1]
    for token_id in range(hidden_states.shape[0]):
        for route_slot in range(expert_indices.shape[1]):
            expert_id = int(expert_indices[token_id, route_slot])
            gate_up = torch.nn.functional.linear(
                hidden_states[token_id : token_id + 1],
                gate_up_weights[expert_id],
            )
            gate, up = gate_up.split(intermediate_size, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            expert_output = torch.nn.functional.linear(
                activated, down_weights[expert_id]
            )
            output[token_id] += (
                expert_output[0] * route_weights[token_id, route_slot]
            )
    return output


def test_triton_token_permutation_groups_assignments_and_round_trips():
    torch = _torch_and_triton()
    from nanovllm.layers.triton_moe import triton_token_permute

    # Experts 1 and 3 deliberately receive no work.  Tokens may occur twice
    # because each token has one independent assignment per top-k route slot.
    hidden = (
        torch.arange(5 * 17, device="cuda", dtype=torch.float16).reshape(5, 17)
        / 17
    )
    indices = torch.tensor(
        [[0, 2], [2, 0], [0, 2], [2, 2], [0, 0]],
        device="cuda",
        dtype=torch.int64,
    )
    weights = torch.full((5, 2), 0.5, device="cuda", dtype=torch.float16)

    dispatch = triton_token_permute(hidden, weights, indices, num_experts=4)

    assert dispatch.expert_counts.cpu().tolist() == [5, 0, 5, 0]
    assert dispatch.expert_offsets.cpu().tolist() == [0, 5, 5, 10, 10]

    destinations = dispatch.inverse_permutation.long()
    torch.testing.assert_close(
        torch.sort(destinations).values,
        torch.arange(10, device="cuda"),
    )
    torch.testing.assert_close(
        dispatch.permuted_tokens[destinations],
        hidden.repeat_interleave(2, dim=0),
    )

    flat_experts = indices.reshape(-1)
    starts = dispatch.expert_offsets[flat_experts].long()
    ends = dispatch.expert_offsets[flat_experts + 1].long()
    assert bool(torch.all(destinations >= starts))
    assert bool(torch.all(destinations < ends))


def test_triton_token_permutation_rejects_out_of_range_expert_ids():
    torch = _torch_and_triton()
    from nanovllm.layers.triton_moe import triton_token_permute

    hidden = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    weights = torch.ones(2, 1, device="cuda", dtype=torch.float16)
    indices = torch.tensor([[0], [2]], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="expert index"):
        triton_token_permute(hidden, weights, indices, num_experts=2)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_triton_grouped_swiglu_matches_torch_with_empty_expert_and_tile_tails(
    dtype_name,
):
    torch = _torch_and_triton()
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported")
    from nanovllm.layers.triton_moe import triton_grouped_swiglu_moe

    dtype = getattr(torch, dtype_name)
    tokens, hidden_size, intermediate_size, num_experts, top_k = 7, 48, 80, 4, 2
    generator = torch.Generator(device="cuda").manual_seed(23)
    hidden = torch.randn(
        tokens, hidden_size, device="cuda", dtype=dtype, generator=generator
    )
    gate_up = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        / hidden_size**0.5
    )
    down = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        / intermediate_size**0.5
    )
    # Expert 3 is empty; the other loads are intentionally uneven.
    indices = torch.tensor(
        [[0, 1], [0, 2], [2, 0], [1, 2], [0, 2], [0, 0], [2, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    logits = torch.randn(
        tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )
    weights = torch.softmax(logits, dim=-1).to(dtype)

    actual = triton_grouped_swiglu_moe(hidden, weights, indices, gate_up, down)
    expected = _reference_swiglu_moe(
        torch, hidden, weights, indices, gate_up, down
    )

    tolerance = 4e-2 if dtype_name == "bfloat16" else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def test_mini_moe_triton_grouped_matches_readable_sparse_reference():
    torch = _torch_and_triton()
    from torch import nn
    from nanovllm.layers.mini_moe import MiniMoE

    class TinySwiGLU(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(
                hidden_size, 2 * intermediate_size, bias=False
            )
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        def forward(self, values):
            gate, up = self.gate_up_proj(values).chunk(2, dim=-1)
            return self.down_proj(torch.nn.functional.silu(gate) * up)

    def experts():
        return [TinySwiGLU(32, 48) for _ in range(4)]

    torch.manual_seed(31)
    reference = MiniMoE(
        32, experts(), top_k=2, implementation="sparse_reference"
    ).cuda().half()
    grouped = MiniMoE(
        32, experts(), top_k=2, implementation="sparse_dispatch"
    ).cuda().half()
    grouped.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 5, 32, device="cuda", dtype=torch.float16)

    expected, expected_weights, expected_indices = reference(
        hidden, return_routing=True
    )
    actual, actual_weights, actual_indices = grouped(hidden, return_routing=True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_moe_rejects_non_contiguous_hidden_states():
    torch = _torch_and_triton()
    from nanovllm.layers.triton_moe import triton_token_permute

    hidden = torch.randn(32, 5, device="cuda", dtype=torch.float16).T
    indices = torch.zeros((5, 1), device="cuda", dtype=torch.int32)
    weights = torch.ones((5, 1), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="contiguous"):
        triton_token_permute(hidden, weights, indices, num_experts=2)
