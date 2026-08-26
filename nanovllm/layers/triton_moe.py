"""Inference-only Triton kernels for a small top-k SwiGLU MoE.

The implementation deliberately separates the route from expert arithmetic:

1. count assignments per expert;
2. exclusive-prefix-sum the counts into expert offsets;
3. permute token rows into expert-contiguous order;
4. run the gate/up and down projections with grouped GEMM kernels; and
5. weighted-scatter each assignment back to its original token row.

It targets a small, single-GPU teaching model.  Production MoE libraries use
more sophisticated sorting, persistent scheduling, capacity policies and
architecture-specific GEMM implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True, slots=True)
class TritonMoEDispatch:
    """Expert-major token buffer plus the metadata needed to undo it."""

    permuted_tokens: torch.Tensor
    inverse_permutation: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor


@triton.jit
def _count_expert_assignments_kernel(
    expert_indices_ptr,
    expert_counts_ptr,
    num_assignments,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < num_assignments
    expert_ids = tl.load(expert_indices_ptr + offsets, mask=valid, other=0)
    tl.atomic_add(expert_counts_ptr + expert_ids, 1, mask=valid)


@triton.jit
def _exclusive_prefix_sum_kernel(
    counts_ptr,
    offsets_ptr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert_ids = tl.arange(0, BLOCK)
    valid = expert_ids < num_experts
    counts = tl.load(counts_ptr + expert_ids, mask=valid, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    exclusive = inclusive - counts
    tl.store(offsets_ptr + expert_ids, exclusive, mask=valid)
    tl.store(offsets_ptr + num_experts, tl.sum(counts, axis=0))


@triton.jit
def _assign_expert_rows_kernel(
    expert_indices_ptr,
    expert_offsets_ptr,
    expert_cursors_ptr,
    inverse_permutation_ptr,
    num_assignments,
    BLOCK: tl.constexpr,
):
    assignments = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = assignments < num_assignments
    expert_ids = tl.load(expert_indices_ptr + assignments, mask=valid, other=0)
    local_rows = tl.atomic_add(
        expert_cursors_ptr + expert_ids, 1, mask=valid
    )
    destinations = tl.load(
        expert_offsets_ptr + expert_ids, mask=valid, other=0
    ) + local_rows
    tl.store(inverse_permutation_ptr + assignments, destinations, mask=valid)


@triton.jit
def _gather_permuted_tokens_kernel(
    hidden_ptr,
    inverse_permutation_ptr,
    permuted_ptr,
    num_assignments,
    hidden_size,
    top_k: tl.constexpr,
    BLOCK_ASSIGN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    assignment_offsets = (
        tl.program_id(0) * BLOCK_ASSIGN + tl.arange(0, BLOCK_ASSIGN)
    )
    valid_assignments = assignment_offsets < num_assignments
    destinations = tl.load(
        inverse_permutation_ptr + assignment_offsets,
        mask=valid_assignments,
        other=0,
    )
    token_ids = assignment_offsets // top_k

    dimension_block = tl.program_id(1)
    dimensions = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    value_mask = valid_assignments[:, None] & (dimensions[None, :] < hidden_size)
    values = tl.load(
        hidden_ptr + token_ids[:, None] * hidden_size + dimensions[None, :],
        mask=value_mask,
        other=0.0,
    )
    tl.store(
        permuted_ptr + destinations[:, None] * hidden_size + dimensions[None, :],
        values,
        mask=value_mask,
    )


@triton.jit
def _grouped_gate_up_swiglu_kernel(
    input_ptr,
    gate_up_weights_ptr,
    output_ptr,
    expert_offsets_ptr,
    num_assignments,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    scheduled_row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    expert_id = 0
    row_block = 0
    tiles_before = 0
    selected = False
    for candidate in range(num_experts):
        candidate_start = tl.load(expert_offsets_ptr + candidate)
        candidate_end = tl.load(expert_offsets_ptr + candidate + 1)
        candidate_tiles = tl.cdiv(candidate_end - candidate_start, BLOCK_M)
        belongs = (scheduled_row_block >= tiles_before) & (
            scheduled_row_block < tiles_before + candidate_tiles
        )
        expert_id = tl.where(belongs, candidate, expert_id)
        row_block = tl.where(
            belongs, scheduled_row_block - tiles_before, row_block
        )
        selected = selected | belongs
        tiles_before += candidate_tiles

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    rows = expert_start + row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator_gate = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    accumulator_up = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_tile in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        inner = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            input_ptr + rows[:, None] * hidden_size + inner[None, :],
            mask=(
                selected
                & (rows[:, None] < expert_end)
                & (rows[:, None] < num_assignments)
                & (inner[None, :] < hidden_size)
            ),
            other=0.0,
        )
        gate_values = tl.load(
            gate_up_weights_ptr
            + expert_id * (2 * intermediate_size * hidden_size)
            + columns[None, :] * hidden_size
            + inner[:, None],
            mask=(columns[None, :] < intermediate_size)
            & (inner[:, None] < hidden_size),
            other=0.0,
        )
        up_values = tl.load(
            gate_up_weights_ptr
            + expert_id * (2 * intermediate_size * hidden_size)
            + (intermediate_size + columns[None, :]) * hidden_size
            + inner[:, None],
            mask=(columns[None, :] < intermediate_size)
            & (inner[:, None] < hidden_size),
            other=0.0,
        )
        accumulator_gate += tl.dot(input_values, gate_values)
        accumulator_up += tl.dot(input_values, up_values)

    activated = accumulator_gate * tl.sigmoid(accumulator_gate)
    output = activated * accumulator_up
    tl.store(
        output_ptr + rows[:, None] * intermediate_size + columns[None, :],
        output,
        mask=(
            selected
            & (rows[:, None] < expert_end)
            & (rows[:, None] < num_assignments)
            & (columns[None, :] < intermediate_size)
        ),
    )


@triton.jit
def _grouped_down_projection_kernel(
    input_ptr,
    down_weights_ptr,
    output_ptr,
    expert_offsets_ptr,
    num_assignments,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    scheduled_row_block = tl.program_id(0)
    column_block = tl.program_id(1)
    expert_id = 0
    row_block = 0
    tiles_before = 0
    selected = False
    for candidate in range(num_experts):
        candidate_start = tl.load(expert_offsets_ptr + candidate)
        candidate_end = tl.load(expert_offsets_ptr + candidate + 1)
        candidate_tiles = tl.cdiv(candidate_end - candidate_start, BLOCK_M)
        belongs = (scheduled_row_block >= tiles_before) & (
            scheduled_row_block < tiles_before + candidate_tiles
        )
        expert_id = tl.where(belongs, candidate, expert_id)
        row_block = tl.where(
            belongs, scheduled_row_block - tiles_before, row_block
        )
        selected = selected | belongs
        tiles_before += candidate_tiles

    expert_start = tl.load(expert_offsets_ptr + expert_id)
    expert_end = tl.load(expert_offsets_ptr + expert_id + 1)
    rows = expert_start + row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = column_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    for k_tile in range(0, tl.cdiv(intermediate_size, BLOCK_K)):
        k_start = k_tile * BLOCK_K
        inner = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(
            input_ptr + rows[:, None] * intermediate_size + inner[None, :],
            mask=(
                selected
                & (rows[:, None] < expert_end)
                & (rows[:, None] < num_assignments)
                & (inner[None, :] < intermediate_size)
            ),
            other=0.0,
        )
        weight_values = tl.load(
            down_weights_ptr
            + expert_id * (hidden_size * intermediate_size)
            + columns[None, :] * intermediate_size
            + inner[:, None],
            mask=(columns[None, :] < hidden_size)
            & (inner[:, None] < intermediate_size),
            other=0.0,
        )
        accumulator += tl.dot(input_values, weight_values)

    tl.store(
        output_ptr + rows[:, None] * hidden_size + columns[None, :],
        accumulator,
        mask=(
            selected
            & (rows[:, None] < expert_end)
            & (rows[:, None] < num_assignments)
            & (columns[None, :] < hidden_size)
        ),
    )


@triton.jit
def _weighted_unpermute_kernel(
    expert_output_ptr,
    inverse_permutation_ptr,
    route_weights_ptr,
    output_ptr,
    hidden_size,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimensions = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    valid = dimensions < hidden_size
    accumulator = tl.zeros((BLOCK_D,), tl.float32)
    for route_slot in range(0, top_k):
        assignment = token_id * top_k + route_slot
        permuted_row = tl.load(inverse_permutation_ptr + assignment)
        route_weight = tl.load(route_weights_ptr + assignment)
        values = tl.load(
            expert_output_ptr + permuted_row * hidden_size + dimensions,
            mask=valid,
            other=0.0,
        )
        accumulator += values * route_weight
    tl.store(
        output_ptr + token_id * hidden_size + dimensions,
        accumulator,
        mask=valid,
    )


def _validate_routing_inputs(
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    *,
    validate_expert_range: bool = False,
) -> None:
    if not hidden_states.is_cuda:
        raise ValueError("Triton MoE requires CUDA tensors")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    if route_weights.shape != expert_indices.shape or route_weights.ndim != 2:
        raise ValueError("route weights and expert indices must have [tokens, top_k]")
    if route_weights.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing token count does not match hidden_states")
    if route_weights.device != hidden_states.device or expert_indices.device != hidden_states.device:
        raise ValueError("routing tensors must be on the hidden-state device")
    if expert_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("expert indices must be int32 or int64")
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton MoE supports FP16 and BF16 hidden states")
    if route_weights.dtype != hidden_states.dtype:
        raise ValueError("route weights must match the hidden-state dtype")
    if num_experts < 2 or num_experts > 64:
        raise ValueError("teaching Triton MoE supports 2..64 experts")
    if hidden_states.shape[0] == 0 or hidden_states.shape[1] < 16:
        raise ValueError("Triton MoE requires non-empty tokens and hidden_size >= 16")
    if route_weights.shape[1] < 1 or route_weights.shape[1] > num_experts:
        raise ValueError("top_k must be between one and num_experts")
    if not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous")
    if not expert_indices.is_contiguous():
        raise ValueError("expert indices must be contiguous")
    if not route_weights.is_contiguous():
        raise ValueError("route weights must be contiguous")
    if validate_expert_range and expert_indices.numel() and (
        torch.any(expert_indices < 0)
        or torch.any(expert_indices >= num_experts)
    ):
        raise ValueError("expert index is outside the configured expert range")


def triton_token_permute(
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    *,
    validate_expert_range: bool = True,
) -> TritonMoEDispatch:
    """Group top-k assignments by expert without a host-side sort.

    Disabling ``validate_expert_range`` avoids a host/device synchronization.
    Only do that when IDs are known to come from ``torch.topk`` over exactly
    ``num_experts`` logits, as in ``MiniMoE``'s internal inference path.
    """

    _validate_routing_inputs(
        hidden_states,
        route_weights,
        expert_indices,
        num_experts,
        validate_expert_range=validate_expert_range,
    )
    tokens, hidden_size = hidden_states.shape
    top_k = expert_indices.shape[1]
    num_assignments = tokens * top_k
    flat_indices = expert_indices.contiguous().view(-1)
    expert_counts = torch.zeros(num_experts, device=hidden_states.device, dtype=torch.int32)
    expert_offsets = torch.empty(num_experts + 1, device=hidden_states.device, dtype=torch.int32)

    count_block = 256
    _count_expert_assignments_kernel[(triton.cdiv(num_assignments, count_block),)](
        flat_indices, expert_counts, num_assignments, BLOCK=count_block
    )
    prefix_block = triton.next_power_of_2(num_experts)
    _exclusive_prefix_sum_kernel[(1,)](
        expert_counts,
        expert_offsets,
        num_experts=num_experts,
        BLOCK=prefix_block,
    )

    expert_cursors = torch.zeros_like(expert_counts)
    permuted = torch.empty(
        num_assignments,
        hidden_size,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    inverse_permutation = torch.empty(
        num_assignments, device=hidden_states.device, dtype=torch.int32
    )
    assignment_block = 256
    _assign_expert_rows_kernel[(
        triton.cdiv(num_assignments, assignment_block),
    )](
        flat_indices,
        expert_offsets,
        expert_cursors,
        inverse_permutation,
        num_assignments,
        BLOCK=assignment_block,
    )
    copy_assignment_block = 8
    feature_block = 128
    _gather_permuted_tokens_kernel[(
        triton.cdiv(num_assignments, copy_assignment_block),
        triton.cdiv(hidden_size, feature_block),
    )](
        hidden_states,
        inverse_permutation,
        permuted,
        num_assignments,
        hidden_size,
        top_k=top_k,
        BLOCK_ASSIGN=copy_assignment_block,
        BLOCK_D=feature_block,
        num_warps=4,
    )
    return TritonMoEDispatch(
        permuted,
        inverse_permutation,
        expert_counts,
        expert_offsets,
    )


def triton_grouped_swiglu_moe(
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    *,
    validate_expert_range: bool = True,
) -> torch.Tensor:
    """Run expert-contiguous Qwen SwiGLU MLPs and combine top-k outputs."""

    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    if gate_up_weights.ndim != 3 or down_weights.ndim != 3:
        raise ValueError("expert weights must be rank-three stacked tensors")
    route_weights = route_weights.contiguous()
    expert_indices = expert_indices.contiguous()
    num_experts = gate_up_weights.shape[0]
    if down_weights.shape[0] != num_experts:
        raise ValueError("gate/up and down projections must have the same experts")
    if gate_up_weights.device != hidden_states.device or down_weights.device != hidden_states.device:
        raise ValueError("expert weights must be on the hidden-state device")
    if gate_up_weights.dtype != hidden_states.dtype or down_weights.dtype != hidden_states.dtype:
        raise ValueError("expert weights must match the hidden-state dtype")
    if not gate_up_weights.is_contiguous() or not down_weights.is_contiguous():
        raise ValueError("stacked expert weights must be contiguous")
    hidden_size = hidden_states.shape[1]
    intermediate_size = down_weights.shape[2]
    if gate_up_weights.shape != (num_experts, 2 * intermediate_size, hidden_size):
        raise ValueError("gate/up expert weight shape is incompatible")
    if down_weights.shape != (num_experts, hidden_size, intermediate_size):
        raise ValueError("down expert weight shape is incompatible")
    dispatch = triton_token_permute(
        hidden_states,
        route_weights,
        expert_indices,
        num_experts,
        validate_expert_range=validate_expert_range,
    )
    num_assignments = dispatch.permuted_tokens.shape[0]

    activated = torch.empty(
        num_assignments,
        intermediate_size,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    expert_output = torch.empty_like(dispatch.permuted_tokens)
    block_m = 16
    block_n = 32
    block_k = 32
    max_row_tiles = triton.cdiv(num_assignments, block_m) + num_experts - 1
    gate_up_grid = (
        max_row_tiles,
        triton.cdiv(intermediate_size, block_n),
    )
    _grouped_gate_up_swiglu_kernel[gate_up_grid](
        dispatch.permuted_tokens,
        gate_up_weights,
        activated,
        dispatch.expert_offsets,
        num_assignments,
        hidden_size,
        intermediate_size,
        num_experts=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    down_grid = (
        max_row_tiles,
        triton.cdiv(hidden_size, block_n),
    )
    _grouped_down_projection_kernel[down_grid](
        activated,
        down_weights,
        expert_output,
        dispatch.expert_offsets,
        num_assignments,
        hidden_size,
        intermediate_size,
        num_experts=num_experts,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    output = torch.zeros_like(hidden_states)
    num_tokens = hidden_states.shape[0]
    top_k = expert_indices.shape[1]
    feature_block = 256
    _weighted_unpermute_kernel[(
        num_tokens, triton.cdiv(hidden_size, feature_block)
    )](
        expert_output,
        dispatch.inverse_permutation,
        route_weights,
        output,
        hidden_size,
        top_k=top_k,
        BLOCK_D=feature_block,
        num_warps=4,
    )
    return output
