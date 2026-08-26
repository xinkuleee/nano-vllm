# Beginner glossary: attention, paging, Triton and GPU kernels

## Prefill, decode and KV cache

An autoregressive language model has two different execution phases.

**Prefill** processes all prompt tokens, usually many rows at once. It computes
the first output-token distribution and stores every layer's attention keys and
values. **Decode** generates one new token per active sequence per step and
reuses the previous keys/values. That stored memory is the **KV cache**.

These phases are not different names for the same kernel: prefill has large
matrix-like work; decode has very few new queries but reads an increasingly long
cache and is usually memory-bandwidth sensitive.

## Contiguous and paged are memory layouts

**Contiguous prefill** means one sequence's K/V rows live consecutively in a
normal tensor. The custom Triton FlashAttention kernel can advance a pointer
linearly through those rows.

In a serving engine, requests arrive and finish at different times. Reserving one
large contiguous KV buffer per request wastes memory and is hard to resize. A
**paged KV cache** divides storage into fixed-size physical blocks. A logical
sequence owns a **block table** mapping logical block numbers to physical blocks.
Paged attention follows this table while reading K/V.

Therefore two independent questions exist:

1. phase: prefill or decode?
2. layout: contiguous or paged?

That produces four conceptual combinations, even though this repository does
not implement all four itself:

| Phase and layout | Meaning | Typical shape here |
| --- | --- | --- |
| Contiguous prefill | Prompt Q/K/V are packed into ordinary consecutive tensors | Many new query rows |
| Paged prefill | A new prompt/chunk attends to K/V addressed through a block table | Many query rows plus cached pages |
| Contiguous decode | One new query reads a linearly stored cache | Not used by this runtime |
| Paged decode | One new query reads cache pages through a block table | One query row per active request |

So “Triton contiguous prefill” is not another name for “paged prefill/decode.”
`Triton` says how the custom kernel is written; `contiguous`/`paged` says how
memory is laid out; `prefill`/`decode` says which inference phase is running.

This repository currently uses:

| Situation | Implementation |
| --- | --- |
| Contiguous prefill | Custom Triton FlashAttention |
| Paged prefix-cache prefill | External `flash-attn` CUDA extension |
| Paged decode | External `flash_attn_with_kvcache` CUDA extension |
| KV-cache write | Small Triton scatter kernel |

“Other GPU path” means a branch that still executes on the GPU but calls an
external compiled CUDA implementation instead of this repository's custom
Triton kernel. It does not mean CPU fallback.

## FlashAttention and paged attention solve different problems

**FlashAttention** is an IO-aware algorithm. It tiles $Q,K,V$, maintains an
online softmax and avoids storing the full $N\times N$ score matrix in HBM
(high-bandwidth device memory).

**Paged attention** describes how K/V cache rows are addressed through a block
table. A kernel can be both Flash-style and paged, but one property does not
imply the other. The teaching Flash kernel is Flash-style but only contiguous.

## Basic GPU and Triton terms

- **Kernel**: a function launched many times in parallel on a GPU.
- **Operator/op**: the user-visible tensor operation, such as attention or
  matrix multiplication. One operator can launch one kernel or many kernels.
- **Grid**: all program instances launched for one kernel call.
- **Triton program**: one tile-level program instance, roughly comparable to a
  CUDA thread block, though Triton chooses lower-level mapping.
- **Tile/block**: a small rectangular subset of a tensor processed together.
- **Warp**: 32 NVIDIA threads executing instructions together.
- **Mask**: a Boolean guard allowing fixed-size tiles to handle array tails.
- **Pointer/stride**: a pointer identifies a memory address; a stride says how
  far to move in memory when one tensor index increases.
- **HBM/global memory**: large, off-chip GPU memory; high bandwidth but much
  slower than registers and on-chip shared memory.
- **Register**: tiny, fastest thread-local storage.
- **Occupancy**: how many warps can remain resident; excessive registers or
  shared memory reduce it.
- **Coalesced access**: neighboring lanes access neighboring addresses, allowing
  efficient memory transactions.
- **Tensor Core**: specialized hardware for small matrix multiply-accumulate.
- **GEMM**: general matrix multiplication, $C=AB$.
- **Grouped GEMM**: multiple independent GEMMs, often with unequal row counts,
  scheduled by one kernel family or launch.
- **Reduction**: combining many values into fewer values, such as sum or max.
- **Prefix sum/scan**: converting values into cumulative sums; an exclusive
  scan stores the sum of all earlier elements.
- **Atomic operation**: an indivisible read-modify-write used when parallel
  workers may update the same address.
- **Gather/scatter**: gather reads selected locations into a new order; scatter
  writes values to selected locations.
- **Permutation/unpermutation**: reorder data and later restore its logical
  order using recorded indices.
- **Fusion**: perform consecutive operations in one kernel to avoid extra
  launches and global-memory round trips.
- **SwiGLU**: the gated MLP activation
  $\operatorname{SiLU}(G)\odot U$, used by Qwen's feed-forward layer.
- **Top-$k$ routing**: choose the $k$ highest-scoring experts per token and
  normalize their weights.
- **Expert**: one independent MLP inside a mixture-of-experts layer.
- **Tensor parallelism (TP)**: split one layer's tensors/computation across
  multiple GPUs, then communicate partial results.
- **Expert parallelism (EP)**: place different experts on different GPUs and
  exchange routed tokens, commonly using all-to-all communication.
- **All-to-all**: every participating GPU may send a different tensor slice to
  every other GPU.
- **Correctness oracle/reference**: a simple trusted implementation used to
  compare an optimized kernel's output.
- **Parity**: outputs agree within an explicit exact or numerical tolerance.
- **Eager execution**: launch operations immediately instead of replaying a
  pre-recorded CUDA Graph.
- **JIT**: just-in-time compilation when a new shape/dtype/config is first used.
- **FP16/BF16**: 16-bit input formats; BF16 has wider numeric range, FP16 more
  fraction precision. Accumulation often uses FP32.
- **CUDA Graph**: records a fixed GPU launch sequence and replays it cheaply;
  dynamic allocations and shapes make capture difficult.

## What Triton hides and what it does not

Triton lets code describe tile pointers, masked loads, reductions and `tl.dot`.
Its compiler maps those operations to NVIDIA threads, warps and instructions.
The author must still choose work decomposition, tile sizes, memory access,
fusion and numerical strategy. Triton therefore removes much CUDA boilerplate,
not the need to understand GPU performance.
