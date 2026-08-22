# Teaching implementation: Triton FlashAttention

## What this implementation is

`nanovllm/layers/triton_flash_attention.py` is a real forward-only Triton
FlashAttention prefill kernel. It accepts the same packed Q/K/V layout produced
by nano-vLLM and supports variable sequence lengths, causal masking, GQA, FP16,
BF16, and head dimensions through 128. Select it with:

```python
llm = LLM(model_path, attention_backend="triton_flash_attention")
```

The source, registration, CPU-side contracts, and test harness are complete.
The kernel has not yet been JIT-compiled on an NVIDIA GPU in this branch, so it
must not be presented as validated or faster until the RTX 3060 gates below
pass.

The complete backend deliberately uses three different operations:

| Phase | Implementation | Reason |
| --- | --- | --- |
| Contiguous prefill | This repository's Triton FlashAttention | The teaching target |
| Paged prefix-cache prefill | External `flash-attn` | K/V are non-contiguous pages |
| Paged decode | External `flash_attn_with_kvcache` | This is paged attention, not prefill FlashAttention |
| KV-cache write | Existing Triton kernel | Already isolated in the baseline |

This distinction matters: “FlashAttention” describes an IO-aware tiled
attention algorithm, while “paged attention” describes how decode reads a
logically contiguous sequence from physical KV-cache pages. A full inference
engine needs both.

## The problem being solved

Ordinary attention conceptually forms

$$
S = QK^{\mathsf{T}}, \qquad P = \operatorname{softmax}(S), \qquad O = PV.
$$

For sequence length $N$, storing $S$ and $P$ costs $O(N^2)$ memory traffic.
FlashAttention keeps one query tile and streams K/V tiles through on-chip
memory. It never writes the full score matrix to HBM.

For every query row, the kernel maintains:

- running maximum $m$;
- running softmax denominator $l$;
- unnormalised output accumulator $o$.

When a new score tile $S_j$ arrives:

$$
m' = \max(m, \operatorname{rowmax}(S_j)),
$$

$$
\alpha = e^{m-m'}, \qquad P_j = e^{S_j-m'},
$$

$$
l' = \alpha l + \operatorname{rowsum}(P_j),
$$

$$
o' = \alpha o + P_jV_j.
$$

After all K/V tiles, the result is $O=o/l$. This recurrence makes softmax
numerically stable even though tiles are processed independently.

## How the Triton program maps to the GPU

The launch grid is:

```text
(query tiles, query heads, sequences)
```

One Triton program therefore owns `BLOCK_M` query rows from one head and one
sequence. The program:

1. Loads its Q tile once.
2. Maps query head to KV head for GQA.
3. Loops over `BLOCK_N`-wide K/V tiles.
4. Uses `tl.dot` for $QK^T$ and $PV$.
5. Applies packed-sequence bounds and the causal mask.
6. Updates online-softmax state in FP32.
7. Stores one output tile in the input dtype.

The causal position for a chunked prefill query is not simply its local row. If
the query chunk is the suffix of a longer key sequence, it is:

$$
p_q = N_K - N_Q + p_{q,mathrm{local}}.
$$

That is why the kernel consumes both `cu_seqlens_q` and `cu_seqlens_k`.

## Why it is an entry project, not a production claim

The implementation is intentionally compact and inspectable. Production Flash
Attention additionally tunes tile shapes per architecture, uses more advanced
warp partitioning and pipelining, handles more head dimensions and masks, and
has backward kernels. Therefore the learning progression is:

1. Prove correctness against a PyTorch reference.
2. Compare with external FlashAttention on the same Qwen prefill workload.
3. Profile HBM traffic, occupancy, register pressure, and tile configurations.
4. Add autotuning and architecture-specific schedules.
5. Implement paged decode separately.

## RTX 3060 validation

```bash
NANOVLLM_TEST_MODEL=~/models/Qwen3-0.6B \
  uv run --locked --extra nvidia pytest -o addopts="-ra" -m gpu \
  tests/gpu -k 'triton_flash_attention'

uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode eager --workload quick \
  --attention-backend flash_attention \
  --output artifacts/gpu-baseline/flash-attn-quick.json

uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode eager --workload quick \
  --attention-backend triton_flash_attention \
  --compare-to artifacts/gpu-baseline/flash-attn-quick.json \
  --output artifacts/gpu-baseline/triton-flash-quick.json
```

The test command validates the kernel numerically against PyTorch. The two
baseline commands then exercise the real engine and require every generated
token ID to match the external FlashAttention path. Only after both gates pass
should throughput or latency numbers be interpreted.

## What the CUDA version would change

The algorithm stays the same, but CUDA requires explicit thread/warp mapping,
shared-memory layouts, synchronization, reductions, vectorized loads, tensor-core
MMA selection, C++/PyTorch bindings, and architecture dispatch. The outline is:

```cpp
// Teaching sketch, not a complete kernel.
template <typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_forward(Params p) {
    // 1. Cooperatively copy a Q tile to shared memory.
    // 2. Stream K/V tiles: QK^T -> causal mask -> online softmax -> P @ V.
    // 3. Warp-reduce m/l, normalize the FP32 accumulator, and store O.
}
```

Triton supplies tile tensors, masked loads, reductions, and `tl.dot`, so it is
usually the better first kernel project. CUDA/CUTLASS is valuable when exact
shared-memory, warp, instruction, or architecture control is the point of the
project. Production inference stacks commonly use both.
