# RTX 5090 measurements

Measured on an NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.105.08, compute capability (12, 0) = Blackwell sm_120. Software: torch 2.7.0+cu128, Triton 3.4.0, flash-attn 2.8.3. The run used `feat/triton-moe-kernels` with the FlashAttention log2-scale constexpr fix (`09e6aef`) and `TORCHDYNAMO_DISABLE=1`. Triton 3.3.0 failed to compile `tl.dot` kernels for sm_120; 3.4.0 compiled them.

## Numeric pytest

`pytest -m gpu tests/gpu/test_teaching_kernels.py tests/gpu/test_triton_moe.py`: **11 passed**.

Triton FlashAttention prefill matched the PyTorch packed-GQA reference in fp16 and bf16, including the chunked-query-suffix causal case. Triton Mini-MoE grouped SwiGLU GEMM matched the PyTorch assignment reference in fp16 and bf16, including empty-expert and tile-tail cases. Tolerances were atol/rtol 2e-2 (bf16 4e-2).

## Engine token IDs

Short prompts (`Hello from nano-vLLM.` and `Count from one to five.`, 8 greedy tokens): `triton_flash_attention` token IDs matched `flash_attention`.

Harder `gpu_baseline` quick workload (128-token random prompts, 32 output tokens, 4 sequences): **52/642 tokens differed**. Two sequences diverged (11/32 and 2/32); the other two matched. The mismatch was deterministic across warmup and 3 iterations.

Mini-MoE cloned-expert end-to-end: **1/16 tokens differed** (15/16 matched; last token of sequence 2 was 8500 vs 2088), deterministic. A pure-PyTorch `sparse_reference` overlay matched the dense model exactly. The Triton grouped-GEMM `sparse_dispatch` overlay was the one that diverged, so routing was correct and the difference is fp16 rounding inside the numeric tolerance above.

## Throughput

Eager quick workload: `flash_attention` 224.98 tok/s, `triton_flash_attention` 224.96 tok/s, peak about 22.8 GiB. The token-ID gate for that workload failed (52/642), so ~225 tok/s is not a resume-usable throughput result.
