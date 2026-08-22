# Teaching implementation: Qwen3 Mini-MoE

## What was added

`MiniMoE` implements the complete inference data flow of a top-k
mixture-of-experts feed-forward layer:

```text
hidden states -> router -> top-k experts -> expert MLPs -> weighted combine
```

The model registry also contains `Qwen3MiniMoEForCausalLM`. A conversion script
creates a small config overlay on top of an existing Qwen3 checkpoint. Selected
dense MLP layers become two or more experts while attention, scheduler, KV
cache, sampling, and tokenizer remain unchanged.

This is a model-architecture experiment, not support for Qwen3-MoE or another
vendor MoE checkpoint. Those checkpoints have their own tensor names, expert
layouts and architectural details.
The integration has not yet run with a real checkpoint on NVIDIA hardware; the
RTX 3060 parity gates below are required before claiming end-to-end validation.

## Routing math

For a token vector $x$, the router produces logits

$$
r = W_rx.
$$

It selects the top-$k$ expert IDs $e_i$ and renormalises only their logits:

$$
g_i = rac{e^{r_{e_i}}}{\sum_{j=1}^{k} e^{r_{e_j}}}.
$$

The MoE output is

$$
y = \sum_{i=1}^{k} g_i E_{e_i}(x),
$$

where each $E_e$ is the same SwiGLU MLP shape used by Qwen3.

## Why the converted checkpoint initially preserves the dense result

The Qwen checkpoint has only one MLP tensor set per layer. During loading, the
custom name-expansion hook copies each dense tensor into every expert. The
router is initialised to zero, so selected expert gates sum to one. Because all
experts initially implement the same function $E$:

$$
\sum_i g_i E(x) = E(x)\sum_i g_i = E(x).
$$

This gives a useful parity checkpoint before training or intentionally
diversifying experts. It does **not** create a trained MoE: useful expert
specialisation requires training/fine-tuning and a load-balancing objective.

## Two execution modes

| Mode | Behaviour | Value | Limitation |
| --- | --- | --- | --- |
| `dense_masked` | Runs every expert, masks and combines selected outputs | Simple, deterministic, static-shape, CUDA-Graph friendly | No compute saving |
| `sparse_dispatch` | Uses `where`, gathers selected tokens, runs each expert, `index_add_` combines | Demonstrates actual sparse dispatch | Python loop, dynamic shapes, eager-only |

Production inference sorts `(expert_id, token_id)` assignments, computes token
offsets for every expert, runs grouped GEMM, then scatters outputs back. That is
the natural next custom Triton/CUDA project. The current code does not claim
grouped-GEMM performance.

## Create an overlay and run it

Start with one converted layer on an RTX 3060. The commands use conservative
limits suitable for either common 6-GB laptop or 12-GB desktop variants; reduce
`gpu_memory_utilization` if other processes occupy VRAM.

```bash
uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode eager --workload smoke \
  --output artifacts/gpu-baseline/dense-smoke.json

uv run --locked --extra nvidia python scripts/convert_qwen3_to_mini_moe.py \
  --source ~/models/Qwen3-0.6B \
  --output ~/models/Qwen3-0.6B-mini-moe \
  --layers 0 --num-experts 4 --top-k 2 \
  --implementation dense_masked

uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B-mini-moe \
  --mode eager --workload smoke \
  --compare-to artifacts/gpu-baseline/dense-smoke.json \
  --output artifacts/gpu-baseline/mini-moe-smoke.json
```

The Mini-MoE run exits non-zero unless every generated token ID matches the
dense run. The overlay records its dense source as the comparison identity, so
the comparator permits this intentional cross-directory check while rejecting
unrelated checkpoints.

The overlay stores a modified `config.json` and relative symlinks to the
original tokenizer and safetensors files; it does not duplicate checkpoint
files on disk. Runtime VRAM does increase because converted expert weights are
duplicated.

Inspect routing independently:

```bash
uv run --locked --extra nvidia python scripts/inspect_moe_routing.py \
  --tokens 128 --hidden-size 256 --experts 4 --top-k 2
```

Useful measurements are expert assignment counts, maximum-to-mean load, router
entropy, dispatch/combine time, expert GEMM time, and end-to-end TPOT. A router
that sends nearly every token to one expert is a correctness-valid but
performance-poor result.

Run the Mini-MoE GPU correctness test explicitly on WSL2:

```bash
NANOVLLM_TEST_MODEL=~/models/Qwen3-0.6B \
NANOVLLM_TEST_MINI_MOE_MODEL=~/models/Qwen3-0.6B-mini-moe \
  uv run --locked --extra nvidia pytest -o addopts="-ra" -m gpu \
  tests/gpu -k mini_moe
```

This runs both the isolated dense/sparse layer test and the dense-versus-overlay
engine token-parity test.

The normal macOS control-plane environment intentionally has no PyTorch, so the
numerical CPU test is skipped there. It runs automatically in an inference or
NVIDIA environment; conversion, source and registry tests remain CPU-only.

## Recommended learning sequence

1. Verify `dense_masked` and `sparse_dispatch` give equal outputs.
2. Verify the cloned-expert overlay matches the dense model's token outputs.
3. Perturb or train experts and observe routing/load changes.
4. Replace Python dispatch with a GPU histogram/prefix-sum/permutation.
5. Replace per-expert loops with Triton grouped GEMM.
6. Add capacity limits, overflow policy and load-balancing metrics.
