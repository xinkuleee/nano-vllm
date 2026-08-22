<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

For the original NVIDIA inference path on a Linux GPU server, start with:

```bash
uv sync --locked --extra nvidia
```

The exact PyTorch/CUDA/FlashAttention/Triton combination still needs to be pinned
and validated on the target server; the current checkpoint has only been tested
locally for CPU control-plane development.

### Local control-plane development (macOS/CPU)

The scheduler, logical KV-cache, prefix-cache, preemption, and worker contracts
can be developed without a GPU or GPU packages:

```bash
uv sync --locked
uv run --locked pytest
```

This environment intentionally does not install PyTorch, Triton, FlashAttention,
or CUDA. It validates runtime control logic, not model execution or GPU kernels.

The `inference` extra provides the framework/model dependencies without selecting
a GPU kernel stack. The `nvidia` extra includes those dependencies plus Triton
and FlashAttention for a Linux NVIDIA server.

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Runtime architecture

The extensible-runtime branch documents its scheduling, KV-cache, model, and
attention-backend boundaries in [docs/architecture.md](docs/architecture.md).
The completed local checkpoint, review evidence, and remaining validation matrix
are in [docs/phase1-summary.md](docs/phase1-summary.md).
The scope and honest resume/interview narrative are in
[docs/resume-and-interview.md](docs/resume-and-interview.md).
For an NVIDIA laptop, follow the reproducible
[Windows 11 + WSL2 GPU validation guide](docs/windows-wsl2-gpu.md); native
Windows is not a supported Triton/FlashAttention execution path.
The guide uses `scripts/gpu_env.py` and `scripts/gpu_baseline.py` to save
environment, eager/CUDA Graph correctness, latency, throughput, KV-cache, and
peak-memory evidence as local JSON artifacts.
Two teaching extensions build on that baseline:

- [Triton FlashAttention](docs/teaching-flash-attention.md) implements packed
  causal prefill with online softmax and plugs into `AttentionBackend`.
- [Qwen3 Mini-MoE](docs/teaching-mini-moe.md) implements top-k routing, dense
  and sparse dispatch, plus a checkpoint-compatible Qwen3 overlay.

Both extensions are currently teaching candidates: local control-plane checks
pass, while NVIDIA JIT compilation, end-to-end parity, and performance remain
gated by the documented WSL2/RTX 3060 runs.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
