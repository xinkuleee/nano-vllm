# Windows + WSL2 NVIDIA GPU validation

This is the supported laptop path for the Phase 2 GPU baseline. Run the
project **inside WSL2 Ubuntu**, not from native Windows PowerShell: the current
runtime requires Linux builds of Triton, FlashAttention, NCCL, and CUDA Graph.
An RTX 3060 is sufficient for single-GPU Qwen3-0.6B smoke tests and a reduced
baseline. It is not a multi-GPU or ROCm validation target.

## 1. Host setup

On Windows 11, install a current NVIDIA Windows driver with WSL CUDA support.
Do not install a Linux NVIDIA display driver inside WSL. From an elevated
PowerShell terminal, install or update WSL and Ubuntu, then restart if asked:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl --shutdown
```

Open Ubuntu and keep the repository in the WSL filesystem (for example under
`~/src`), not `/mnt/c`, because package builds and model reads are much faster
there.

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs python3-dev
mkdir -p ~/src
cd ~/src
git clone --branch feat/gpu-baseline \
  https://github.com/xinkuleee/nano-vllm.git
cd nano-vllm
```

The repository is private, so GitHub will require an authenticated credential.
Use a credential manager, SSH key, or a fine-grained token; do not put a token
in a command, script, or committed file.

## 2. Verify GPU passthrough first

These checks must succeed before installing Python packages:

```bash
nvidia-smi
uname -m
```

Expected architecture is `x86_64`, and `nvidia-smi` should show the RTX 3060.
The CUDA version printed by `nvidia-smi` is the driver's maximum supported
runtime version; it is not proof that `nvcc` is installed.

## 3. Install uv and the Python environment

Install uv using its official installer, restart the shell if necessary, and
confirm it is available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

First reproduce the CPU control-plane checkpoint:

```bash
uv sync --locked
uv run --locked pytest
```

Then install the NVIDIA extra:

```bash
uv sync --locked --extra nvidia
```

`flash-attn` is currently resolved from a source distribution. Its build needs
the matching PyTorch build to be installed and may need a CUDA toolkit with
`nvcc`; it is the most likely installation failure. If the one-step sync fails
there, preserve the full error and environment report instead of changing
random versions. The following is a diagnostic staged-install fallback:

```bash
uv sync --locked --extra inference
uv sync --locked --extra nvidia --no-build-isolation-package flash-attn
```

If the fallback reports that `nvcc` or CUDA headers are missing, install a CUDA
toolkit **inside WSL** that is compatible with the PyTorch wheel, but still do
not install a second Linux GPU driver. Record the exact toolkit package/version.
The current lock is an initial matrix, not yet a known-good RTX 3060 pin; do not
run `uv lock --upgrade` while establishing the first result.

## 4. Capture the environment report

Run the repository's GPU environment check from the `feat/gpu-baseline` branch,
then keep the generated JSON with the result artifacts:

```bash
uv run --locked --extra nvidia python scripts/gpu_env.py \
  --output artifacts/gpu-baseline/environment.json
```

At minimum the report should show:

- WSL/Linux `x86_64`, Python, uv, Git commit, and dirty-worktree state;
- GPU name, compute capability, VRAM, and NVIDIA driver;
- PyTorch, its compiled CUDA runtime, `torch.cuda.is_available()`, Triton, and
  FlashAttention versions;
- whether CUDA Graph and BF16 are available.

Before running model code, a direct PyTorch check is useful:

```bash
uv run --locked --extra nvidia python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

## 5. Download the model

Use the dense `Qwen/Qwen3-0.6B` checkpoint. The engine currently implements
`Qwen3ForCausalLM`; do not substitute an MoE or Qwen3.5 architecture for the
first baseline. The Hugging Face CLI is available through the locked
dependencies:

```bash
mkdir -p ~/models/Qwen3-0.6B
uv run --locked --extra nvidia hf download Qwen/Qwen3-0.6B \
  --local-dir ~/models/Qwen3-0.6B
```

## 6. Validation order on an RTX 3060

Use single-GPU settings throughout: `tensor_parallel_size=1`. Start with eager
execution and conservative limits. The upstream defaults (`16384` batched
tokens, `512` sequences, `4096` context, 90% VRAM) are not the first-run
configuration for a 6 GB laptop 3060. A 12 GB desktop card can be raised only
after the reduced run succeeds.

Run in this order:

1. GPU environment and dependency imports.
2. Existing CPU tests.
3. One short prompt in eager mode (`enforce_eager=True`) with roughly
   `max_model_len=1024`, `max_num_batched_tokens=1024`, `max_num_seqs=8`, and
   `gpu_memory_utilization=0.75`.
4. Deterministic eager repeatability/parity checks from the GPU baseline suite.
5. The same small case with CUDA Graph enabled (`enforce_eager=False`).
6. Reduced performance baseline, then longer-context or higher-concurrency
   sweeps while watching `nvidia-smi`.

The optional pytest smoke gates use the same conservative engine limits:

```bash
NANOVLLM_TEST_MODEL=~/models/Qwen3-0.6B \
  uv run --locked --extra nvidia pytest -o addopts="-ra" -m gpu tests/gpu
```

The repository provides one JSON-producing command for steps 3--6. Start with
the smallest eager case:

```bash
uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode eager \
  --workload smoke \
  --output artifacts/gpu-baseline/eager-smoke.json
```

Then run the guarded quick baseline in both execution modes:

```bash
uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode eager \
  --workload quick \
  --output artifacts/gpu-baseline/eager-quick.json

uv run --locked --extra nvidia python scripts/gpu_baseline.py \
  --model ~/models/Qwen3-0.6B \
  --mode cudagraph \
  --workload quick \
  --compare-to artifacts/gpu-baseline/eager-quick.json \
  --output artifacts/gpu-baseline/cudagraph-quick.json
```

The `quick` preset uses 128 prompt tokens, 32 output tokens, concurrency 4, one
warmup, and three measured iterations. The CLI rejects workloads that exceed
the configured model length, sequence count, or prefill token budget. The two
result files use the same random seed. Compare every request and iteration with
the repository helper:

```bash
uv run --locked --extra nvidia python scripts/compare_gpu_baselines.py \
  artifacts/gpu-baseline/eager-quick.json \
  artifacts/gpu-baseline/cudagraph-quick.json \
  --output artifacts/gpu-baseline/eager-vs-cudagraph.json
```

A zero exit status and `all_token_ids_match: true` establish eager/CUDA Graph
token parity for this workload.

Do not use `bench.py` unchanged for the first laptop run: it generates 256
requests with up to 1024 input and 1024 output tokens and only prints aggregate
throughput. It neither protects a 6 GB card from OOM nor captures TTFT, TPOT,
memory, environment, or correctness evidence.

## 7. What counts as a baseline result

A useful result directory contains the environment JSON, exact command and Git
commit, raw correctness/smoke output, raw benchmark records, and any failure
trace. Record at least model/dtype, eager or CUDA Graph mode, prompt/output
lengths, concurrency, TTFT, TPOT/inter-token latency, output throughput, wall
time, peak allocated/reserved GPU memory, logical KV-cache utilization, and
preemption count. Here TTFT is the sum of synchronized prefill steps and TPOT is
the mean synchronized decode-step latency for the closed batch; neither is a
network-serving latency. Run warmups separately and report
several measured iterations; never compare the upstream README's RTX 4070
number as if it were measured on this branch.

The first Windows run validates only the NVIDIA single-GPU path. CPU tests
passing on macOS does not imply GPU correctness: eager attention, KV-cache
writes, paged decode, CUDA Graph replay, numerical behavior, and memory sizing
all remain GPU-specific.

## Troubleshooting

- `nvidia-smi` fails in WSL: fix Windows driver/WSL passthrough before Python.
- `torch.cuda.is_available()` is false: capture PyTorch version,
  `torch.version.cuda`, driver output, and `uv pip freeze`; do not proceed.
- `flash-attn` build cannot find Torch: retry with the documented staged sync.
- `nvcc: command not found`: a source build needs the WSL CUDA toolkit; the
  Windows driver alone supplies GPU access, not necessarily the compiler.
- immediate OOM during runner warmup: lower `max_model_len`,
  `max_num_batched_tokens`, `max_num_seqs`, and memory utilization together.
- CUDA Graph fails but eager succeeds: save both records; this isolates capture
  or replay compatibility from the basic execution path.
- WSL process exits or the terminal freezes under load: check Windows Task
  Manager and WSL memory/swap limits as well as GPU VRAM.
