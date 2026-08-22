#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    prompt_length: int
    output_length: int
    concurrency: int
    warmups: int
    repetitions: int


WORKLOADS = {
    "smoke": Workload("smoke", 32, 8, 1, 0, 1),
    "quick": Workload("quick", 128, 32, 4, 1, 3),
    "standard": Workload("standard", 256, 64, 8, 1, 5),
}
SUPPORTED_MODEL_ARCHITECTURES = {
    "Qwen3ForCausalLM",
    "Qwen3MiniMoEForCausalLM",
}


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conservative single-GPU nano-vLLM smoke test or baseline."
    )
    parser.add_argument("--model", required=True, type=Path, help="local Qwen3 model directory")
    parser.add_argument("--output", required=True, type=Path, help="JSON result path")
    parser.add_argument("--mode", choices=("eager", "cudagraph"), default="eager")
    parser.add_argument(
        "--attention-backend",
        choices=("flash_attention", "triton_flash_attention"),
        default="flash_attention",
    )
    parser.add_argument("--workload", choices=tuple(WORKLOADS), default="quick")
    parser.add_argument("--prompt-length", type=_positive_int)
    parser.add_argument("--output-length", type=_positive_int)
    parser.add_argument("--concurrency", type=_positive_int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repetitions", type=_positive_int)
    parser.add_argument("--max-model-len", type=_positive_int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=1024)
    parser.add_argument("--max-num-seqs", type=_positive_int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="optional prior report; fail if any measured output token IDs differ",
    )
    return parser.parse_args()


def resolve_workload(args: argparse.Namespace) -> Workload:
    preset = WORKLOADS[args.workload]
    workload = Workload(
        name=args.workload,
        prompt_length=args.prompt_length or preset.prompt_length,
        output_length=args.output_length or preset.output_length,
        concurrency=args.concurrency or preset.concurrency,
        warmups=preset.warmups if args.warmups is None else args.warmups,
        repetitions=args.repetitions or preset.repetitions,
    )
    if workload.warmups < 0:
        raise ValueError("warmups cannot be negative")
    if workload.prompt_length + workload.output_length > args.max_model_len:
        raise ValueError("prompt_length + output_length must not exceed max_model_len")
    if workload.concurrency > args.max_num_seqs:
        raise ValueError("concurrency exceeds max_num_seqs")
    if workload.prompt_length * workload.concurrency > args.max_num_batched_tokens:
        raise ValueError(
            "prompt_length * concurrency exceeds max_num_batched_tokens; "
            "raise the token budget or reduce the workload"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    return workload


def _git_revision(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _aggregate(measurements: list[dict[str, Any]]) -> dict[str, float]:
    wall_times = [item["wall_time_seconds"] for item in measurements]
    throughputs = [item["output_tokens_per_second"] for item in measurements]
    ttfts = [item["ttft_seconds"] for item in measurements]
    tpots = [item["tpot_seconds"] for item in measurements]
    return {
        "wall_time_mean_seconds": statistics.fmean(wall_times),
        "wall_time_p50_seconds": statistics.median(wall_times),
        "wall_time_p95_seconds": _percentile(wall_times, 0.95),
        "output_throughput_mean_tokens_per_second": statistics.fmean(throughputs),
        "output_throughput_p50_tokens_per_second": statistics.median(throughputs),
        "output_throughput_p95_tokens_per_second": _percentile(throughputs, 0.95),
        "ttft_mean_seconds": statistics.fmean(ttfts),
        "ttft_p95_seconds": _percentile(ttfts, 0.95),
        "tpot_mean_seconds": statistics.fmean(tpots),
        "tpot_p95_seconds": _percentile(tpots, 0.95),
        "peak_allocated_memory_gib_max": max(item["peak_allocated_memory_gib"] for item in measurements),
        "peak_reserved_memory_gib_max": max(item["peak_reserved_memory_gib"] for item in measurements),
    }


def compare_token_outputs(
    left_report: dict[str, Any],
    right_report: dict[str, Any],
) -> dict[str, Any]:
    left_model = left_report.get("comparison_model", left_report.get("model"))
    right_model = right_report.get("comparison_model", right_report.get("model"))
    # Physical cache capacity varies with the process and the attention backend
    # is the implementation under test. Neither changes the logical workload.
    ignored_engine_fields = {"attention_backend", "num_kvcache_blocks"}
    left_engine = {
        key: value for key, value in left_report.get("engine", {}).items()
        if key not in ignored_engine_fields
    }
    right_engine = {
        key: value for key, value in right_report.get("engine", {}).items()
        if key not in ignored_engine_fields
    }
    configuration_matches = (
        left_model == right_model
        and left_report.get("workload") == right_report.get("workload")
        and left_engine == right_engine
    )
    left = [item["output_token_ids"] for item in left_report.get("measurements", [])]
    right = [item["output_token_ids"] for item in right_report.get("measurements", [])]
    comparable = configuration_matches and bool(left) and len(left) == len(right)
    matching_iterations = sum(
        left_tokens == right_tokens
        for left_tokens, right_tokens in zip(left, right)
    )
    return {
        "comparable": comparable,
        "configuration_matches": configuration_matches,
        "left_iterations": len(left),
        "right_iterations": len(right),
        "matching_iterations": matching_iterations,
        "all_token_ids_match": comparable and matching_iterations == len(left),
    }


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_prompts(tokenizer: Any, workload: Workload, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    vocab_size = len(tokenizer)
    special_ids = set(tokenizer.all_special_ids)
    candidates = [token for token in range(vocab_size) if token not in special_ids]
    if not candidates:
        raise RuntimeError("tokenizer does not expose any non-special tokens")
    return [
        [rng.choice(candidates) for _ in range(workload.prompt_length)]
        for _ in range(workload.concurrency)
    ]


def _validate_model_architecture(model: Path) -> None:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    architectures = config.architectures or []
    if not SUPPORTED_MODEL_ARCHITECTURES.intersection(architectures):
        requested = ", ".join(architectures) or config.model_type or "<missing>"
        supported = ", ".join(sorted(SUPPORTED_MODEL_ARCHITECTURES))
        raise ValueError(
            f"unsupported model architecture {requested}; this baseline supports "
            f"{supported} (recommended base: Qwen/Qwen3-0.6B)"
        )
    return config


def _comparison_model(model: Path, model_config: Any) -> str:
    """Resolve the dense checkpoint identity used for parity reports."""

    base_model = getattr(model_config, "mini_moe_base_model", None)
    if base_model is None:
        return str(model.resolve())
    base_path = Path(base_model).expanduser()
    if not base_path.is_absolute():
        base_path = model / base_path
    return str(base_path.resolve())


def _generate_measured(
    llm: Any,
    prompts: list[list[int]],
    sampling: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for prompt in prompts:
        llm.add_request(prompt, sampling)
    outputs: dict[int, tuple[int, ...]] = {}
    prefill_seconds = 0.0
    decode_seconds: list[float] = []
    scheduled_prefill_tokens = 0
    scheduled_decode_tokens = 0
    preemptions = 0
    peak_cache_utilization = 0.0
    while not llm.is_finished():
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        result = llm.step()
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - step_started
        peak_cache_utilization = max(peak_cache_utilization, result.cache_stats.utilization)
        preemptions += len(result.preempted_seq_ids)
        if result.mode.is_prefill:
            prefill_seconds += elapsed_seconds
            scheduled_prefill_tokens += result.num_scheduled_tokens
        else:
            decode_seconds.append(elapsed_seconds)
            scheduled_decode_tokens += result.num_scheduled_tokens
        for output in result.outputs:
            outputs[output.request_id] = output.token_ids
    ordered = [outputs[request_id] for request_id in sorted(outputs)]
    decoded = [
        {"token_ids": list(token_ids), "text": llm.tokenizer.decode(token_ids)}
        for token_ids in ordered
    ]
    return decoded, {
        "ttft_seconds": prefill_seconds,
        "tpot_seconds": statistics.fmean(decode_seconds) if decode_seconds else 0.0,
        "decode_step_count": len(decode_seconds),
        "scheduled_prefill_tokens": scheduled_prefill_tokens,
        "scheduled_decode_tokens": scheduled_decode_tokens,
        "preemptions": preemptions,
        "peak_logical_kv_cache_utilization": peak_cache_utilization,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    workload = resolve_workload(args)
    if not args.model.is_dir():
        raise ValueError(f"model directory does not exist: {args.model}")

    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run scripts/gpu_env.py first")

    model_config = _validate_model_architecture(args.model)
    if (
        getattr(model_config, "mini_moe_implementation", None) == "sparse_dispatch"
        and args.mode != "eager"
    ):
        raise ValueError("Mini-MoE sparse_dispatch requires --mode eager")
    _seed_everything(torch, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts = _build_prompts(tokenizer, workload, args.seed)
    sampling = SamplingParams(
        temperature=1e-5,
        max_tokens=workload.output_length,
        ignore_eos=True,
    )

    init_started = time.perf_counter()
    llm = LLM(
        str(args.model),
        enforce_eager=args.mode == "eager",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
    )
    init_seconds = time.perf_counter() - init_started
    warmup_measurements = []
    measurements = []
    try:
        for iteration in range(workload.warmups + workload.repetitions):
            _seed_everything(torch, args.seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs, step_metrics = _generate_measured(llm, prompts, sampling, torch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output_tokens = sum(len(output["token_ids"]) for output in outputs)
            expected_tokens = workload.output_length * workload.concurrency
            if len(outputs) != workload.concurrency or output_tokens != expected_tokens:
                raise RuntimeError(
                    f"unexpected output shape: {len(outputs)} requests/{output_tokens} tokens; "
                    f"expected {workload.concurrency}/{expected_tokens}"
                )
            item = {
                "iteration": iteration,
                "wall_time_seconds": elapsed,
                "input_tokens": workload.prompt_length * workload.concurrency,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / elapsed,
                "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_memory_gib": torch.cuda.max_memory_reserved() / 2**30,
                "sample_token_ids": outputs[0]["token_ids"],
                "sample_text": outputs[0]["text"],
                "output_token_ids": [output["token_ids"] for output in outputs],
                **step_metrics,
            }
            (warmup_measurements if iteration < workload.warmups else measurements).append(item)
    finally:
        llm.exit()

    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_revision(REPO_ROOT),
        "platform": {"system": platform.system(), "release": platform.release()},
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        },
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda},
        "model": str(args.model.resolve()),
        "comparison_model": _comparison_model(args.model, model_config),
        "mode": args.mode,
        "workload": asdict(workload),
        "engine": {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": 1,
            "attention_backend": args.attention_backend,
            "num_kvcache_blocks": llm.config.num_kvcache_blocks,
        },
        "initialization_seconds": init_seconds,
        "warmups": warmup_measurements,
        "measurements": measurements,
        "summary": _aggregate(measurements),
        "metric_scope": {
            "wall_time": "end-to-end engine stepping time for a closed batch",
            "throughput": "completed output tokens divided by wall time",
            "ttft": "sum of synchronized prefill step latencies; closed-batch approximation",
            "tpot": "mean synchronized decode iteration latency; batch-level inter-token latency",
        },
    }


def main() -> int:
    args = parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"result: {args.output}")
    if args.compare_to is not None:
        comparison = compare_token_outputs(
            json.loads(args.compare_to.read_text(encoding="utf-8")),
            report,
        )
        print(json.dumps({"comparison": comparison}, indent=2, sort_keys=True))
        if not comparison["all_token_ids_match"]:
            return 1
    return 0


def compare_main(left: Path, right: Path, output: Path | None = None) -> int:
    comparison = compare_token_outputs(
        json.loads(left.read_text(encoding="utf-8")),
        json.loads(right.read_text(encoding="utf-8")),
    )
    rendered = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if comparison["all_token_ids_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
