from __future__ import annotations

import importlib.metadata
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_PACKAGES = ("torch", "triton", "flash-attn", "transformers")
REQUIRED_IMPORTS = {
    "triton": "triton",
    "flash-attn": "flash_attn",
    "transformers": "transformers",
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True


def _run(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": None, "error": str(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _import_checks() -> list[CheckResult]:
    checks = []
    for package, module_name in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            checks.append(CheckResult(f"import:{package}", "fail", repr(exc)))
        else:
            checks.append(CheckResult(f"import:{package}", "pass", module_name))
    return checks


def _flash_attention_runtime_check(torch: Any) -> CheckResult:
    """Compile and execute the exact FlashAttention APIs used by the engine."""

    if not torch.cuda.is_available():
        return CheckResult(
            "flash_attention_runtime",
            "fail",
            "CUDA is unavailable",
        )
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

        dtype = torch.float16
        query = torch.randn(1, 1, 128, device="cuda", dtype=dtype)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        cumulative = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cumulative,
            cu_seqlens_k=cumulative,
            max_seqlen_q=1,
            max_seqlen_k=1,
            causal=True,
        )
        key_cache = torch.randn(1, 256, 1, 128, device="cuda", dtype=dtype)
        value_cache = torch.randn_like(key_cache)
        cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
        block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        flash_attn_with_kvcache(
            query.unsqueeze(1),
            key_cache,
            value_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )
        torch.cuda.synchronize()
    except Exception as exc:
        return CheckResult("flash_attention_runtime", "fail", repr(exc))
    return CheckResult(
        "flash_attention_runtime",
        "pass",
        "varlen prefill and paged KV-cache decode calls succeeded",
    )


def _torch_report() -> tuple[dict[str, Any], list[CheckResult]]:
    checks: list[CheckResult] = []
    try:
        import torch
    except Exception as exc:
        checks.append(CheckResult("torch_import", "fail", repr(exc)))
        return {"import_error": repr(exc)}, checks

    available = torch.cuda.is_available()
    checks.append(CheckResult(
        "cuda_available",
        "pass" if available else "fail",
        f"torch.cuda.is_available()={available}",
    ))
    report: dict[str, Any] = {
        "version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count() if available else 0,
        "cuda_graph_api_available": bool(
            hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph")
        ),
    }
    if not available:
        return report, checks

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": round(properties.total_memory / 2**30, 2),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        })
    report["devices"] = devices
    checks.append(CheckResult(
        "compute_capability",
        "pass" if devices[0]["compute_capability"] >= [8, 0] else "fail",
        f"device 0 capability={tuple(devices[0]['compute_capability'])}; FlashAttention 2 requires Ampere or newer",
    ))
    checks.append(CheckResult(
        "cuda_graph_api",
        "pass" if report["cuda_graph_api_available"] else "fail",
        f"torch.cuda CUDA Graph API available={report['cuda_graph_api_available']}",
    ))
    try:
        tensor = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        del tensor
    except Exception as exc:
        checks.append(CheckResult("cuda_allocation", "fail", repr(exc)))
    else:
        checks.append(CheckResult("cuda_allocation", "pass", "allocated and synchronized a CUDA tensor"))
    checks.append(_flash_attention_runtime_check(torch))
    return report, checks


def _git_report(repo_root: Path) -> dict[str, Any]:
    revision = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return {
        "commit": revision.get("stdout") if revision.get("returncode") == 0 else None,
        "dirty": bool(status.get("stdout")) if status.get("returncode") == 0 else None,
    }


def collect_environment(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    system = platform.system()
    machine = platform.machine()
    checks = [
        CheckResult(
            "platform",
            "pass" if system == "Linux" and machine == "x86_64" else "fail",
            f"{system} {machine}; the NVIDIA runtime is supported in Linux x86_64 (use WSL2 on Windows)",
        ),
    ]
    versions = _package_versions()
    for package in REQUIRED_PACKAGES:
        checks.append(CheckResult(
            f"package:{package}",
            "pass" if versions[package] is not None else "fail",
            versions[package] or "not installed",
        ))
    checks.extend(_import_checks())
    torch_report, torch_checks = _torch_report()
    checks.extend(torch_checks)
    nvidia_smi = (
        _run(["nvidia-smi"])
        if shutil.which("nvidia-smi")
        else {"error": "not found"}
    )
    checks.append(CheckResult(
        "nvidia_smi",
        "pass" if nvidia_smi.get("returncode") == 0 else "fail",
        (nvidia_smi.get("stdout") or nvidia_smi.get("stderr") or nvidia_smi.get("error", "failed")).splitlines()[0],
    ))
    wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
    report = {
        "schema_version": 1,
        "system": {
            "platform": system,
            "release": platform.release(),
            "machine": machine,
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "is_wsl": wsl,
            "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
        },
        "git": _git_report(root),
        "packages": versions,
        "torch": torch_report,
        "commands": {
            "nvidia_smi": nvidia_smi,
            "uv": _run(["uv", "--version"]) if shutil.which("uv") else {"error": "not found"},
        },
        "checks": [asdict(check) for check in checks],
    }
    report["ready"] = all(
        check.status == "pass" for check in checks if check.required
    )
    return report


def write_json_report(report: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
