#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nanovllm.benchmarks.gpu_environment import collect_environment, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the Linux NVIDIA environment before nano-vLLM model execution."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON report path (for example artifacts/gpu-baseline/environment.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_environment()
    if args.output:
        write_json_report(report, args.output)

    print("nano-vLLM GPU environment check")
    for check in report["checks"]:
        print(f"[{check['status'].upper():4}] {check['name']}: {check['detail']}")
    print(f"ready: {report['ready']}")
    if args.output:
        print(f"report: {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
