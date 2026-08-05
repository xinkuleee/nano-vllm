#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gpu_baseline import compare_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare all output token IDs in two GPU baseline reports.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return compare_main(args.left, args.right, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
