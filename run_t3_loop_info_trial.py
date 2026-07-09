#!/usr/bin/env python3
"""Prepare or run the guarded benchmark that captures T3 loop telemetry."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv/bin/python"
BASELINE = (
    ROOT
    / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run the temporary experimental Vulkan worker and benchmark.",
    )
    parser.add_argument("--expected-visible-cu", type=int, default=24)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trial-label",
        default="t3 loop info no watermark default quality",
        help="Label used for saved benchmark artifacts.",
    )
    args = parser.parse_args()

    cmd = [
        str(PYTHON if PYTHON.exists() else Path(sys.executable)),
        str(ROOT / "run_bc250_cu_trial.py"),
        "--trial-label",
        args.trial_label,
        "--expected-visible-cu",
        str(args.expected_visible_cu),
        "--requests",
        str(args.requests),
        "--baseline",
        str(BASELINE),
        "--baseline-label",
        "32 enabled / 24 reported no watermark default quality",
        "--env",
        "CHATTERBOX_APPLY_WATERMARK=0",
        "--require-debug-field",
        "t3_loop_info",
        "--require-debug-field",
        "t3_loop_info.timings.step_ggml_wall_seconds",
        "--require-debug-field",
        "t3_loop_info.timings.step_sampling_seconds",
    ]
    if args.overwrite:
        cmd.append("--overwrite")

    print(" ".join(shlex.quote(part) for part in cmd))
    if not args.run:
        print("Dry run only. Pass --run to execute the guarded benchmark.")
        return 0
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
