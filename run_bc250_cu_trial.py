#!/usr/bin/env python3
"""Run one guarded BC-250 CU trial and compare it to the saved baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
BASELINE = BENCH / "vulkan_hybrid_api_24cu_current_default_guarded_pre40_2026-07-08.json"


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_").lower()
    if not slug:
        raise SystemExit("trial label must contain at least one alphanumeric character")
    return slug


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial-label",
        required=True,
        help="Human label for this hardware setting, for example '28 enabled / 24 reported'.",
    )
    parser.add_argument(
        "--expected-visible-cu",
        type=int,
        default=24,
        help="RADV num_cu value required before the benchmark can run.",
    )
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--baseline-label", default="24-CU baseline")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment override for the temporary API, formatted KEY=VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--require-debug-field",
        action="append",
        default=[],
        help="Require this dot-path under /debug/last_request.last_request for every successful request.",
    )
    parser.add_argument("--date", default=dt.date.today().isoformat())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    label_slug = slugify(args.trial_label)
    date_slug = args.date
    result_json = BENCH / f"vulkan_hybrid_api_{label_slug}_guarded_{date_slug}.json"
    baseline_slug = slugify(args.baseline_label)
    comparison_json = BENCH / f"cu_comparison_{baseline_slug}_vs_{label_slug}_{date_slug}.json"

    for path in (result_json, comparison_json, comparison_json.with_suffix(".md")):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path}; pass --overwrite or choose a new --trial-label")

    benchmark_cmd = [
        sys.executable,
        "benchmark_40cu_gate.py",
        "--run",
        "--expected-cu",
        str(args.expected_visible_cu),
        "--requests",
        str(args.requests),
        "--artifact-label",
        label_slug,
        "--output",
        result_json.as_posix(),
    ]
    for item in args.env:
        benchmark_cmd.extend(["--env", item])
    for item in args.require_debug_field:
        benchmark_cmd.extend(["--require-debug-field", item])
    run(benchmark_cmd)

    result = load_json(result_json)
    if result.get("status") != "ok":
        print(f"benchmark status={result.get('status')}; not writing comparison")
        raise SystemExit(2)

    run(
        [
            sys.executable,
            "compare_cu_benchmarks.py",
            "--baseline",
            args.baseline.as_posix(),
            "--candidate",
            result_json.as_posix(),
            "--output",
            comparison_json.as_posix(),
            "--baseline-label",
            args.baseline_label,
            "--candidate-label",
            args.trial_label,
        ]
    )

    comparison = load_json(comparison_json)
    wall = comparison["candidate_stages"].get("wall")
    speedup = comparison["comparisons"].get("wall", {}).get("speedup")
    target = comparison["target_2x_seconds"]
    gap = wall - target if wall is not None else None
    print(f"result={result_json}")
    print(f"comparison={comparison_json}")
    print(f"comparison_md={comparison_json.with_suffix('.md')}")
    print(f"candidate_wall={wall:.3f}s")
    print(f"speedup_vs_baseline={speedup:.3f}x")
    print(f"gap_to_2x_target={gap:.3f}s")


if __name__ == "__main__":
    main()
