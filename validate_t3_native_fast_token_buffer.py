#!/usr/bin/env python3
"""Validate the native-sampler fast token-buffer T3 path.

This is a T3-only check. It does not generate audio, start an API worker, or
run ROCm/HIP. The script shells out to the existing T3 benchmark with a tiny
hello prompt, then verifies that loop telemetry proves the optimized branch was
used and produced stable tokens across repeated requests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "benchmark_t3_runtime_env_sweep.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
OUT_DIR = ROOT / "exports" / "benchmarks"


def now_label() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def first_runs(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results") or []
    if not results:
        return []
    return results[0].get("runs") or []


def summarize(report: dict[str, Any], bench_output: Path, proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    runs = first_runs(report)
    loops = [run.get("loop_info") or {} for run in runs]
    all_equal = bool(runs) and all(bool(run.get("tokens_equal_reference")) for run in runs)
    native_fast = bool(loops) and all(loop.get("native_fast_token_buffer") is True for loop in loops)
    native_sampler = bool(loops) and all(loop.get("native_sampler") is True for loop in loops)
    no_logits_to_torch = bool(loops) and all(
        ((loop.get("timings") or {}).get("step_logits_to_torch_seconds") in (0, 0.0))
        for loop in loops
    )
    generated_tokens = [loop.get("generated_tokens") for loop in loops]
    ok = proc.returncode == 0 and all_equal and native_fast and native_sampler and no_logits_to_torch
    return {
        "status": "ok" if ok else "failed",
        "ok": ok,
        "description": "T3-only native fast token-buffer validation. No audio generation, worker start, or ROCm/HIP probing.",
        "benchmark_output": bench_output.as_posix(),
        "benchmark_returncode": proc.returncode,
        "request_count": len(runs),
        "all_tokens_equal_reference": all_equal,
        "native_sampler": native_sampler,
        "native_fast_token_buffer": native_fast,
        "no_logits_to_torch_copy": no_logits_to_torch,
        "seconds": [run.get("seconds") for run in runs],
        "generated_tokens": generated_tokens,
        "loop_iterations": [loop.get("loop_iterations") for loop in loops],
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# T3 Native Fast Token Buffer Validation",
        "",
        f"- Status: `{summary['status']}`",
        f"- Benchmark output: `{summary['benchmark_output']}`",
        f"- Requests: `{summary['request_count']}`",
        f"- Tokens equal reference: `{summary['all_tokens_equal_reference']}`",
        f"- Native sampler: `{summary['native_sampler']}`",
        f"- Native fast token buffer: `{summary['native_fast_token_buffer']}`",
        f"- Logits-to-Torch copy avoided: `{summary['no_logits_to_torch_copy']}`",
        f"- Seconds: `{summary['seconds']}`",
        f"- Generated tokens: `{summary['generated_tokens']}`",
        "",
        "This check uses only the T3 token path. It does not generate audio, start an API worker, or run ROCm/HIP.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--prefill-threads", type=int, default=8)
    parser.add_argument("--max-gen-len", type=int, default=80)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or OUT_DIR / f"t3_native_fast_token_buffer_validation_{now_label()}.json"
    bench_output = output.with_name(output.stem + "_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    py = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        py.as_posix(),
        BENCH.as_posix(),
        "--case",
        "hello",
        "--requests",
        str(args.requests),
        "--max-gen-len",
        str(args.max_gen_len),
        "--prefill-threads",
        str(args.prefill_threads),
        "--prefix-prefill",
        "--output",
        bench_output.as_posix(),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env["CHATTERBOX_T3_NATIVE_SAMPLER"] = "1"
    env.setdefault("CHATTERBOX_PROGRESS", "0")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    if bench_output.exists():
        report = json.loads(bench_output.read_text())
    else:
        report = {"results": []}
    summary = summarize(report, bench_output, proc)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, output.with_suffix(".md"))

    print(f"json={output}")
    print(f"markdown={output.with_suffix('.md')}")
    print(f"benchmark_json={bench_output}")
    print(f"status={summary['status']}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
