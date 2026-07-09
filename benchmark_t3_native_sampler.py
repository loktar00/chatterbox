#!/usr/bin/env python3
"""Build and run the standalone native T3 sampler microbench."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/root/chatterbox")
SRC = ROOT / "bench_t3_native_sampler.cpp"
BIN = ROOT / "bench_t3_native_sampler"
PY_PROFILE = ROOT / "exports/benchmarks/t3_sampler_candidate_profile_2026-07-08.json"
OUT_JSON = ROOT / "exports/benchmarks/t3_native_sampler_microbench_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/t3_native_sampler_microbench_2026-07-08.md"


def run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def load_python_profile() -> dict[str, Any] | None:
    if not PY_PROFILE.exists():
        return None
    return json.loads(PY_PROFILE.read_text())


def py_current_per_trial_ms(profile: dict[str, Any] | None, seen_len: int) -> float | None:
    if not profile:
        return None
    for case in profile.get("cases", []):
        if case.get("seen_len") == seen_len:
            current = case.get("variants", {}).get("current_full_vocab", {})
            total = current.get("total_seconds")
            trials = case.get("trials")
            if isinstance(total, (int, float)) and isinstance(trials, int) and trials > 0:
                return total * 1000.0 / trials
    return None


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# T3 Native Sampler Microbench - 2026-07-08",
        "",
        f"- Status: `{report['status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "| Seen Len | Native ms/trial | Python Current ms/trial | Native Speedup | TopK | TopP | Repetition | Softmax+Sample |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report.get("cases", []):
        lines.append(
            f"| {case['seen_len']} | `{case['native_per_trial_ms']:.4f}` | "
            f"`{case['python_current_per_trial_ms']:.4f}` | "
            f"`{case['native_speedup_vs_python_current']:.2f}x` | "
            f"`{case['native_breakdown_per_trial_ms']['topk']:.4f}` | "
            f"`{case['native_breakdown_per_trial_ms']['top_p']:.4f}` | "
            f"`{case['native_breakdown_per_trial_ms']['repetition']:.4f}` | "
            f"`{case['native_breakdown_per_trial_ms']['softmax_sample']:.4f}` |"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    compiler = shutil.which("g++") or shutil.which("c++")
    report: dict[str, Any] = {
        "description": "Native C++ T3 sampler microbench; no Chatterbox model load and no Vulkan execution.",
        "source": str(SRC),
        "binary": str(BIN),
        "compiler": compiler,
        "status": "not_started",
        "cases": [],
        "recommendation": "none",
        "decision": "",
    }
    if compiler is None:
        report["status"] = "missing_compiler"
        report["decision"] = "No compiler found; native sampler benchmark was not run."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    compile_cmd = [
        compiler,
        "-O3",
        "-march=native",
        "-std=c++17",
        SRC.as_posix(),
        "-o",
        BIN.as_posix(),
    ]
    compiled = run(compile_cmd, timeout=60.0)
    report["compile"] = {
        "cmd": compile_cmd,
        "returncode": compiled.returncode,
        "stdout": compiled.stdout,
        "stderr": compiled.stderr,
    }
    if compiled.returncode != 0:
        report["status"] = "compile_failed"
        report["decision"] = "Native sampler benchmark did not compile."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    bench = run([BIN.as_posix(), "10000", "8192"], timeout=60.0)
    report["run"] = {"returncode": bench.returncode, "stderr": bench.stderr}
    if bench.returncode != 0:
        report["status"] = "run_failed"
        report["decision"] = "Native sampler benchmark failed at runtime."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    native = json.loads(bench.stdout)
    py_profile = load_python_profile()
    report["native_raw"] = native
    report["python_profile"] = str(PY_PROFILE) if py_profile else None
    speedups = []
    for case in native.get("cases", []):
        seen_len = int(case["seen_len"])
        py_ms = py_current_per_trial_ms(py_profile, seen_len)
        native_ms = float(case["per_trial_ms"]["total"])
        if py_ms is None:
            continue
        speedup = py_ms / native_ms if native_ms > 0 else None
        speedups.append(speedup or 0.0)
        report["cases"].append(
            {
                "seen_len": seen_len,
                "unique_len": case["unique_len"],
                "native_per_trial_ms": native_ms,
                "python_current_per_trial_ms": py_ms,
                "native_speedup_vs_python_current": speedup,
                "native_breakdown_per_trial_ms": case["per_trial_ms"],
            }
        )

    min_speedup = min(speedups) if speedups else None
    report["min_native_speedup_vs_python_current"] = min_speedup
    report["status"] = "ok"
    if min_speedup is not None and min_speedup >= 2.0:
        report["recommendation"] = "native_sampler_worth_bridge_prototype"
        report["decision"] = (
            "Native sampling is fast enough to justify a guarded bridge prototype, but it must be distribution-validated against the Python sampler before runtime use."
        )
    else:
        report["recommendation"] = "no_runtime_change"
        report["decision"] = (
            "Native sampling does not show enough standalone speedup to prioritize bridge integration over larger T3 dispatch/fusion work."
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"status={report['status']}")
    print(f"recommendation={report['recommendation']}")
    if min_speedup is not None:
        print(f"min_native_speedup_vs_python_current={min_speedup:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
