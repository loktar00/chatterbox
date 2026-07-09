#!/usr/bin/env python3
"""Benchmark the isolated C ABI native T3 sampler from Python ctypes."""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/chatterbox")
SRC = ROOT / "t3_native_sampler_bridge.cpp"
LIB = ROOT / "libt3_native_sampler_bridge.so"
PY_PROFILE = ROOT / "exports/benchmarks/t3_sampler_candidate_profile_2026-07-08.json"
OUT_JSON = ROOT / "exports/benchmarks/t3_native_sampler_bridge_ctypes_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/t3_native_sampler_bridge_ctypes_2026-07-08.md"


def run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def py_current_per_trial_ms(seen_len: int) -> float | None:
    if not PY_PROFILE.exists():
        return None
    data = json.loads(PY_PROFILE.read_text())
    for case in data.get("cases", []):
        if case.get("seen_len") == seen_len:
            total = case.get("variants", {}).get("current_full_vocab", {}).get("total_seconds")
            trials = case.get("trials")
            if isinstance(total, (int, float)) and isinstance(trials, int) and trials > 0:
                return total * 1000.0 / trials
    return None


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB.as_posix())
    lib.cb_t3_native_sample.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.cb_t3_native_sample.restype = ctypes.c_int
    return lib


def run_case(lib: ctypes.CDLL, seen_len: int, vocab: int, bank: int, epochs: int) -> dict[str, Any]:
    rng = np.random.default_rng(20260708 + seen_len)
    logits_bank = rng.standard_normal((bank, vocab), dtype=np.float32)
    input_ids = rng.integers(0, vocab, size=(seen_len,), dtype=np.int32)
    if seen_len > 4:
        input_ids[::7] = input_ids[0]
    out = ctypes.c_int32(-1)
    logits_ptrs = [
        row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        for row in logits_bank
    ]
    input_ptr = input_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    # One warm pass for CPU caches and dynamic linker paths.
    for index in range(min(16, bank)):
        rc = lib.cb_t3_native_sample(
            logits_ptrs[index],
            vocab,
            input_ptr,
            seen_len,
            1234 + index,
            0.8,
            1000,
            0.95,
            1.2,
            ctypes.byref(out),
        )
        if rc != 0:
            raise RuntimeError(f"native sampler warmup failed rc={rc}")

    started = time.perf_counter()
    checksum = 0
    calls = 0
    for epoch in range(epochs):
        for index in range(bank):
            rc = lib.cb_t3_native_sample(
                logits_ptrs[index],
                vocab,
                input_ptr,
                seen_len,
                20260708 + epoch * bank + index,
                0.8,
                1000,
                0.95,
                1.2,
                ctypes.byref(out),
            )
            if rc != 0:
                raise RuntimeError(f"native sampler failed rc={rc}")
            checksum += int(out.value) + 1
            calls += 1
    seconds = time.perf_counter() - started
    native_ms = seconds * 1000.0 / calls
    py_ms = py_current_per_trial_ms(seen_len)
    return {
        "seen_len": seen_len,
        "unique_len": int(np.unique(input_ids).size),
        "vocab": vocab,
        "calls": calls,
        "bank": bank,
        "epochs": epochs,
        "seconds": seconds,
        "native_ctypes_per_call_ms": native_ms,
        "python_current_per_trial_ms": py_ms,
        "native_ctypes_speedup_vs_python_current": (py_ms / native_ms if py_ms is not None and native_ms > 0 else None),
        "checksum": checksum,
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# T3 Native Sampler Bridge ctypes Benchmark - 2026-07-08",
        "",
        f"- Status: `{report['status']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "| Seen Len | ctypes ms/call | Python Current ms/trial | Speedup | Calls |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report.get("cases", []):
        lines.append(
            f"| {case['seen_len']} | `{case['native_ctypes_per_call_ms']:.4f}` | "
            f"`{case['python_current_per_trial_ms']:.4f}` | "
            f"`{case['native_ctypes_speedup_vs_python_current']:.2f}x` | {case['calls']} |"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    compiler = shutil.which("g++") or shutil.which("c++")
    report: dict[str, Any] = {
        "description": "ctypes benchmark for isolated native T3 sampler C ABI. No Chatterbox model load and no Vulkan execution.",
        "source": str(SRC),
        "library": str(LIB),
        "compiler": compiler,
        "status": "not_started",
        "cases": [],
        "recommendation": "none",
        "decision": "",
    }
    if compiler is None:
        report["status"] = "missing_compiler"
        report["decision"] = "No compiler found."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    compile_cmd = [
        compiler,
        "-O3",
        "-march=native",
        "-std=c++17",
        "-fPIC",
        "-shared",
        SRC.as_posix(),
        "-o",
        LIB.as_posix(),
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
        report["decision"] = "Native sampler C ABI failed to compile."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    lib = load_lib()
    cases = [run_case(lib, seen_len, 8192, bank=1000, epochs=10) for seen_len in (1, 64, 358)]
    report["cases"] = cases
    speedups = [case["native_ctypes_speedup_vs_python_current"] or 0.0 for case in cases]
    min_speedup = min(speedups)
    report["min_native_ctypes_speedup_vs_python_current"] = min_speedup
    report["status"] = "ok"
    if min_speedup >= 2.0:
        report["recommendation"] = "native_sampler_bridge_viable"
        report["decision"] = (
            "ctypes overhead is low enough to justify a guarded runtime prototype. Next step is distribution validation against Python logits on real T3 loop data, then opt-in audio benchmarking."
        )
    else:
        report["recommendation"] = "no_runtime_change"
        report["decision"] = "ctypes overhead erases too much native sampler benefit; do not wire."

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"status={report['status']}")
    print(f"recommendation={report['recommendation']}")
    print(f"min_native_ctypes_speedup_vs_python_current={min_speedup:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
