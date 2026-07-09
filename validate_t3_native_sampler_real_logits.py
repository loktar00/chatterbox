#!/usr/bin/env python3
"""Validate native sampler C ABI against saved real T3 logits."""

from __future__ import annotations

import ctypes
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/root/chatterbox")
FIXTURE = ROOT / "exports/ggml_t3_real_prompt_multistep_chunk270_s4_p935"
MANIFEST = FIXTURE / "manifest.json"
SRC = ROOT / "t3_native_sampler_bridge.cpp"
LIB = ROOT / "libt3_native_sampler_bridge.so"
OUT_JSON = ROOT / "exports/benchmarks/t3_native_sampler_real_logits_validation_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/t3_native_sampler_real_logits_validation_2026-07-08.md"

PARAMS = {
    "temperature": 0.8,
    "top_k": 1000,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
}


def run(cmd: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)


def compile_lib() -> dict[str, Any]:
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        return {"ok": False, "error": "missing compiler"}
    cmd = [
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
    proc = run(cmd)
    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB.as_posix())
    lib.cb_t3_native_filter_probs.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.cb_t3_native_filter_probs.restype = ctypes.c_int
    return lib


def python_probs(logits_np: np.ndarray, input_ids_np: np.ndarray) -> dict[int, float]:
    logits = torch.from_numpy(logits_np.astype(np.float32, copy=False)).unsqueeze(0)
    input_ids = torch.from_numpy(input_ids_np.astype(np.int64, copy=False)).unsqueeze(0)
    scores = logits
    if PARAMS["temperature"] > 0 and PARAMS["temperature"] != 1.0:
        scores = scores / PARAMS["temperature"]
    top_k = min(PARAMS["top_k"], scores.size(-1))
    top_values, top_indices = torch.topk(scores, top_k)
    sorted_logits, sorted_order = torch.sort(top_values, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - PARAMS["top_p"])
    sorted_indices_to_remove[..., -1:] = 0
    top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    top_indices_to_remove = top_indices_to_remove.scatter(1, sorted_order, sorted_indices_to_remove)
    scores = torch.full_like(scores, -float("inf"))
    scores = scores.scatter(1, top_indices, top_values.masked_fill(top_indices_to_remove, -float("inf")))
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * PARAMS["repetition_penalty"], score / PARAMS["repetition_penalty"])
    scores = scores.scatter(1, input_ids, score)
    probs = F.softmax(scores, dim=-1).squeeze(0)
    finite = torch.nonzero(probs > 0, as_tuple=False).squeeze(1)
    return {int(index): float(probs[index].item()) for index in finite}


def native_probs(lib: ctypes.CDLL, logits_np: np.ndarray, input_ids_np: np.ndarray) -> dict[int, float]:
    logits = np.ascontiguousarray(logits_np.astype(np.float32, copy=False))
    ids = np.ascontiguousarray(input_ids_np.astype(np.int32, copy=False))
    max_out = min(PARAMS["top_k"], logits.shape[0])
    out_tokens = np.empty((max_out,), dtype=np.int32)
    out_probs = np.empty((max_out,), dtype=np.float32)
    out_count = ctypes.c_int32(0)
    rc = lib.cb_t3_native_filter_probs(
        logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(logits.shape[0]),
        ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(ids.shape[0]),
        float(PARAMS["temperature"]),
        int(PARAMS["top_k"]),
        float(PARAMS["top_p"]),
        float(PARAMS["repetition_penalty"]),
        out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(max_out),
        ctypes.byref(out_count),
    )
    if rc != 0:
        raise RuntimeError(f"cb_t3_native_filter_probs failed rc={rc}")
    count = int(out_count.value)
    return {int(out_tokens[i]): float(out_probs[i]) for i in range(count)}


def compare_dicts(py: dict[int, float], native: dict[int, float]) -> dict[str, Any]:
    py_keys = set(py)
    native_keys = set(native)
    union = py_keys | native_keys
    max_abs = 0.0
    l1 = 0.0
    for key in union:
        diff = abs(py.get(key, 0.0) - native.get(key, 0.0))
        max_abs = max(max_abs, diff)
        l1 += diff
    return {
        "support_equal": py_keys == native_keys,
        "python_support": len(py_keys),
        "native_support": len(native_keys),
        "max_abs_prob_diff": max_abs,
        "l1_prob_diff": l1,
        "python_prob_sum": sum(py.values()),
        "native_prob_sum": sum(native.values()),
        "top_python": sorted(py.items(), key=lambda item: item[1], reverse=True)[:10],
        "top_native": sorted(native.items(), key=lambda item: item[1], reverse=True)[:10],
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# T3 Native Sampler Real-Logits Validation - 2026-07-08",
        "",
        f"- Status: `{report['status']}`",
        f"- Overall pass: `{report['overall_pass']}`",
        f"- Fixture: `{FIXTURE}`",
        "",
        "| Step | History Len | Support Equal | Max Abs Diff | L1 Diff | Python Support | Native Support |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("steps", []):
        cmp = item["comparison"]
        lines.append(
            f"| {item['step']} | {item['history_len']} | {cmp['support_equal']} | "
            f"`{cmp['max_abs_prob_diff']:.3g}` | `{cmp['l1_prob_diff']:.3g}` | "
            f"{cmp['python_support']} | {cmp['native_support']} |"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    compile_report = compile_lib()
    report: dict[str, Any] = {
        "description": "Compare isolated native sampler C ABI against Python sampler probabilities on saved real T3 logits.",
        "params": PARAMS,
        "fixture": str(FIXTURE),
        "compile": compile_report,
        "status": "not_started",
        "overall_pass": False,
        "steps": [],
        "decision": "",
    }
    if not compile_report.get("ok"):
        report["status"] = "compile_failed"
        report["decision"] = "Native sampler bridge did not compile; no runtime use."
        OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        return 0

    manifest = json.loads(MANIFEST.read_text())
    steps = manifest["steps_meta"]
    lib = load_lib()
    history: list[int] = []
    for index, meta in enumerate(steps):
        if index == 0:
            history = [int(meta["input_token"])]
        else:
            history.append(int(steps[index - 1]["next_argmax"]))
        logits_path = Path(meta["tensors"]["ref_logits"]["path"])
        logits = np.fromfile(logits_path, dtype=np.float32)
        input_ids = np.asarray(history, dtype=np.int32)
        py = python_probs(logits, input_ids)
        native = native_probs(lib, logits, input_ids)
        comparison = compare_dicts(py, native)
        report["steps"].append(
            {
                "step": index,
                "logits_path": str(logits_path),
                "history": list(history),
                "history_len": len(history),
                "comparison": comparison,
            }
        )

    overall = all(
        item["comparison"]["support_equal"]
        and item["comparison"]["max_abs_prob_diff"] <= 1e-6
        and item["comparison"]["l1_prob_diff"] <= 1e-5
        for item in report["steps"]
    )
    report["overall_pass"] = overall
    report["status"] = "ok" if overall else "mismatch"
    report["decision"] = (
        "Native sampler bridge matches Python probability distributions on saved real T3 logits. Next safe step is an opt-in runtime prototype."
        if overall
        else "Native sampler bridge does not yet match Python on saved real T3 logits; do not wire runtime."
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"status={report['status']}")
    print(f"overall_pass={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
