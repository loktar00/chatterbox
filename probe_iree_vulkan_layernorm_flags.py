#!/usr/bin/env python3
"""Try IREE Vulkan compiler flag variants for the S3 LayerNorm repro."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
CASE_DIR = ROOT / "exports" / "s3_flow_vulkan_components" / "s3_flow_mid_norm1_t16"
OUT_JSON = ROOT / "exports" / "s3_flow_vulkan_components" / "s3_flow_layernorm_flag_matrix_2026-07-08.json"
OUT_MD = ROOT / "exports" / "s3_flow_vulkan_components" / "s3_flow_layernorm_flag_matrix_2026-07-08.md"


CONFIGS: list[tuple[str, list[str]]] = [
    ("gfx1013_base", ["--iree-vulkan-target=gfx1013"]),
    (
        "gfx1013_split_reduction",
        [
            "--iree-vulkan-target=gfx1013",
            "--iree-dispatch-creation-split-matmul-reduction=4",
            "--iree-dispatch-creation-enable-split-reduction",
        ],
    ),
    ("gfx1013_index32", ["--iree-vulkan-target=gfx1013", "--iree-spirv-index-bits=32"]),
    ("gfx1013_index64", ["--iree-vulkan-target=gfx1013", "--iree-spirv-index-bits=64"]),
    ("gfx1013_data_tiling", ["--iree-vulkan-target=gfx1013", "--iree-opt-data-tiling"]),
    (
        "gfx1013_dispatch_data_tiling",
        ["--iree-vulkan-target=gfx1013", "--iree-dispatch-creation-data-tiling"],
    ),
    ("gfx1013_global_data_tiling", ["--iree-vulkan-target=gfx1013", "--iree-global-opt-data-tiling"]),
    (
        "gfx1013_no_aggressive_reshape",
        ["--iree-vulkan-target=gfx1013", "--iree-dispatch-creation-enable-aggressive-reshape-movement=false"],
    ),
    (
        "gfx1013_multi_reduction_fuse",
        ["--iree-vulkan-target=gfx1013", "--iree-dispatch-creation-element-wise-fuse-multi-reduction"],
    ),
    ("rdna1_base", ["--iree-vulkan-target=rdna1"]),
    ("rdna2_base", ["--iree-vulkan-target=rdna2"]),
    ("gfx1030_base", ["--iree-vulkan-target=gfx1030"]),
]


def run_cmd(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_tail": completed.stdout.splitlines()[-20:],
            "stderr_tail": completed.stderr.splitlines()[-50:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "stdout_tail": (exc.stdout or "").splitlines()[-20:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-50:] if isinstance(exc.stderr, str) else [],
        }


def compare(path: Path) -> dict[str, Any]:
    actual = np.load(path)
    expected = np.load(CASE_DIR / "torch_output.npy")
    diff = np.abs(actual - expected)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def run_case(name: str, flags: list[str]) -> dict[str, Any]:
    vmfb = CASE_DIR / f"s3_flow_mid_norm1_t16_{name}.vmfb"
    out = CASE_DIR / f"iree_vulkan_{name}_output.npy"
    compile_cmd = [
        IREE_COMPILE.as_posix(),
        (CASE_DIR / "s3_flow_mid_norm1_t16.mlir").as_posix(),
        "--iree-hal-target-backends=vulkan-spirv",
        *flags,
        f"-o={vmfb}",
    ]
    compile_result = run_cmd(compile_cmd)
    result: dict[str, Any] = {
        "name": name,
        "flags": flags,
        "compile": compile_result,
        "vmfb": vmfb.as_posix(),
    }
    if compile_result["returncode"] != 0:
        result["status"] = "compile_failed"
        return result

    if out.exists():
        out.unlink()
    run_result = run_cmd(
        [
            IREE_RUN.as_posix(),
            f"--module={vmfb}",
            "--device=vulkan",
            "--function=forward",
            f"--input=@{CASE_DIR / 'input_0.npy'}",
            f"--output=@{out}",
        ]
    )
    result["run"] = run_result
    result["output"] = out.as_posix()
    if run_result["returncode"] != 0:
        result["status"] = "run_failed"
        return result
    result["compare"] = compare(out)
    result["status"] = "ok"
    return result


def main() -> None:
    results = []
    for name, flags in CONFIGS:
        result = run_case(name, flags)
        results.append(result)
        compare_result = result.get("compare", {})
        if compare_result:
            print(
                f"{name}: max={compare_result['max_abs_error']:.6g} "
                f"mean={compare_result['mean_abs_error']:.6g} "
                f"ok={compare_result['allclose_1e_4']}"
            )
        else:
            print(f"{name}: {result['status']}")

    report = {
        "description": "IREE Vulkan flag matrix for S3 LayerNorm repro.",
        "case_dir": CASE_DIR.as_posix(),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# S3 LayerNorm IREE Vulkan Flag Matrix",
        "",
        f"Case: `{CASE_DIR / 's3_flow_mid_norm1_t16.mlir'}`",
        "",
        "| Config | Status | Max abs | Mean abs | allclose 1e-4 |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        compare_result = result.get("compare")
        if compare_result:
            lines.append(
                f"| `{result['name']}` | {result['status']} | "
                f"{compare_result['max_abs_error']:.6g} | "
                f"{compare_result['mean_abs_error']:.6g} | "
                f"{compare_result['allclose_1e_4']} |"
            )
        else:
            lines.append(f"| `{result['name']}` | {result['status']} |  |  |  |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote={OUT_JSON}")
    print(f"wrote={OUT_MD}")


if __name__ == "__main__":
    main()
