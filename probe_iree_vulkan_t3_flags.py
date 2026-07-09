#!/usr/bin/env python3
"""Try IREE Vulkan compiler/runtime variants for saved T3 subgraphs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
OUT_DIR = EXPORT_DIR / "flag_variants"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"


@dataclass(frozen=True)
class Variant:
    name: str
    compile_flags: tuple[str, ...] = ()
    run_flags: tuple[str, ...] = ()


VARIANTS: tuple[Variant, ...] = (
    Variant("baseline"),
    Variant("robust_buffers", run_flags=("--vulkan_robust_buffer_access=true",)),
    Variant("generalize_matmul", compile_flags=("--iree-opt-generalize-matmul",)),
    Variant(
        "split_matmul_reduction_2",
        compile_flags=("--iree-dispatch-creation-split-matmul-reduction=2",),
    ),
    Variant(
        "split_matmul_reduction_4",
        compile_flags=("--iree-dispatch-creation-split-matmul-reduction=4",),
    ),
    Variant(
        "split_matmul_reduction_8",
        compile_flags=("--iree-dispatch-creation-split-matmul-reduction=8",),
    ),
    Variant(
        "vectorize_pipeline",
        compile_flags=("--iree-codegen-llvmgpu-vectorize-pipeline",),
    ),
    Variant(
        "no_vector_distribution",
        compile_flags=("--iree-codegen-llvmgpu-use-vector-distribution=false",),
    ),
    Variant(
        "no_tile_and_fuse_matmul",
        compile_flags=("--iree-codegen-llvmgpu-use-tile-and-fuse-matmul=false",),
    ),
    Variant(
        "no_mmt4d_intrinsics",
        compile_flags=("--iree-codegen-mmt4d-use-intrinsics=false",),
    ),
    Variant(
        "enable_split_reduction",
        compile_flags=("--iree-dispatch-creation-enable-split-reduction",),
    ),
    Variant(
        "no_reduction_vector_distribution",
        compile_flags=("--iree-codegen-llvmgpu-use-reduction-vector-distribution=false",),
    ),
    Variant(
        "fuse_multi_reduction",
        compile_flags=("--iree-dispatch-creation-element-wise-fuse-multi-reduction",),
    ),
    Variant(
        "no_aggressive_fusion",
        compile_flags=("--iree-dispatch-creation-enable-aggressive-fusion=false",),
    ),
    Variant(
        "no_fuse_multi_use",
        compile_flags=("--iree-dispatch-creation-fuse-multi-use=false",),
    ),
    Variant(
        "dispatch_opt_0",
        compile_flags=("--iree-dispatch-creation-opt-level=0",),
    ),
    Variant(
        "split_matmul4_split_reduction",
        compile_flags=(
            "--iree-dispatch-creation-split-matmul-reduction=4",
            "--iree-dispatch-creation-enable-split-reduction",
        ),
    ),
    Variant(
        "split_matmul8_split_reduction",
        compile_flags=(
            "--iree-dispatch-creation-split-matmul-reduction=8",
            "--iree-dispatch-creation-enable-split-reduction",
        ),
    ),
)


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
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
            "stdout_tail": completed.stdout.splitlines()[-30:],
            "stderr_tail": completed.stderr.splitlines()[-60:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-30:]
            if isinstance(exc.stdout, str)
            else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-60:]
            if isinstance(exc.stderr, str)
            else [],
        }


def compare(actual_path: Path, expected_path: Path) -> dict[str, Any]:
    actual = np.load(actual_path)
    expected = np.load(expected_path)
    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def probe_variant(
    subgraph: str,
    variant: Variant,
    compile_timeout: int,
    run_timeout: int,
) -> dict[str, Any]:
    variant_dir = OUT_DIR / subgraph / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    mlir = EXPORT_DIR / f"{subgraph}.mlir"
    input_paths = sorted(EXPORT_DIR.glob(f"{subgraph}_input_*.npy"))
    if not input_paths:
        raise FileNotFoundError(f"No input files found for {subgraph}")
    expected_path = EXPORT_DIR / f"{subgraph}_torch_output.npy"
    vmfb = variant_dir / f"{subgraph}_{variant.name}.vmfb"
    output = variant_dir / f"{subgraph}_{variant.name}_vulkan.npy"

    compile_cmd = [
        IREE_COMPILE.as_posix(),
        mlir.as_posix(),
        "--iree-hal-target-backends=vulkan-spirv",
        "--iree-vulkan-target=gfx1013",
        *variant.compile_flags,
        f"-o={vmfb}",
    ]
    compile_result = run_cmd(compile_cmd, timeout=compile_timeout)
    result: dict[str, Any] = {
        "name": variant.name,
        "subgraph": subgraph,
        "compile_flags": list(variant.compile_flags),
        "run_flags": list(variant.run_flags),
        "compile": compile_result,
        "vmfb": vmfb.as_posix(),
    }
    if compile_result["returncode"] != 0:
        result["status"] = "compile_failed"
        return result

    if output.exists():
        output.unlink()
    run_cmdline = [
        IREE_RUN.as_posix(),
        f"--module={vmfb}",
        "--device=vulkan",
        "--function=forward",
        *variant.run_flags,
        *[f"--input=@{input_path}" for input_path in input_paths],
        f"--output=@{output}",
    ]
    run_result = run_cmd(run_cmdline, timeout=run_timeout)
    result["run"] = run_result
    if run_result["returncode"] != 0:
        result["status"] = "run_failed"
        return result
    if not output.exists():
        result["status"] = "missing_output"
        return result

    result["output"] = output.as_posix()
    result["compare"] = compare(output, expected_path)
    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgraph", default="gpt2_mlp_c_proj_t1")
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument(
        "--variants",
        default=",".join(variant.name for variant in VARIANTS),
        help="Comma-separated variant names to run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_vulkan_flag_probe_latest.json",
    )
    args = parser.parse_args()

    selected = {name.strip() for name in args.variants.split(",") if name.strip()}
    variants = [variant for variant in VARIANTS if variant.name in selected]
    if not variants:
        raise SystemExit(f"No matching variants selected: {sorted(selected)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for variant in variants:
        result = probe_variant(
            args.subgraph,
            variant,
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
        )
        results.append(result)
        compare_result = result.get("compare", {})
        max_abs = compare_result.get("max_abs_error")
        suffix = f" max_abs={max_abs:.3e}" if isinstance(max_abs, float) else ""
        print(f"{args.subgraph}/{variant.name}: {result['status']}{suffix}")

    report = {
        "subgraph": args.subgraph,
        "seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
