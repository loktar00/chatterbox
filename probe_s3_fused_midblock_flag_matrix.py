#!/usr/bin/env python3
"""Compile-flag matrix for an existing fused S3 mid-block MLIR.

This is intentionally isolated: it does not load Chatterbox, does not generate
audio, and does not use ROCm/HIP. It reuses saved MLIR and numpy fixtures from
`probe_s3_fused_midblock_vulkan.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import iree.runtime as ireert
import numpy as np

from benchmark_s3_estimator_distinct_iree_runtime_chain import distinct_vmfb, helper_vmfb, to_host_array


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"

VARIANT_FLAGS = {
    "baseline_split4": (
        "--iree-vulkan-target=gfx1013",
        "--iree-dispatch-creation-split-matmul-reduction=4",
        "--iree-dispatch-creation-enable-split-reduction",
    ),
    "no_split": (
        "--iree-vulkan-target=gfx1013",
    ),
    "split2": (
        "--iree-vulkan-target=gfx1013",
        "--iree-dispatch-creation-split-matmul-reduction=2",
        "--iree-dispatch-creation-enable-split-reduction",
    ),
    "split8": (
        "--iree-vulkan-target=gfx1013",
        "--iree-dispatch-creation-split-matmul-reduction=8",
        "--iree-dispatch-creation-enable-split-reduction",
    ),
}


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
            "stderr_tail": completed.stderr.splitlines()[-80:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-30:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-80:] if isinstance(exc.stderr, str) else [],
        }


def rss_mb() -> float:
    try:
        with Path("/proc/self/status").open() as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return -1.0


def load_npy(path: Path, dtype: np.dtype | None = np.float32) -> np.ndarray:
    array = np.load(path)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def diff_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual.astype(np.float32, copy=False) - expected.astype(np.float32, copy=False))
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def time_call(fn: Callable[[], object], iterations: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "warmup": warmup,
        "total_seconds": elapsed,
        "mean_ms": elapsed * 1000.0 / iterations,
        "items_per_second": iterations / elapsed,
    }


def compile_variant(
    mlir_path: Path,
    vmfb_path: Path,
    flags: tuple[str, ...],
    timeout: int,
    force: bool,
) -> dict[str, Any]:
    if vmfb_path.exists() and not force:
        return {
            "status": "exists",
            "returncode": 0,
            "flags": list(flags),
            "vmfb": vmfb_path.as_posix(),
            "vmfb_size_bytes": vmfb_path.stat().st_size,
        }
    cmd = [
        IREE_COMPILE.as_posix(),
        mlir_path.as_posix(),
        "--iree-hal-target-backends=vulkan-spirv",
        *flags,
        f"-o={vmfb_path}",
    ]
    result = run_cmd(cmd, timeout=timeout)
    result["flags"] = list(flags)
    result["status"] = "ok" if result["returncode"] == 0 and vmfb_path.exists() else "failed"
    result["vmfb"] = vmfb_path.as_posix()
    if vmfb_path.exists():
        result["vmfb_size_bytes"] = vmfb_path.stat().st_size
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1222)
    parser.add_argument("--mid-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument(
        "--variants",
        default="baseline_split4,no_split,split2,split8",
        help="comma-separated variants from: " + ",".join(sorted(VARIANT_FLAGS)),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_fused_midblock0_t1222_compile_flag_matrix_2026-07-08.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    selected_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [name for name in selected_variants if name not in VARIANT_FLAGS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    name = f"s3_fused_mid{args.mid_index}_block_t{args.frames}"
    fixture_dir = BASE / "fused_estimator" / name
    mlir_path = fixture_dir / f"{name}.mlir"
    expected_path = fixture_dir / "torch_output.npy"
    input_paths = [
        fixture_dir / "input_hidden.npy",
        fixture_dir / "input_mask.npy",
        fixture_dir / "input_attention_bias.npy",
        fixture_dir / "input_time_emb.npy",
    ]
    required = [mlir_path, expected_path, *input_paths]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing required fused-midblock fixtures:\n" + "\n".join(missing))

    matrix_dir = fixture_dir / "compile_flag_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "description": "IREE compile-flag matrix for saved fused S3 estimator mid-block MLIR.",
        "frames": args.frames,
        "mid_index": args.mid_index,
        "fixture_dir": fixture_dir.as_posix(),
        "mlir": mlir_path.as_posix(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "rss_start_mb": rss_mb(),
        "variants": [],
    }
    started_all = time.perf_counter()

    expected = load_npy(expected_path)
    inputs = [load_npy(path) for path in input_paths]
    device = ireert.get_device("vulkan")
    device_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in inputs
    ]

    mid_resnet = ireert.load_vm_flatbuffer_file(
        distinct_vmfb(f"s3_distinct_mid_resnet{args.mid_index}_t{args.frames}").as_posix(),
        driver="vulkan",
    )
    transformers = [
        ireert.load_vm_flatbuffer_file(
            distinct_vmfb(f"s3_distinct_mid{args.mid_index}_transformer{index}_t{args.frames}").as_posix(),
            driver="vulkan",
        )
        for index in range(4)
    ]
    c_to_t = ireert.load_vm_flatbuffer_file(
        helper_vmfb(f"s3_flow_transpose_c256_t{args.frames}").as_posix(),
        driver="vulkan",
    )
    t_to_c = ireert.load_vm_flatbuffer_file(
        helper_vmfb(f"s3_flow_transpose_t{args.frames}_c256").as_posix(),
        driver="vulkan",
    )
    hidden, mask, attention_bias, time_emb = device_inputs

    def stitched_no_fetch():
        value = mid_resnet["forward"](hidden, mask, time_emb)
        value = c_to_t["forward"](value)
        for transformer in transformers:
            value = transformer["forward"](value, attention_bias, time_emb)
        return t_to_c["forward"](value)

    stitched_output = to_host_array(stitched_no_fetch())
    result["stitched_validation"] = diff_summary(stitched_output, expected)
    result["stitched_timings"] = {
        "no_fetch": time_call(stitched_no_fetch, args.iterations, args.warmup),
        "fetch_output": time_call(lambda: to_host_array(stitched_no_fetch()), args.iterations, args.warmup),
    }

    for variant in selected_variants:
        flags = VARIANT_FLAGS[variant]
        vmfb_path = matrix_dir / f"{name}_{variant}_vulkan_gfx1013.vmfb"
        row: dict[str, Any] = {
            "name": variant,
            "flags": list(flags),
            "vmfb": vmfb_path.as_posix(),
        }
        row["compile"] = compile_variant(
            mlir_path,
            vmfb_path,
            flags,
            timeout=args.compile_timeout,
            force=args.force_compile,
        )
        if row["compile"].get("status") not in {"ok", "exists"}:
            row["status"] = "compile_failed"
            result["variants"].append(row)
            continue
        module = ireert.load_vm_flatbuffer_file(vmfb_path.as_posix(), driver="vulkan")

        def fused_no_fetch():
            return module["forward"](hidden, mask, attention_bias, time_emb)

        fused_output = to_host_array(fused_no_fetch())
        row["validation"] = diff_summary(fused_output, expected)
        row["fused_vs_stitched"] = diff_summary(fused_output, stitched_output)
        row["timings"] = {
            "no_fetch": time_call(fused_no_fetch, args.iterations, args.warmup),
            "fetch_output": time_call(lambda: to_host_array(fused_no_fetch()), args.iterations, args.warmup),
        }
        fused_no_fetch_ms = row["timings"]["no_fetch"]["mean_ms"]
        fused_fetch_ms = row["timings"]["fetch_output"]["mean_ms"]
        stitched_no_fetch_ms = result["stitched_timings"]["no_fetch"]["mean_ms"]
        stitched_fetch_ms = result["stitched_timings"]["fetch_output"]["mean_ms"]
        row["speedup_vs_stitched_no_fetch"] = stitched_no_fetch_ms / fused_no_fetch_ms if fused_no_fetch_ms else None
        row["speedup_vs_stitched_fetch_output"] = stitched_fetch_ms / fused_fetch_ms if fused_fetch_ms else None
        row["no_fetch_save_ms"] = stitched_no_fetch_ms - fused_no_fetch_ms
        row["fetch_output_save_ms"] = stitched_fetch_ms - fused_fetch_ms
        row["status"] = "ok" if row["validation"]["allclose_1e_4"] else "validation_failed"
        result["variants"].append(row)

    ok_variants = [row for row in result["variants"] if row.get("status") == "ok"]
    best_no_fetch = max(
        ok_variants,
        key=lambda row: row.get("no_fetch_save_ms", float("-inf")),
        default=None,
    )
    best_fetch = max(
        ok_variants,
        key=lambda row: row.get("fetch_output_save_ms", float("-inf")),
        default=None,
    )
    result["best_no_fetch_variant"] = best_no_fetch["name"] if best_no_fetch else None
    result["best_no_fetch_save_ms"] = best_no_fetch.get("no_fetch_save_ms") if best_no_fetch else None
    result["best_fetch_output_variant"] = best_fetch["name"] if best_fetch else None
    result["best_fetch_output_save_ms"] = best_fetch.get("fetch_output_save_ms") if best_fetch else None
    result["status"] = "ok" if ok_variants else "no_valid_variant"
    result["total_seconds"] = time.perf_counter() - started_all
    result["rss_end_mb"] = rss_mb()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}")
    print(f"status={result['status']}")
    print(f"stitched_no_fetch_ms={result['stitched_timings']['no_fetch']['mean_ms']:.3f}")
    print(f"best_no_fetch_variant={result['best_no_fetch_variant']} save_ms={result['best_no_fetch_save_ms']}")
    print(f"best_fetch_output_variant={result['best_fetch_output_variant']} save_ms={result['best_fetch_output_save_ms']}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
