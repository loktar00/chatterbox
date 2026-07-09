#!/usr/bin/env python3
"""Validate and benchmark a cached T3 KV chunk through IREE's Python runtime."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import iree.runtime as ireert
import numpy as np


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
SUBGRAPH = "gpt2_cached_stack_kv_s0_l4_p128_t1"
VMFB = (
    EXPORT_DIR
    / "flag_variants"
    / SUBGRAPH
    / "split_matmul4_split_reduction"
    / f"{SUBGRAPH}_split_matmul4_split_reduction.vmfb"
)


def load_arrays(prefix: str, count: int) -> list[np.ndarray]:
    return [
        np.load(EXPORT_DIR / f"{SUBGRAPH}_{prefix}_{index}.npy").astype(np.float32, copy=False)
        for index in range(count)
    ]


def to_host_array(value) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def compare_outputs(actual_outputs, expected_outputs: list[np.ndarray]) -> list[dict]:
    comparisons = []
    for index, (actual, expected) in enumerate(zip(actual_outputs, expected_outputs)):
        actual_np = to_host_array(actual)
        diff = np.abs(actual_np - expected)
        comparisons.append(
            {
                "index": index,
                "shape": list(actual_np.shape),
                "max_abs_error": float(diff.max()),
                "mean_abs_error": float(diff.mean()),
                "p95_abs_error": float(np.percentile(diff, 95)),
                "allclose_1e_4": bool(np.allclose(actual_np, expected, atol=1e-4, rtol=1e-4)),
                "allclose_1e_3": bool(np.allclose(actual_np, expected, atol=1e-3, rtol=1e-3)),
            }
        )
    return comparisons


def time_call(fn: Callable[[], object], iterations: int, warmup: int) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=EXPORT_DIR / "t3_kv_iree_runtime_latest.json")
    args = parser.parse_args()

    if not VMFB.exists():
        raise SystemExit(f"Missing VMFB: {VMFB}")

    inputs = load_arrays("input", 9)
    expected_outputs = load_arrays("torch_output", 9)

    load_started = time.perf_counter()
    module = ireert.load_vm_flatbuffer_file(VMFB.as_posix(), driver="vulkan")
    load_seconds = time.perf_counter() - load_started

    first_outputs = module["forward"](*inputs)
    comparisons = compare_outputs(first_outputs, expected_outputs)

    device = ireert.get_device("vulkan")
    device_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in inputs
    ]
    first_device_outputs = module["forward"](*device_inputs)
    device_input_comparisons = compare_outputs(first_device_outputs, expected_outputs)

    def host_inputs_no_fetch():
        return module["forward"](*inputs)

    def host_inputs_fetch():
        outputs = module["forward"](*inputs)
        return [to_host_array(output) for output in outputs]

    def device_inputs_no_fetch():
        return module["forward"](*device_inputs)

    def device_inputs_fetch():
        outputs = module["forward"](*device_inputs)
        return [to_host_array(output) for output in outputs]

    timings = {
        "host_inputs_no_output_fetch": time_call(host_inputs_no_fetch, args.iterations, args.warmup),
        "host_inputs_fetch_all_outputs": time_call(host_inputs_fetch, args.iterations, args.warmup),
        "device_inputs_no_output_fetch": time_call(device_inputs_no_fetch, args.iterations, args.warmup),
        "device_inputs_fetch_all_outputs": time_call(device_inputs_fetch, args.iterations, args.warmup),
    }

    report = {
        "subgraph": SUBGRAPH,
        "vmfb": VMFB.as_posix(),
        "vmfb_size_bytes": VMFB.stat().st_size,
        "load_seconds": load_seconds,
        "input_shapes": [list(array.shape) for array in inputs],
        "output_shapes": [list(array.shape) for array in expected_outputs],
        "validation": {
            "host_inputs": comparisons,
            "device_inputs": device_input_comparisons,
            "host_inputs_allclose_1e_4": all(item["allclose_1e_4"] for item in comparisons),
            "device_inputs_allclose_1e_4": all(
                item["allclose_1e_4"] for item in device_input_comparisons
            ),
        },
        "timings": timings,
        "notes": [
            "host_inputs modes pass NumPy arrays each call and include Python/runtime host input handling.",
            "device_inputs modes reuse preloaded Vulkan DeviceArray inputs and avoid host input uploads.",
            "fetch_all_outputs materializes hidden plus all eight K/V outputs back to host each iteration.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"load_seconds={load_seconds:.3f}")
    print(f"host_inputs_allclose_1e_4={report['validation']['host_inputs_allclose_1e_4']}")
    print(f"device_inputs_allclose_1e_4={report['validation']['device_inputs_allclose_1e_4']}")
    for name, timing in timings.items():
        print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")


if __name__ == "__main__":
    main()
