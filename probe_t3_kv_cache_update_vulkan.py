#!/usr/bin/env python3
"""Compile and benchmark a tiny Vulkan K/V cache roll-append helper for T3."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import iree.runtime as ireert
import numpy as np
import torch
from iree.turbine import aot
from torch import nn


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
OUT_DIR = EXPORT_DIR / "cache_update"
T3_VARIANT = "split_matmul4_split_reduction"
T3_SUBGRAPH = "gpt2_cached_stack_kv_s0_l4_p128_t1"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"


class KVCacheRollAppend(nn.Module):
    def forward(
        self,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_key = torch.cat((past_key[:, :, 1:, :], new_key), dim=2)
        next_value = torch.cat((past_value[:, :, 1:, :], new_value), dim=2)
        return next_key.contiguous(), next_value.contiguous()


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
            "stdout_tail": (exc.stdout or "").splitlines()[-30:]
            if isinstance(exc.stdout, str)
            else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-80:]
            if isinstance(exc.stderr, str)
            else [],
        }


def load_t3_input(index: int) -> np.ndarray:
    return np.load(EXPORT_DIR / f"{T3_SUBGRAPH}_input_{index}.npy").astype(np.float32, copy=False)


def load_t3_output(index: int) -> np.ndarray:
    return np.load(EXPORT_DIR / f"{T3_SUBGRAPH}_torch_output_{index}.npy").astype(np.float32, copy=False)


def to_host_array(value) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_exact": bool(np.array_equal(actual, expected)),
        "allclose_1e_6": bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-6)),
        "allclose_1e_5": bool(np.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
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


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def export_and_compile(name: str, compile_timeout: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module = KVCacheRollAppend().eval()
    inputs = (
        torch.from_numpy(load_t3_input(1)),
        torch.from_numpy(load_t3_input(2)),
        torch.from_numpy(load_t3_output(1)),
        torch.from_numpy(load_t3_output(2)),
    )
    with torch.inference_mode():
        expected = module(*inputs)

    input_paths = []
    for index, tensor in enumerate(inputs):
        path = OUT_DIR / f"{name}_input_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        input_paths.append(path)

    expected_paths = []
    for index, tensor in enumerate(expected):
        path = OUT_DIR / f"{name}_torch_output_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        expected_paths.append(path)

    mlir = OUT_DIR / f"{name}.mlir"
    vmfb = OUT_DIR / f"{name}_vulkan_gfx1013.vmfb"
    graph = OUT_DIR / f"{name}.torch_export.txt"
    exported = aot.export(module, args=inputs, module_name=name, function_name="forward")
    exported.save_mlir(mlir)
    graph.write_text(str(torch.export.export(module, inputs).graph_module) + "\n")

    compile_result = run_cmd(
        [
            IREE_COMPILE.as_posix(),
            mlir.as_posix(),
            "--iree-hal-target-backends=vulkan-spirv",
            "--iree-vulkan-target=gfx1013",
            f"-o={vmfb}",
        ],
        timeout=compile_timeout,
    )
    return {
        "name": name,
        "input_paths": [path.as_posix() for path in input_paths],
        "expected_paths": [path.as_posix() for path in expected_paths],
        "mlir": mlir.as_posix(),
        "vmfb": vmfb.as_posix(),
        "graph": graph.as_posix(),
        "mlir_size_bytes": mlir.stat().st_size if mlir.exists() else 0,
        "vmfb_size_bytes": vmfb.stat().st_size if vmfb.exists() else 0,
        "compile": compile_result,
    }


def run_module_validation(probe: dict[str, Any], run_timeout: int) -> dict[str, Any]:
    vmfb = Path(probe["vmfb"])
    output_paths = [OUT_DIR / f"{probe['name']}_vulkan_output_{index}.npy" for index in range(2)]
    for path in output_paths:
        if path.exists():
            path.unlink()

    result = run_cmd(
        [
            IREE_RUN.as_posix(),
            f"--module={vmfb}",
            "--device=vulkan",
            "--function=forward",
            *[f"--input=@{path}" for path in probe["input_paths"]],
            *[f"--output=@{path}" for path in output_paths],
        ],
        timeout=run_timeout,
    )
    validation: dict[str, Any] = {
        "run": result,
        "output_paths": [path.as_posix() for path in output_paths],
    }
    if result["returncode"] != 0:
        validation["status"] = "run_failed"
        return validation
    if any(not path.exists() for path in output_paths):
        validation["status"] = "missing_output"
        return validation

    validation["status"] = "ok"
    validation["comparisons"] = [
        compare(np.load(output_paths[index]), np.load(probe["expected_paths"][index]))
        for index in range(2)
    ]
    return validation


def t3_vmfb() -> Path:
    return (
        EXPORT_DIR
        / "flag_variants"
        / T3_SUBGRAPH
        / T3_VARIANT
        / f"{T3_SUBGRAPH}_{T3_VARIANT}.vmfb"
    )


def benchmark_runtime(probe: dict[str, Any], iterations: int, warmup: int) -> dict[str, Any]:
    update_module = ireert.load_vm_flatbuffer_file(probe["vmfb"], driver="vulkan")
    t3_module = ireert.load_vm_flatbuffer_file(t3_vmfb().as_posix(), driver="vulkan")
    device = ireert.get_device("vulkan")

    t3_host_inputs = [load_t3_input(index) for index in range(9)]
    t3_device_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in t3_host_inputs
    ]

    t3_outputs = t3_module["forward"](*t3_device_inputs)
    layer_updates = []
    update_comparisons = []
    for layer in range(4):
        past_key = t3_device_inputs[1 + layer * 2]
        past_value = t3_device_inputs[2 + layer * 2]
        new_key = t3_outputs[1 + layer * 2]
        new_value = t3_outputs[2 + layer * 2]
        updated_key, updated_value = update_module["forward"](past_key, past_value, new_key, new_value)
        layer_updates.append((updated_key, updated_value))

        expected_key = np.concatenate(
            (t3_host_inputs[1 + layer * 2][:, :, 1:, :], load_t3_output(1 + layer * 2)),
            axis=2,
        )
        expected_value = np.concatenate(
            (t3_host_inputs[2 + layer * 2][:, :, 1:, :], load_t3_output(2 + layer * 2)),
            axis=2,
        )
        update_comparisons.append(
            {
                "layer": layer,
                "key": compare(to_host_array(updated_key), expected_key),
                "value": compare(to_host_array(updated_value), expected_value),
            }
        )

    def four_updates_from_cached_t3_outputs():
        updates = []
        for layer in range(4):
            updates.append(
                update_module["forward"](
                    t3_device_inputs[1 + layer * 2],
                    t3_device_inputs[2 + layer * 2],
                    t3_outputs[1 + layer * 2],
                    t3_outputs[2 + layer * 2],
                )
            )
        return updates

    def t3_chunk_no_fetch():
        return t3_module["forward"](*t3_device_inputs)

    def t3_plus_four_updates_no_fetch():
        outputs = t3_module["forward"](*t3_device_inputs)
        updates = []
        for layer in range(4):
            updates.append(
                update_module["forward"](
                    t3_device_inputs[1 + layer * 2],
                    t3_device_inputs[2 + layer * 2],
                    outputs[1 + layer * 2],
                    outputs[2 + layer * 2],
                )
            )
        return outputs, updates

    def t3_plus_four_updates_fetch_updated_cache():
        _, updates = t3_plus_four_updates_no_fetch()
        return [(to_host_array(key), to_host_array(value)) for key, value in updates]

    rss_before = rss_mb()
    timings = {
        "t3_chunk_no_fetch": time_call(
            t3_chunk_no_fetch,
            iterations,
            warmup,
        ),
        "four_cache_updates_from_cached_t3_outputs": time_call(
            four_updates_from_cached_t3_outputs,
            iterations,
            warmup,
        ),
        "t3_chunk_plus_four_cache_updates_no_fetch": time_call(
            t3_plus_four_updates_no_fetch,
            iterations,
            warmup,
        ),
        "t3_chunk_plus_four_cache_updates_fetch_updated_cache": time_call(
            t3_plus_four_updates_fetch_updated_cache,
            iterations,
            warmup,
        ),
    }
    rss_after = rss_mb()

    return {
        "update_comparisons": update_comparisons,
        "all_updates_allclose_1e_6": all(
            item["key"]["allclose_1e_6"] and item["value"]["allclose_1e_6"]
            for item in update_comparisons
        ),
        "all_updates_allclose_1e_5": all(
            item["key"]["allclose_1e_5"] and item["value"]["allclose_1e_5"]
            for item in update_comparisons
        ),
        "all_updates_allclose_1e_4": all(
            item["key"]["allclose_1e_4"] and item["value"]["allclose_1e_4"]
            for item in update_comparisons
        ),
        "timings": timings,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_kv_cache_update_vulkan_latest.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    name = "t3_kv_cache_roll_append_p128_t1"
    probe = export_and_compile(name, compile_timeout=args.compile_timeout)
    run_validation = None
    runtime = None
    if probe["compile"]["returncode"] == 0:
        run_validation = run_module_validation(probe, run_timeout=args.run_timeout)
        if run_validation["status"] == "ok":
            runtime = benchmark_runtime(probe, iterations=args.iterations, warmup=args.warmup)

    report = {
        "seconds": time.perf_counter() - started,
        "probe": probe,
        "run_validation": run_validation,
        "runtime": runtime,
        "notes": [
            "This is a fixed-size rolling-window cache update: it drops the oldest slot and appends the new token.",
            "It validates device-resident cache movement for bucketed experiments, not exact growing-cache T3 generation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"compile_status={probe['compile']['returncode']} vmfb_size={probe['vmfb_size_bytes']}")
    if run_validation is not None:
        print(f"run_validation={run_validation['status']}")
    if runtime is not None:
        print(f"all_updates_allclose_1e_6={runtime['all_updates_allclose_1e_6']}")
        print(f"all_updates_allclose_1e_5={runtime['all_updates_allclose_1e_5']}")
        print(f"all_updates_allclose_1e_4={runtime['all_updates_allclose_1e_4']}")
        for name, timing in runtime["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
        print(f"rss_delta_mb={runtime['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
