#!/usr/bin/env python3
"""Compile a fixed-slot K/V cache updater for an arbitrary cache length."""

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

from probe_t3_kv_cache_slot_update_vulkan import (
    KVCacheSlotUpdate,
    compare,
    expected_update,
    slot_mask,
    to_host_array,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability" / "cache_update"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"


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


def make_inputs(max_len: int, slot: int) -> tuple[torch.Tensor, ...]:
    rng = np.random.default_rng(1234)
    past_key = rng.standard_normal((1, 16, max_len, 64), dtype=np.float32)
    past_value = rng.standard_normal((1, 16, max_len, 64), dtype=np.float32)
    new_key = rng.standard_normal((1, 16, 1, 64), dtype=np.float32)
    new_value = rng.standard_normal((1, 16, 1, 64), dtype=np.float32)
    return (
        torch.from_numpy(past_key),
        torch.from_numpy(past_value),
        torch.from_numpy(new_key),
        torch.from_numpy(new_value),
        torch.from_numpy(slot_mask(max_len, slot)),
    )


def export_and_compile(max_len: int, slot: int, compile_timeout: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"t3_kv_cache_slot_update_p{max_len}_t1"
    module = KVCacheSlotUpdate().eval()
    inputs = make_inputs(max_len, slot)
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
        "max_len": max_len,
        "slot": slot,
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
    output_paths = [OUT_DIR / f"{probe['name']}_vulkan_output_{index}.npy" for index in range(2)]
    for path in output_paths:
        if path.exists():
            path.unlink()
    result = run_cmd(
        [
            IREE_RUN.as_posix(),
            f"--module={probe['vmfb']}",
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
    validation["status"] = "ok"
    validation["comparisons"] = [
        compare(np.load(output_paths[index]), np.load(probe["expected_paths"][index]))
        for index in range(2)
    ]
    return validation


def benchmark_runtime(probe: dict[str, Any], iterations: int, warmup: int) -> dict[str, Any]:
    module = ireert.load_vm_flatbuffer_file(probe["vmfb"], driver="vulkan")
    device = ireert.get_device("vulkan")
    host_inputs = [np.load(path).astype(np.float32, copy=False) for path in probe["input_paths"]]
    device_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in host_inputs
    ]

    expected_key, expected_value = expected_update(
        host_inputs[0],
        host_inputs[1],
        host_inputs[2],
        host_inputs[3],
        probe["slot"],
    )
    first_key, first_value = module["forward"](*device_inputs)

    def one_update():
        return module["forward"](*device_inputs)

    def twenty_four_updates():
        outputs = []
        for _ in range(24):
            outputs.append(module["forward"](*device_inputs))
        return outputs

    return {
        "first_compare": {
            "key": compare(to_host_array(first_key), expected_key),
            "value": compare(to_host_array(first_value), expected_value),
        },
        "timings": {
            "one_slot_update": time_call(one_update, iterations, warmup),
            "twenty_four_slot_updates": time_call(twenty_four_updates, iterations, warmup),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--slot", type=int, default=935)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_kv_cache_slot_update_shape_vulkan_latest.json",
    )
    args = parser.parse_args()
    if args.slot >= args.max_len:
        raise SystemExit("--slot must be less than --max-len")

    started = time.perf_counter()
    probe = export_and_compile(args.max_len, args.slot, compile_timeout=args.compile_timeout)
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"compile_status={probe['compile']['returncode']} vmfb_size={probe['vmfb_size_bytes']}")
    if run_validation is not None:
        print(f"run_validation={run_validation['status']}")
    if runtime is not None:
        print(f"first_key_exact={runtime['first_compare']['key']['allclose_exact']}")
        print(f"first_value_exact={runtime['first_compare']['value']['allclose_exact']}")
        for name, timing in runtime["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")


if __name__ == "__main__":
    main()
