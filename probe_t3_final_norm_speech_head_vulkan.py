#!/usr/bin/env python3
"""Compile GPT2 final layer norm plus T3 speech head for IREE Vulkan."""

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
import torch
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS
from probe_t3_masked_cache_block_vulkan import compare_arrays, to_host_array


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
DEFAULT_COMPILE_FLAGS = (
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class FinalNormSpeechHead(nn.Module):
    def __init__(self, ln_f: nn.Module, speech_head: nn.Module) -> None:
        super().__init__()
        self.ln_f = ln_f
        self.speech_head = speech_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.speech_head(self.ln_f(hidden_states))


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


def export_compile_validate(iterations: int, warmup: int, compile_timeout: int, run_timeout: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    module = FinalNormSpeechHead(t3.tfmr.ln_f, t3.speech_head).eval()
    hidden = torch.randn(1, 1, t3.dim, dtype=torch.float32)
    with torch.inference_mode():
        expected = module(hidden)

    name = "t3_final_norm_speech_head_t1"
    input_path = OUT_DIR / f"{name}_input_0.npy"
    expected_path = OUT_DIR / f"{name}_torch_output.npy"
    np.save(input_path, hidden.detach().cpu().numpy())
    np.save(expected_path, expected.detach().cpu().numpy())

    mlir = OUT_DIR / f"{name}.mlir"
    vmfb = OUT_DIR / f"{name}_vulkan_gfx1013.vmfb"
    graph = OUT_DIR / f"{name}.torch_export.txt"
    exported = aot.export(module, args=(hidden,), module_name=name, function_name="forward")
    exported.save_mlir(mlir)
    graph.write_text(str(torch.export.export(module, (hidden,)).graph_module) + "\n")

    compile_result = run_cmd(
        [
            IREE_COMPILE.as_posix(),
            mlir.as_posix(),
            "--iree-hal-target-backends=vulkan-spirv",
            "--iree-vulkan-target=gfx1013",
            *DEFAULT_COMPILE_FLAGS,
            f"-o={vmfb}",
        ],
        timeout=compile_timeout,
    )

    output_path = OUT_DIR / f"{name}_vulkan_output.npy"
    run_validation = None
    runtime = None
    if compile_result["returncode"] == 0:
        if output_path.exists():
            output_path.unlink()
        run_result = run_cmd(
            [
                IREE_RUN.as_posix(),
                f"--module={vmfb}",
                "--device=vulkan",
                "--function=forward",
                f"--input=@{input_path}",
                f"--output=@{output_path}",
            ],
            timeout=run_timeout,
        )
        run_validation = {"run": run_result, "output_path": output_path.as_posix()}
        if run_result["returncode"] == 0 and output_path.exists():
            run_validation["status"] = "ok"
            run_validation["comparison"] = compare_arrays(
                np.load(output_path),
                np.load(expected_path),
            )
            runtime_module = ireert.load_vm_flatbuffer_file(vmfb.as_posix(), driver="vulkan")
            device = ireert.get_device("vulkan")
            device_hidden = ireert.asdevicearray(
                device,
                np.load(input_path).astype(np.float32, copy=False),
                implicit_host_transfer=False,
            )

            first = runtime_module["forward"](device_hidden)

            def no_fetch():
                return runtime_module["forward"](device_hidden)

            def fetch():
                return to_host_array(runtime_module["forward"](device_hidden))

            runtime = {
                "first_compare": compare_arrays(to_host_array(first), np.load(expected_path)),
                "timings": {
                    "final_norm_speech_head_no_fetch": time_call(no_fetch, iterations, warmup),
                    "final_norm_speech_head_fetch": time_call(fetch, iterations, warmup),
                },
            }
        else:
            run_validation["status"] = "run_failed"

    return {
        "name": name,
        "input": input_path.as_posix(),
        "expected": expected_path.as_posix(),
        "mlir": mlir.as_posix(),
        "vmfb": vmfb.as_posix(),
        "graph": graph.as_posix(),
        "mlir_size_bytes": mlir.stat().st_size if mlir.exists() else 0,
        "vmfb_size_bytes": vmfb.stat().st_size if vmfb.exists() else 0,
        "compile": compile_result,
        "run_validation": run_validation,
        "runtime": runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_final_norm_speech_head_vulkan_2026-07-08.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    report = export_compile_validate(
        iterations=args.iterations,
        warmup=args.warmup,
        compile_timeout=args.compile_timeout,
        run_timeout=args.run_timeout,
    )
    report["seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"compile_status={report['compile']['returncode']} vmfb_size={report['vmfb_size_bytes']}")
    if report["run_validation"] is not None:
        print(f"run_validation={report['run_validation']['status']}")
        comparison = report["run_validation"].get("comparison")
        if comparison:
            print(f"max_abs_error={comparison['max_abs_error']:.3e} allclose_1e_4={comparison['allclose_1e_4']}")
    if report["runtime"] is not None:
        for name, timing in report["runtime"]["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")


if __name__ == "__main__":
    main()
