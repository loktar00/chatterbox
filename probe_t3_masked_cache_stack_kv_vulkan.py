#!/usr/bin/env python3
"""Probe a masked preallocated-cache 4-layer T3 stack on IREE Vulkan."""

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
from probe_t3_masked_cache_block_vulkan import (
    CachedManualBlockKVWrapper,
    MaskedPreallocatedBlockKVWrapper,
    compare_arrays,
    make_attention_bias,
    to_host_array,
)


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
OUT_DIR = EXPORT_DIR / "masked_cache"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
DEFAULT_COMPILE_FLAGS = (
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class DynamicCacheStackKVWrapper(nn.Module):
    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CachedManualBlockKVWrapper(block) for block in blocks])

    def forward(self, hidden_states: torch.Tensor, *past_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        new_cache: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden_states, new_key, new_value = block(
                hidden_states,
                past_tensors[index * 2],
                past_tensors[index * 2 + 1],
            )
            new_cache.extend((new_key, new_value))
        return (hidden_states, *new_cache)


class MaskedPreallocatedStackKVWrapper(nn.Module):
    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([MaskedPreallocatedBlockKVWrapper(block) for block in blocks])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_bias: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        new_cache: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden_states, new_key, new_value = block(
                hidden_states,
                cache_tensors[index * 2],
                cache_tensors[index * 2 + 1],
                attention_bias,
            )
            new_cache.extend((new_key, new_value))
        return (hidden_states, *new_cache)


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


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def export_probe(
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    expected_outputs: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = []
    for index, tensor in enumerate(inputs):
        path = OUT_DIR / f"{name}_input_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        input_paths.append(path)

    expected_paths = []
    output_shapes = []
    for index, tensor in enumerate(expected_outputs):
        path = OUT_DIR / f"{name}_torch_output_{index}.npy"
        array = tensor.detach().cpu().numpy()
        np.save(path, array)
        expected_paths.append(path)
        output_shapes.append(list(array.shape))

    mlir = OUT_DIR / f"{name}.mlir"
    vmfb = OUT_DIR / f"{name}_vulkan_gfx1013.vmfb"
    graph = OUT_DIR / f"{name}.torch_export.txt"
    exported = aot.export(module, args=inputs, module_name=name, function_name="forward")
    exported.save_mlir(mlir)
    graph.write_text(str(torch.export.export(module, inputs).graph_module) + "\n")
    return {
        "name": name,
        "inputs": [path.as_posix() for path in input_paths],
        "expected": [path.as_posix() for path in expected_paths],
        "mlir": mlir.as_posix(),
        "vmfb": vmfb.as_posix(),
        "graph": graph.as_posix(),
        "output_shapes": output_shapes,
        "mlir_size_bytes": mlir.stat().st_size if mlir.exists() else 0,
    }


def compile_probe(probe: dict[str, Any], compile_timeout: int) -> dict[str, Any]:
    vmfb = Path(probe["vmfb"])
    result = run_cmd(
        [
            IREE_COMPILE.as_posix(),
            probe["mlir"],
            "--iree-hal-target-backends=vulkan-spirv",
            "--iree-vulkan-target=gfx1013",
            *DEFAULT_COMPILE_FLAGS,
            f"-o={vmfb}",
        ],
        timeout=compile_timeout,
    )
    probe["compile"] = result
    probe["vmfb_size_bytes"] = vmfb.stat().st_size if vmfb.exists() else 0
    return probe


def run_module_validation(probe: dict[str, Any], run_timeout: int) -> dict[str, Any]:
    output_paths = [OUT_DIR / f"{probe['name']}_vulkan_output_{index}.npy" for index in range(len(probe["expected"]))]
    for path in output_paths:
        if path.exists():
            path.unlink()
    result = run_cmd(
        [
            IREE_RUN.as_posix(),
            f"--module={probe['vmfb']}",
            "--device=vulkan",
            "--function=forward",
            *[f"--input=@{path}" for path in probe["inputs"]],
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
        compare_arrays(np.load(output_paths[index]), np.load(probe["expected"][index]))
        for index in range(len(output_paths))
    ]
    return validation


def benchmark_runtime(probe: dict[str, Any], iterations: int, warmup: int) -> dict[str, Any]:
    module = ireert.load_vm_flatbuffer_file(probe["vmfb"], driver="vulkan")
    device = ireert.get_device("vulkan")
    host_inputs = [np.load(path).astype(np.float32, copy=False) for path in probe["inputs"]]
    device_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in host_inputs
    ]
    expected = [np.load(path).astype(np.float32, copy=False) for path in probe["expected"]]
    first_outputs = module["forward"](*device_inputs)
    first_compare = [
        compare_arrays(to_host_array(actual), expected_output)
        for actual, expected_output in zip(first_outputs, expected)
    ]

    def no_fetch():
        return module["forward"](*device_inputs)

    def fetch_outputs():
        return [to_host_array(output) for output in module["forward"](*device_inputs)]

    rss_before = rss_mb()
    timings = {
        "masked_stack_no_fetch": time_call(no_fetch, iterations, warmup),
        "masked_stack_fetch_outputs": time_call(fetch_outputs, iterations, warmup),
    }
    rss_after = rss_mb()
    return {
        "first_call_compare": first_compare,
        "timings": timings,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--valid-len", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_masked_cache_stack_kv_vulkan_latest.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(0)

    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    end_layer = args.start_layer + args.layers
    blocks = [t3.tfmr.h[index].eval() for index in range(args.start_layer, end_layer)]
    dynamic = DynamicCacheStackKVWrapper(blocks).eval()
    masked = MaskedPreallocatedStackKVWrapper(blocks).eval()
    first = masked.blocks[0]

    hidden = torch.randn(1, 1, t3.dim, dtype=torch.float32)
    attention_bias = make_attention_bias(args.max_len, args.valid_len)
    cache_tensors: list[torch.Tensor] = []
    dynamic_tensors: list[torch.Tensor] = []
    for _ in range(args.layers):
        key = torch.randn(1, first.num_heads, args.max_len, first.head_dim, dtype=torch.float32)
        value = torch.randn(1, first.num_heads, args.max_len, first.head_dim, dtype=torch.float32)
        cache_tensors.extend((key, value))
        dynamic_tensors.extend(
            (
                key[:, :, : args.valid_len, :].contiguous(),
                value[:, :, : args.valid_len, :].contiguous(),
            )
        )

    with torch.inference_mode():
        dynamic_outputs = dynamic(hidden, *dynamic_tensors)
        masked_outputs = masked(hidden, attention_bias, *cache_tensors)

    static_vs_dynamic = [
        compare_arrays(
            masked_outputs[index].detach().cpu().numpy(),
            dynamic_outputs[index].detach().cpu().numpy(),
        )
        for index in range(len(masked_outputs))
    ]

    name = f"gpt2_masked_cache_stack_s{args.start_layer}_l{args.layers}_p{args.max_len}_valid{args.valid_len}_t1"
    probe = export_probe(name, masked, (hidden, attention_bias, *cache_tensors), masked_outputs)
    probe = compile_probe(probe, compile_timeout=args.compile_timeout)

    run_validation = None
    runtime = None
    if probe["compile"]["returncode"] == 0:
        run_validation = run_module_validation(probe, run_timeout=args.run_timeout)
        if run_validation["status"] == "ok":
            runtime = benchmark_runtime(probe, iterations=args.iterations, warmup=args.warmup)

    report = {
        "seconds": time.perf_counter() - started,
        "start_layer": args.start_layer,
        "end_layer": end_layer,
        "layers": args.layers,
        "max_len": args.max_len,
        "valid_len": args.valid_len,
        "static_vs_dynamic": static_vs_dynamic,
        "static_vs_dynamic_allclose_1e_4": all(item["allclose_1e_4"] for item in static_vs_dynamic),
        "probe": probe,
        "run_validation": run_validation,
        "runtime": runtime,
        "notes": [
            "This scales the masked preallocated-cache attention pattern from one block to a safe 4-layer T3 chunk.",
            "It validates the static masked-cache chunk against a dynamic-cache PyTorch reference before Vulkan.",
            "This is still a subgraph probe, not a complete T3 generation loop.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"static_vs_dynamic_allclose_1e_4={report['static_vs_dynamic_allclose_1e_4']}")
    for index, comparison in enumerate(static_vs_dynamic):
        print(f"static_vs_dynamic output{index}: max_abs={comparison['max_abs_error']:.3e}")
    print(f"compile_status={probe['compile']['returncode']} vmfb_size={probe['vmfb_size_bytes']}")
    if run_validation is not None:
        print(f"run_validation={run_validation['status']}")
        for index, comparison in enumerate(run_validation.get("comparisons", [])):
            print(
                f"vulkan output{index}: max_abs={comparison['max_abs_error']:.3e} "
                f"allclose_1e_4={comparison['allclose_1e_4']}"
            )
    if runtime is not None:
        for name, timing in runtime["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
        print(f"rss_delta_mb={runtime['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
