#!/usr/bin/env python3
"""Probe T3 block attention over a preallocated masked K/V cache on Vulkan."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import iree.runtime as ireert
import numpy as np
import torch
import torch.nn.functional as F
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
OUT_DIR = EXPORT_DIR / "masked_cache"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
DEFAULT_COMPILE_FLAGS = (
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class CachedManualBlockKVWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.split_size = block.attn.split_size
        self.scale_attn_weights = block.attn.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = block.attn.scale_attn_by_inverse_layer_idx
        self.layer_idx = block.attn.layer_idx or 0

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self.block.attn.c_attn(hidden_states).split(
            self.split_size,
            dim=2,
        )
        query_shape = (*query.shape[:-1], -1, self.head_dim)
        query = query.view(query_shape).transpose(1, 2)
        key_shape = (*key.shape[:-1], -1, self.head_dim)
        key = key.view(key_shape).transpose(1, 2)
        value = value.view(key_shape).transpose(1, 2)
        return query, key, value

    def _finish(
        self,
        residual: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(float(self.head_dim))
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        if attention_bias is not None:
            attn_weights = attn_weights + attention_bias
        attn_weights = F.softmax(attn_weights, dim=-1).type(value.dtype)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.block.attn.c_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.block.ln_2(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)
        query, new_key, new_value = self._project_qkv(hidden_states)
        key = torch.cat((past_key, new_key), dim=2)
        value = torch.cat((past_value, new_value), dim=2)
        hidden_states = self._finish(residual, query, key, value)
        return hidden_states, new_key.contiguous(), new_value.contiguous()


class MaskedPreallocatedBlockKVWrapper(CachedManualBlockKVWrapper):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_key: torch.Tensor,
        cache_value: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)
        query, new_key, new_value = self._project_qkv(hidden_states)
        key = torch.cat((cache_key, new_key), dim=2)
        value = torch.cat((cache_value, new_value), dim=2)
        hidden_states = self._finish(residual, query, key, value, attention_bias)
        return hidden_states, new_key.contiguous(), new_value.contiguous()


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


def make_attention_bias(max_len: int, valid_len: int) -> torch.Tensor:
    bias = torch.full((1, 1, 1, max_len + 1), -10000.0, dtype=torch.float32)
    bias[:, :, :, :valid_len] = 0.0
    bias[:, :, :, max_len:] = 0.0
    return bias


def to_host_array(value) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def compare_arrays(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def compare_outputs(actual_outputs, expected_outputs: list[np.ndarray]) -> list[dict[str, Any]]:
    return [
        compare_arrays(to_host_array(actual), expected)
        for actual, expected in zip(actual_outputs, expected_outputs)
    ]


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
    for index, tensor in enumerate(expected_outputs):
        path = OUT_DIR / f"{name}_torch_output_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        expected_paths.append(path)

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
        "mlir_size_bytes": mlir.stat().st_size if mlir.exists() else 0,
    }


def compile_probe(probe: dict[str, Any], compile_timeout: int) -> dict[str, Any]:
    vmfb = Path(probe["vmfb"])
    compile_result = run_cmd(
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
    probe["compile"] = compile_result
    probe["vmfb_size_bytes"] = vmfb.stat().st_size if vmfb.exists() else 0
    return probe


def run_module_validation(probe: dict[str, Any], run_timeout: int) -> dict[str, Any]:
    output_paths = [OUT_DIR / f"{probe['name']}_vulkan_output_{index}.npy" for index in range(3)]
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
        for index in range(3)
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
    first_compare = compare_outputs(first_outputs, expected)

    def call_no_fetch():
        return module["forward"](*device_inputs)

    def call_fetch_outputs():
        return [to_host_array(output) for output in module["forward"](*device_inputs)]

    rss_before = rss_mb()
    timings = {
        "masked_block_no_fetch": time_call(call_no_fetch, iterations, warmup),
        "masked_block_fetch_outputs": time_call(call_fetch_outputs, iterations, warmup),
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
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--valid-len", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_masked_cache_block_vulkan_latest.json",
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
    block = t3.tfmr.h[args.block_index].eval()
    dynamic = CachedManualBlockKVWrapper(block).eval()
    masked = MaskedPreallocatedBlockKVWrapper(block).eval()

    hidden = torch.randn(1, 1, t3.dim, dtype=torch.float32)
    cache_key = torch.randn(1, masked.num_heads, args.max_len, masked.head_dim, dtype=torch.float32)
    cache_value = torch.randn(1, masked.num_heads, args.max_len, masked.head_dim, dtype=torch.float32)
    attention_bias = make_attention_bias(args.max_len, args.valid_len)

    with torch.inference_mode():
        dynamic_outputs = dynamic(
            hidden,
            cache_key[:, :, : args.valid_len, :].contiguous(),
            cache_value[:, :, : args.valid_len, :].contiguous(),
        )
        masked_outputs = masked(hidden, cache_key, cache_value, attention_bias)

    static_vs_dynamic = [
        compare_arrays(
            masked_outputs[index].detach().cpu().numpy(),
            dynamic_outputs[index].detach().cpu().numpy(),
        )
        for index in range(3)
    ]

    name = f"gpt2_masked_cache_block{args.block_index}_p{args.max_len}_valid{args.valid_len}_t1"
    probe = export_probe(name, masked, (hidden, cache_key, cache_value, attention_bias), masked_outputs)
    probe = compile_probe(probe, compile_timeout=args.compile_timeout)

    run_validation = None
    runtime = None
    if probe["compile"]["returncode"] == 0:
        run_validation = run_module_validation(probe, run_timeout=args.run_timeout)
        if run_validation["status"] == "ok":
            runtime = benchmark_runtime(probe, iterations=args.iterations, warmup=args.warmup)

    report = {
        "seconds": time.perf_counter() - started,
        "block_index": args.block_index,
        "max_len": args.max_len,
        "valid_len": args.valid_len,
        "static_vs_dynamic": static_vs_dynamic,
        "static_vs_dynamic_allclose_1e_4": all(item["allclose_1e_4"] for item in static_vs_dynamic),
        "probe": probe,
        "run_validation": run_validation,
        "runtime": runtime,
        "notes": [
            "Static cache length is fixed; valid cache slots are selected by an additive attention bias.",
            "The final current-token position is always valid.",
            "This tests masked attention over preallocated cache, not a full T3 generation loop.",
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
        for index, comparison in enumerate(run_validation.get('comparisons', [])):
            print(f"vulkan output{index}: max_abs={comparison['max_abs_error']:.3e} allclose_1e_4={comparison['allclose_1e_4']}")
    if runtime is not None:
        for name, timing in runtime["timings"].items():
            print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
        print(f"rss_delta_mb={runtime['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
