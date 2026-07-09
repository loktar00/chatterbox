#!/usr/bin/env python3
"""Probe a fused S3 estimator mid block on IREE Vulkan.

The current S3 estimator path is correct but dispatch-heavy: each mid block is
stitched from separate ResNet, transpose, transformer, and transpose VMFBs. This
probe tests fusing one mid block into a single VMFB without changing the math.
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
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
from iree.turbine import aot
from torch import nn

from benchmark_s3_estimator_distinct_iree_runtime_chain import (
    distinct_vmfb,
    helper_vmfb,
    to_host_array,
)
from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
OUT_DIR = BASE / "fused_estimator"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
IREE_FLAGS = (
    "--iree-vulkan-target=gfx1013",
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class TransformerWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            timestep=timestep,
        )


class FusedMidBlock(nn.Module):
    def __init__(self, mid_block: nn.Module) -> None:
        super().__init__()
        self.resnet = mid_block[0]
        self.transformers = nn.ModuleList(TransformerWrapper(block) for block in mid_block[1])

    def forward(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.resnet(hidden, mask, time_emb)
        hidden = hidden.transpose(1, 2).contiguous()
        for block in self.transformers:
            hidden = block(hidden, attention_bias, time_emb)
        return hidden.transpose(1, 2).contiguous()


def force_eager_attention(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(AttnProcessor())
            count += 1
    return count


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


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


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


def load_npy(path: Path, dtype: np.dtype | None = np.float32) -> np.ndarray:
    array = np.load(path)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def compile_vulkan(mlir_path: Path, vmfb_path: Path, timeout: int, force: bool) -> dict[str, Any]:
    if vmfb_path.exists() and not force:
        return {"status": "exists", "vmfb_size_bytes": vmfb_path.stat().st_size}
    cmd = [
        IREE_COMPILE.as_posix(),
        mlir_path.as_posix(),
        "--iree-hal-target-backends=vulkan-spirv",
        *IREE_FLAGS,
        f"-o={vmfb_path}",
    ]
    result = run_cmd(cmd, timeout=timeout)
    result["status"] = "ok" if result["returncode"] == 0 and vmfb_path.exists() else "failed"
    if vmfb_path.exists():
        result["vmfb_size_bytes"] = vmfb_path.stat().st_size
    return result


def run_vulkan_module(
    vmfb_path: Path,
    input_paths: list[Path],
    output_path: Path,
    timeout: int,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()
    cmd = [
        IREE_RUN.as_posix(),
        f"--module={vmfb_path}",
        "--device=vulkan",
        "--function=forward",
        *[f"--input=@{path}" for path in input_paths],
        f"--output=@{output_path}",
    ]
    result = run_cmd(cmd, timeout=timeout)
    result["status"] = "ok" if result["returncode"] == 0 and output_path.exists() else "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1210)
    parser.add_argument("--mid-index", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=600)
    parser.add_argument("--run-timeout", type=int, default=180)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_fused_midblock0_t1210_probe_2026-07-08.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass

    name = f"s3_fused_mid{args.mid_index}_block_t{args.frames}"
    probe_dir = OUT_DIR / name
    probe_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = probe_dir / f"{name}.mlir"
    graph_path = probe_dir / f"{name}.torch_export.txt"
    vmfb_path = probe_dir / f"{name}_vulkan_gfx1013.vmfb"
    output_path = probe_dir / "iree_vulkan_output.npy"
    input_paths = [
        probe_dir / "input_hidden.npy",
        probe_dir / "input_mask.npy",
        probe_dir / "input_attention_bias.npy",
        probe_dir / "input_time_emb.npy",
    ]
    expected_path = probe_dir / "torch_output.npy"

    result: dict[str, Any] = {
        "description": "Fused S3 estimator mid-block IREE Vulkan probe.",
        "frames": args.frames,
        "mid_index": args.mid_index,
        "module_name": name,
        "iree_flags": list(IREE_FLAGS),
        "paths": {
            "probe_dir": probe_dir.as_posix(),
            "mlir": mlir_path.as_posix(),
            "vmfb": vmfb_path.as_posix(),
            "expected": expected_path.as_posix(),
            "output": output_path.as_posix(),
        },
        "rss_start_mb": rss_mb(),
    }
    started_all = time.perf_counter()

    export_needed = (
        args.force_export
        or not mlir_path.exists()
        or not expected_path.exists()
        or not all(path.exists() for path in input_paths)
    )
    if export_needed:
        torch.manual_seed(20260708 + args.frames + args.mid_index)
        load_started = time.perf_counter()
        model = ChatterboxTurboTTS.from_pretrained("cpu")
        estimator = model.s3gen.flow.decoder.estimator.cpu().eval()
        result["model_load_seconds"] = time.perf_counter() - load_started
        result["attention_processors_changed"] = force_eager_attention(estimator)
        module = FusedMidBlock(estimator.mid_blocks[args.mid_index]).cpu().eval()

        hidden = torch.randn(1, 256, args.frames, dtype=torch.float32)
        mask = torch.ones(1, 1, args.frames, dtype=torch.float32)
        attention_bias = torch.zeros(1, 1, args.frames, dtype=torch.float32)
        time_emb = torch.randn(1, 1024, dtype=torch.float32)
        torch_inputs = (hidden, mask, attention_bias, time_emb)

        with torch.inference_mode():
            expected = module(*torch_inputs).detach().cpu().numpy()
        for path, tensor in zip(input_paths, torch_inputs, strict=True):
            np.save(path, tensor.detach().cpu().numpy())
        np.save(expected_path, expected)

        export_started = time.perf_counter()
        exported = aot.export(module, args=torch_inputs, module_name=name, function_name="forward")
        exported.save_mlir(mlir_path)
        graph_path.write_text(str(torch.export.export(module, torch_inputs).graph_module) + "\n")
        result["export_seconds"] = time.perf_counter() - export_started
        result["export_status"] = "exported"
    else:
        result["export_status"] = "exists"

    result["compile"] = compile_vulkan(
        mlir_path,
        vmfb_path,
        timeout=args.compile_timeout,
        force=args.force_compile,
    )
    if result["compile"].get("status") != "ok" and result["compile"].get("status") != "exists":
        result["status"] = "compile_failed"
        result["total_seconds"] = time.perf_counter() - started_all
        result["rss_end_mb"] = rss_mb()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"results={args.output}")
        return

    result["run"] = run_vulkan_module(vmfb_path, input_paths, output_path, timeout=args.run_timeout)
    if result["run"].get("status") != "ok":
        result["status"] = "run_failed"
        result["total_seconds"] = time.perf_counter() - started_all
        result["rss_end_mb"] = rss_mb()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"results={args.output}")
        return

    expected_np = load_npy(expected_path)
    actual_np = load_npy(output_path)
    result["validation"] = diff_summary(actual_np, expected_np)

    device = ireert.get_device("vulkan")
    hidden = ireert.asdevicearray(device, load_npy(input_paths[0]), implicit_host_transfer=False)
    mask = ireert.asdevicearray(device, load_npy(input_paths[1]), implicit_host_transfer=False)
    attention_bias = ireert.asdevicearray(device, load_npy(input_paths[2]), implicit_host_transfer=False)
    time_emb = ireert.asdevicearray(device, load_npy(input_paths[3]), implicit_host_transfer=False)

    fused_module = ireert.load_vm_flatbuffer_file(vmfb_path.as_posix(), driver="vulkan")
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

    def fused_no_fetch():
        return fused_module["forward"](hidden, mask, attention_bias, time_emb)

    def stitched_no_fetch():
        value = mid_resnet["forward"](hidden, mask, time_emb)
        value = c_to_t["forward"](value)
        for transformer in transformers:
            value = transformer["forward"](value, attention_bias, time_emb)
        return t_to_c["forward"](value)

    stitched_output = to_host_array(stitched_no_fetch())
    fused_output = to_host_array(fused_no_fetch())
    result["fused_vs_stitched"] = diff_summary(fused_output, stitched_output)
    result["timings"] = {
        "fused_no_fetch": time_call(fused_no_fetch, args.iterations, args.warmup),
        "stitched_no_fetch": time_call(stitched_no_fetch, args.iterations, args.warmup),
        "fused_fetch_output": time_call(lambda: to_host_array(fused_no_fetch()), args.iterations, args.warmup),
        "stitched_fetch_output": time_call(
            lambda: to_host_array(stitched_no_fetch()),
            args.iterations,
            args.warmup,
        ),
    }
    fused_ms = result["timings"]["fused_no_fetch"]["mean_ms"]
    stitched_ms = result["timings"]["stitched_no_fetch"]["mean_ms"]
    result["speedup_vs_stitched_no_fetch"] = stitched_ms / fused_ms if fused_ms else None
    result["status"] = "ok"
    result["total_seconds"] = time.perf_counter() - started_all
    result["rss_end_mb"] = rss_mb()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results={args.output}")
    print(f"status={result['status']}")
    print(f"validation_allclose_1e_4={result['validation']['allclose_1e_4']}")
    print(f"fused_no_fetch_ms={fused_ms:.3f}")
    print(f"stitched_no_fetch_ms={stitched_ms:.3f}")
    print(f"speedup_vs_stitched_no_fetch={result['speedup_vs_stitched_no_fetch']:.3f}")


if __name__ == "__main__":
    main()
