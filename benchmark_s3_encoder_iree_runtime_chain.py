#!/usr/bin/env python3
"""Benchmark a chained S3 encoder path through IREE Vulkan runtime."""

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


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
BENCH_DIR = BASE / "benchmarks"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_FLAGS = (
    "--iree-vulkan-target=gfx1013",
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class Transpose12(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.transpose(1, 2).contiguous()


def component_dir(name: str) -> Path:
    return BASE / name


def vmfb_path(name: str) -> Path:
    return component_dir(name) / f"{name}_vulkan_gfx1013.vmfb"


def npy_path(name: str, filename: str) -> Path:
    return component_dir(name) / filename


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
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


def export_transpose_if_missing(name: str, shape: tuple[int, int, int], compile_timeout: int) -> dict[str, Any]:
    vmfb = vmfb_path(name)
    if vmfb.exists():
        return {
            "name": name,
            "status": "exists",
            "vmfb": vmfb.as_posix(),
            "vmfb_size_bytes": vmfb.stat().st_size,
        }

    probe_dir = component_dir(name)
    probe_dir.mkdir(parents=True, exist_ok=True)
    module = Transpose12().eval()
    example = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    with torch.inference_mode():
        expected = module(example).detach().cpu().numpy()
    np.save(probe_dir / "input_0.npy", example.detach().cpu().numpy())
    np.save(probe_dir / "torch_output.npy", expected)

    exported = aot.export(module, args=(example,), module_name=name, function_name="forward")
    mlir_path = probe_dir / f"{name}.mlir"
    exported.save_mlir(mlir_path)
    graph_path = probe_dir / f"{name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, (example,)).graph_module) + "\n")

    compile_cmd = [
        IREE_COMPILE.as_posix(),
        mlir_path.as_posix(),
        "--iree-hal-target-backends=vulkan-spirv",
        *IREE_FLAGS,
        f"-o={vmfb}",
    ]
    compile_result = run_cmd(compile_cmd, timeout=compile_timeout)
    status = "ok" if compile_result["returncode"] == 0 and vmfb.exists() else "compile_failed"
    result = {
        "name": name,
        "status": status,
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "vmfb": vmfb.as_posix(),
        "compile": compile_result,
    }
    if status != "ok":
        raise RuntimeError(f"Failed to compile {name}: {compile_result['stderr_tail']}")
    result["vmfb_size_bytes"] = vmfb.stat().st_size
    return result


def to_host_array(value: Any) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def diff_summary(actual: Any, expected: np.ndarray, atol: float = 1e-4, rtol: float = 1e-4) -> dict[str, Any]:
    actual_np = to_host_array(actual)
    expected_np = np.asarray(expected)
    diff = np.abs(actual_np.astype(np.float32, copy=False) - expected_np.astype(np.float32, copy=False))
    return {
        "shape": list(actual_np.shape),
        "dtype": str(actual_np.dtype),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "allclose_1e_4": bool(np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)),
        "allclose_1e_3": bool(np.allclose(actual_np, expected_np, atol=1e-3, rtol=1e-3)),
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


def load_array(name: str, filename: str, dtype: np.dtype | None = None) -> np.ndarray:
    array = np.load(npy_path(name, filename))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def build_cpu_reference(token_hidden: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    encoder = model.s3gen.flow.encoder.cpu().eval()
    hidden = torch.from_numpy(token_hidden.astype(np.float32, copy=False))
    lens = torch.from_numpy(lengths.astype(np.int64, copy=False))
    with torch.inference_mode():
        expected_hidden, expected_mask = encoder(hidden, lens)
    return (
        expected_hidden.detach().cpu().numpy(),
        expected_mask.detach().cpu().numpy().astype(np.float32),
    )


def encoder_module_names(token_frames: int = 605, up_frames: int = 1210) -> dict[str, str]:
    module_names = {
        "embed": f"s3_encoder_embed_fmask_t{token_frames}",
        "pre_lookahead": f"s3_encoder_pre_lookahead_t{token_frames}",
        "lower_transpose": f"s3_encoder_transpose_t{token_frames}_c512",
        "up_layer": f"s3_encoder_up_layer_t{token_frames}",
        "up_transpose": f"s3_encoder_transpose_c512_t{up_frames}",
        "up_embed": f"s3_encoder_up_embed_fmask_t{up_frames}",
        "after_norm": f"s3_encoder_after_norm_t{up_frames}",
    }
    module_names.update(
        {f"lower_layer_{index}": f"s3_encoder_layer{index}_fmask_t{token_frames}" for index in range(6)}
    )
    module_names.update(
        {f"upper_layer_{index}": f"s3_encoder_up_layer{index}_fmask_t{up_frames}" for index in range(4)}
    )
    return module_names


def load_modules(token_frames: int = 605, up_frames: int = 1210, compile_timeout: int = 180) -> dict[str, Any]:
    export_transpose_if_missing(
        f"s3_encoder_transpose_t{token_frames}_c512",
        (1, token_frames, 512),
        compile_timeout,
    )
    export_transpose_if_missing(
        f"s3_encoder_transpose_c512_t{up_frames}",
        (1, 512, up_frames),
        compile_timeout,
    )
    module_names = encoder_module_names(token_frames, up_frames)

    missing = [path.as_posix() for path in (vmfb_path(name) for name in module_names.values()) if not path.exists()]
    if missing:
        raise SystemExit("Missing required VMFBs:\n" + "\n".join(missing))
    return {
        key: ireert.load_vm_flatbuffer_file(vmfb_path(name).as_posix(), driver="vulkan")
        for key, name in module_names.items()
    }


def component_projection_seconds(token_frames: int = 605, up_frames: int = 1210) -> float | None:
    if token_frames != 605 or up_frames != 1210:
        return None
    scorecard = BASE / "s3_flow_encoder_components_benchmark_2026-07-08.json"
    data = json.loads(scorecard.read_text())
    timings = data["vulkan_mean_seconds"]
    return float(
        timings["embed_fmask_t605"]
        + timings["pre_lookahead_t605"]
        + sum(timings[f"encoder_layer{index}_fmask_t605"] for index in range(6))
        + timings["up_layer_t605"]
        + timings["up_embed_fmask_t1210"]
        + sum(timings[f"up_encoder_layer{index}_fmask_t1210"] for index in range(4))
        + timings["after_norm_t1210"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--token-frames", type=int, default=605)
    parser.add_argument("--up-frames", type=int, default=1210)
    parser.add_argument("--skip-cpu-validation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        suffix = "" if (args.token_frames, args.up_frames) == (605, 1210) else f"_t{args.token_frames}_t{args.up_frames}"
        output = BASE / f"s3_encoder_iree_runtime_chain{suffix}_2026-07-08.json"

    transposes = [
        export_transpose_if_missing(
            f"s3_encoder_transpose_t{args.token_frames}_c512",
            (1, args.token_frames, 512),
            args.compile_timeout,
        ),
        export_transpose_if_missing(
            f"s3_encoder_transpose_c512_t{args.up_frames}",
            (1, 512, args.up_frames),
            args.compile_timeout,
        ),
    ]

    token_hidden = load_array(f"s3_encoder_embed_fmask_t{args.token_frames}", "input_0.npy", np.float32)
    token_mask = load_array(f"s3_encoder_embed_fmask_t{args.token_frames}", "input_1.npy", np.float32)
    up_mask = load_array(f"s3_encoder_up_embed_fmask_t{args.up_frames}", "input_1.npy", np.float32)
    lengths = load_array(f"s3_encoder_up_layer_t{args.token_frames}", "input_1.npy", np.int64)

    load_started = time.perf_counter()
    modules = load_modules(args.token_frames, args.up_frames, compile_timeout=args.compile_timeout)
    module_load_seconds = time.perf_counter() - load_started

    device = ireert.get_device("vulkan")
    device_token_hidden = ireert.asdevicearray(device, token_hidden, implicit_host_transfer=False)
    device_token_mask = ireert.asdevicearray(device, token_mask, implicit_host_transfer=False)
    device_up_mask = ireert.asdevicearray(device, up_mask, implicit_host_transfer=False)
    device_lengths = ireert.asdevicearray(device, lengths, implicit_host_transfer=False)

    def chain_device_no_fetch():
        lower_hidden, lower_pos, lower_mask = modules["embed"]["forward"](
            device_token_hidden,
            device_token_mask,
        )
        hidden = modules["pre_lookahead"]["forward"](lower_hidden)
        for index in range(6):
            hidden = modules[f"lower_layer_{index}"]["forward"](
                hidden,
                lower_mask,
                lower_pos,
                lower_mask,
            )
        hidden_ct = modules["lower_transpose"]["forward"](hidden)
        up_hidden_ct = modules["up_layer"]["forward"](hidden_ct, device_lengths)
        up_hidden = modules["up_transpose"]["forward"](up_hidden_ct)
        upper_hidden, upper_pos, upper_mask = modules["up_embed"]["forward"](up_hidden, device_up_mask)
        hidden = upper_hidden
        for index in range(4):
            hidden = modules[f"upper_layer_{index}"]["forward"](
                hidden,
                upper_mask,
                upper_pos,
                upper_mask,
            )
        hidden = modules["after_norm"]["forward"](hidden)
        return hidden, upper_mask

    first_hidden, first_mask = chain_device_no_fetch()

    validation = None
    if not args.skip_cpu_validation:
        expected_hidden, expected_mask = build_cpu_reference(token_hidden, lengths)
        validation = {
            "hidden": diff_summary(first_hidden, expected_hidden),
            "mask": diff_summary(first_mask, expected_mask),
        }
        validation["allclose_1e_4"] = (
            validation["hidden"]["allclose_1e_4"] and validation["mask"]["allclose_1e_4"]
        )

    def chain_device_fetch_final_hidden():
        hidden, _ = chain_device_no_fetch()
        return to_host_array(hidden)

    def chain_device_fetch_final_hidden_and_mask():
        hidden, mask = chain_device_no_fetch()
        return to_host_array(hidden), to_host_array(mask)

    rss_before = rss_mb()
    timings = {
        "device_chain_no_fetch": time_call(chain_device_no_fetch, args.iterations, args.warmup),
        "device_chain_fetch_final_hidden": time_call(
            chain_device_fetch_final_hidden,
            args.iterations,
            args.warmup,
        ),
        "device_chain_fetch_final_hidden_and_mask": time_call(
            chain_device_fetch_final_hidden_and_mask,
            args.iterations,
            args.warmup,
        ),
    }
    rss_after = rss_mb()

    projected_seconds = component_projection_seconds(args.token_frames, args.up_frames)
    no_fetch_seconds = timings["device_chain_no_fetch"]["mean_ms"] / 1000.0
    report = {
        "description": "Full fixed-shape S3 encoder IREE Vulkan runtime chain benchmark.",
        "token_frames": args.token_frames,
        "up_frames": args.up_frames,
        "shape": {
            "token_hidden": list(token_hidden.shape),
            "token_mask": list(token_mask.shape),
            "up_mask": list(up_mask.shape),
            "lengths": list(lengths.shape),
        },
        "transposes": transposes,
        "module_load_seconds": module_load_seconds,
        "validation": validation,
        "timings": timings,
        "component_projection_seconds": projected_seconds,
        "device_chain_no_fetch_seconds": no_fetch_seconds,
        "device_chain_over_component_projection": (
            no_fetch_seconds / projected_seconds if projected_seconds is not None else None
        ),
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "This is a synthetic all-valid chunk270-sized encoder shape, not a full API run.",
            "The chain keeps hidden, position, and mask tensors as IREE DeviceArray values across module calls.",
            "Two fixed-shape transpose VMFBs cover the layout changes around the upsample layer.",
            "The pre-lookahead layer is included on Vulkan here; the prior mixed projection kept it on CPU because the isolated Vulkan timing was slower than CPU.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={output}")
    print(f"module_load_seconds={module_load_seconds:.3f}")
    if validation is not None:
        print(
            "validation_allclose_1e_4="
            f"{validation['allclose_1e_4']} "
            f"hidden_max_abs={validation['hidden']['max_abs_error']:.3e}"
        )
    for name, timing in timings.items():
        print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
    if projected_seconds is not None:
        print(f"component_projection={projected_seconds * 1000.0:.3f} ms")
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")


if __name__ == "__main__":
    main()
