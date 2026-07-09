#!/usr/bin/env python3
"""Benchmark a representative S3 estimator chain through IREE Vulkan runtime.

This repeats the already-exported representative down/mid/up blocks in the real
estimator call pattern. It measures runtime handoff overhead before compiling
all distinct estimator weights.
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

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
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


class Transpose12(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.transpose(1, 2).contiguous()


class CatChannel(nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.cat((left, right), dim=1)


def force_eager_attention(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(AttnProcessor())
            count += 1
    return count


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


def export_helper_if_missing(
    name: str,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    compile_timeout: int,
) -> dict[str, Any]:
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
    module = module.eval()
    with torch.inference_mode():
        expected = module(*args).detach().cpu().numpy()
    for index, arg in enumerate(args):
        np.save(probe_dir / f"input_{index}.npy", arg.detach().cpu().numpy())
    np.save(probe_dir / "torch_output.npy", expected)

    exported = aot.export(module, args=args, module_name=name, function_name="forward")
    mlir_path = probe_dir / f"{name}.mlir"
    exported.save_mlir(mlir_path)
    graph_path = probe_dir / f"{name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, args).graph_module) + "\n")

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


def export_helpers(compile_timeout: int) -> list[dict[str, Any]]:
    c_t = torch.randn(1, 256, 1210, dtype=torch.float32)
    t_c = torch.randn(1, 1210, 256, dtype=torch.float32)
    return [
        export_helper_if_missing(
            "s3_flow_transpose_c256_t1210",
            Transpose12(),
            (c_t,),
            compile_timeout,
        ),
        export_helper_if_missing(
            "s3_flow_transpose_t1210_c256",
            Transpose12(),
            (t_c,),
            compile_timeout,
        ),
        export_helper_if_missing(
            "s3_flow_cat_channel_256_256_t1210",
            CatChannel(),
            (c_t, torch.randn(1, 256, 1210, dtype=torch.float32)),
            compile_timeout,
        ),
    ]


def to_host_array(value: Any) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def diff_summary(actual: Any, expected: np.ndarray) -> dict[str, Any]:
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
        "allclose_1e_4": bool(np.allclose(actual_np, expected_np, atol=1e-4, rtol=1e-4)),
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


def load_array(name: str, filename: str, dtype: np.dtype | None = np.float32) -> np.ndarray:
    array = np.load(npy_path(name, filename))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def build_cpu_reference(
    x_np: np.ndarray,
    mask_np: np.ndarray,
    attention_bias_np: np.ndarray,
    time_emb_np: np.ndarray,
) -> np.ndarray:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    estimator = model.s3gen.flow.decoder.estimator.cpu().eval()
    force_eager_attention(estimator)

    down_transformer = TransformerWrapper(estimator.down_blocks[0][1][0]).eval()
    mid_transformer = TransformerWrapper(estimator.mid_blocks[0][1][0]).eval()
    up_transformer = TransformerWrapper(estimator.up_blocks[0][1][0]).eval()

    x = torch.from_numpy(x_np)
    mask = torch.from_numpy(mask_np)
    attention_bias = torch.from_numpy(attention_bias_np)
    time_emb = torch.from_numpy(time_emb_np)

    with torch.inference_mode():
        x = estimator.down_blocks[0][0](x, mask, time_emb)
        x = x.transpose(1, 2).contiguous()
        for _ in range(4):
            x = down_transformer(x, attention_bias, time_emb)
        x = x.transpose(1, 2).contiguous()
        skip = x
        x = estimator.down_blocks[0][2](x * mask)

        for _ in range(12):
            x = estimator.mid_blocks[0][0](x, mask, time_emb)
            x = x.transpose(1, 2).contiguous()
            for _ in range(4):
                x = mid_transformer(x, attention_bias, time_emb)
            x = x.transpose(1, 2).contiguous()

        x = torch.cat((x[:, :, : skip.shape[-1]], skip), dim=1)
        x = estimator.up_blocks[0][0](x, mask, time_emb)
        x = x.transpose(1, 2).contiguous()
        for _ in range(4):
            x = up_transformer(x, attention_bias, time_emb)
        x = x.transpose(1, 2).contiguous()
        x = estimator.up_blocks[0][2](x * mask)
        x = estimator.final_block(x, mask)
        output = estimator.final_proj(x * mask)
        output = output * mask
    return output.detach().cpu().numpy()


def load_modules() -> dict[str, Any]:
    names = {
        "down_resnet": "s3_flow_down_resnet_t1210",
        "down_transformer": "s3_flow_down_transformer_bias_t1210",
        "downsample": "s3_flow_downsample_t1210",
        "mid_resnet": "s3_flow_mid_resnet_full_t1210",
        "mid_transformer": "s3_flow_mid_transformer_full_t1210",
        "up_resnet": "s3_flow_up_resnet_t1210",
        "up_transformer": "s3_flow_up_transformer_t1210",
        "upsample": "s3_flow_upsample_t1210",
        "final_block": "s3_flow_final_block_t1210",
        "final_proj": "s3_flow_final_proj_t1210",
        "c_to_t": "s3_flow_transpose_c256_t1210",
        "t_to_c": "s3_flow_transpose_t1210_c256",
        "cat": "s3_flow_cat_channel_256_256_t1210",
    }
    missing = [vmfb_path(name).as_posix() for name in names.values() if not vmfb_path(name).exists()]
    if missing:
        raise SystemExit("Missing required VMFBs:\n" + "\n".join(missing))
    return {
        key: ireert.load_vm_flatbuffer_file(vmfb_path(name).as_posix(), driver="vulkan")
        for key, name in names.items()
    }


def representative_projection_seconds() -> float:
    data = json.loads((BASE / "s3_flow_actualshape_estimator_components_benchmark_2026-07-08.json").read_text())
    ms = data["vulkan_mean_ms"]
    return float(
        ms["down_resnet_t1210"]
        + 4 * ms["down_transformer_bias_t1210"]
        + ms["downsample_t1210"]
        + 12 * ms["mid_resnet_full_t1210"]
        + 48 * ms["mid_transformer_full_t1210"]
        + ms["up_resnet_t1210"]
        + 4 * ms["up_transformer_t1210"]
        + ms["upsample_t1210"]
        + ms["final_block_t1210"]
        + ms["final_proj_t1210"]
    ) / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=180)
    parser.add_argument("--skip-cpu-validation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_estimator_representative_iree_runtime_chain_2026-07-08.json",
    )
    args = parser.parse_args()

    helpers = export_helpers(args.compile_timeout)

    x = load_array("s3_flow_down_resnet_t1210", "input_0.npy")
    mask = load_array("s3_flow_down_resnet_t1210", "input_1.npy")
    time_emb = load_array("s3_flow_down_resnet_t1210", "input_2.npy")
    attention_bias = load_array("s3_flow_down_transformer_bias_t1210", "input_1.npy")

    load_started = time.perf_counter()
    modules = load_modules()
    module_load_seconds = time.perf_counter() - load_started

    device = ireert.get_device("vulkan")
    device_x = ireert.asdevicearray(device, x, implicit_host_transfer=False)
    device_mask = ireert.asdevicearray(device, mask, implicit_host_transfer=False)
    device_time = ireert.asdevicearray(device, time_emb, implicit_host_transfer=False)
    device_attention_bias = ireert.asdevicearray(device, attention_bias, implicit_host_transfer=False)

    def chain_device_no_fetch():
        hidden = modules["down_resnet"]["forward"](device_x, device_mask, device_time)
        hidden = modules["c_to_t"]["forward"](hidden)
        for _ in range(4):
            hidden = modules["down_transformer"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = modules["t_to_c"]["forward"](hidden)
        skip = hidden
        hidden = modules["downsample"]["forward"](hidden)

        for _ in range(12):
            hidden = modules["mid_resnet"]["forward"](hidden, device_mask, device_time)
            hidden = modules["c_to_t"]["forward"](hidden)
            for _ in range(4):
                hidden = modules["mid_transformer"]["forward"](
                    hidden,
                    device_attention_bias,
                    device_time,
                )
            hidden = modules["t_to_c"]["forward"](hidden)

        hidden = modules["cat"]["forward"](hidden, skip)
        hidden = modules["up_resnet"]["forward"](hidden, device_mask, device_time)
        hidden = modules["c_to_t"]["forward"](hidden)
        for _ in range(4):
            hidden = modules["up_transformer"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = modules["t_to_c"]["forward"](hidden)
        hidden = modules["upsample"]["forward"](hidden)
        hidden = modules["final_block"]["forward"](hidden, device_mask)
        return modules["final_proj"]["forward"](hidden)

    first_output = chain_device_no_fetch()

    validation = None
    if not args.skip_cpu_validation:
        expected = build_cpu_reference(x, mask, attention_bias, time_emb)
        validation = diff_summary(first_output, expected)

    def chain_device_fetch_final_output():
        return to_host_array(chain_device_no_fetch())

    rss_before = rss_mb()
    timings = {
        "device_chain_no_fetch": time_call(chain_device_no_fetch, args.iterations, args.warmup),
        "device_chain_fetch_final_output": time_call(
            chain_device_fetch_final_output,
            args.iterations,
            args.warmup,
        ),
    }
    rss_after = rss_mb()

    projected_seconds = representative_projection_seconds()
    no_fetch_seconds = timings["device_chain_no_fetch"]["mean_ms"] / 1000.0
    report = {
        "description": "Representative fixed-shape S3 estimator IREE Vulkan runtime chain benchmark.",
        "limitation": "Uses representative down/mid/up ResNet and transformer weights repeated in the real call pattern; it is not the full distinct-weight estimator.",
        "shape": {
            "x": list(x.shape),
            "mask": list(mask.shape),
            "attention_bias": list(attention_bias.shape),
            "time_emb": list(time_emb.shape),
        },
        "call_pattern": {
            "down_resnet": 1,
            "down_transformer_representative": 4,
            "downsample": 1,
            "mid_resnet_representative": 12,
            "mid_transformer_representative": 48,
            "up_resnet": 1,
            "up_transformer_representative": 4,
            "upsample": 1,
            "final_block": 1,
            "final_proj": 1,
        },
        "helpers": helpers,
        "module_load_seconds": module_load_seconds,
        "validation_against_representative_cpu_chain": validation,
        "timings": timings,
        "component_projection_seconds": projected_seconds,
        "device_chain_no_fetch_seconds": no_fetch_seconds,
        "device_chain_over_component_projection": no_fetch_seconds / projected_seconds,
        "two_estimator_calls_projected_seconds": no_fetch_seconds * 2.0,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "This tests IREE runtime orchestration and DeviceArray handoff for the estimator call pattern.",
            "Full model validation still requires compiling all distinct estimator blocks.",
            "The live API remains CPU-only.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={args.output}")
    print(f"module_load_seconds={module_load_seconds:.3f}")
    if validation is not None:
        print(
            "representative_validation_allclose_1e_4="
            f"{validation['allclose_1e_4']} "
            f"max_abs={validation['max_abs_error']:.3e}"
        )
    for name, timing in timings.items():
        print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
    print(f"component_projection={projected_seconds * 1000.0:.3f} ms")
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")


if __name__ == "__main__":
    main()
