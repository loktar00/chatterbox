#!/usr/bin/env python3
"""Benchmark the full distinct-weight S3 estimator chain with IREE Vulkan."""

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
DISTINCT = BASE / "distinct_estimator"
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous()


class CatChannel(nn.Module):
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat((x[:, :, : skip.shape[-1]], skip), dim=1)


def force_eager_attention(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(AttnProcessor())
            count += 1
    return count


def distinct_vmfb(name: str) -> Path:
    return DISTINCT / name / f"{name}_vulkan_gfx1013.vmfb"


def helper_vmfb(name: str) -> Path:
    return BASE / name / f"{name}_vulkan_gfx1013.vmfb"


def distinct_npy(name: str, filename: str) -> Path:
    return DISTINCT / name / filename


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
    vmfb = helper_vmfb(name)
    if vmfb.exists():
        return {
            "name": name,
            "status": "exists",
            "vmfb": vmfb.as_posix(),
            "vmfb_size_bytes": vmfb.stat().st_size,
        }

    helper_dir = BASE / name
    helper_dir.mkdir(parents=True, exist_ok=True)
    args = tuple(arg.contiguous() for arg in args)
    with torch.inference_mode():
        expected = module.eval()(*args).detach().cpu().numpy()
    for index, arg in enumerate(args):
        np.save(helper_dir / f"input_{index}.npy", arg.detach().cpu().numpy())
    np.save(helper_dir / "torch_output.npy", expected)

    exported = aot.export(module, args=args, module_name=name, function_name="forward")
    mlir_path = helper_dir / f"{name}.mlir"
    exported.save_mlir(mlir_path)
    graph_path = helper_dir / f"{name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, args).graph_module) + "\n")

    compile_result = run_cmd(
        [
            IREE_COMPILE.as_posix(),
            mlir_path.as_posix(),
            "--iree-hal-target-backends=vulkan-spirv",
            *IREE_FLAGS,
            f"-o={vmfb}",
        ],
        timeout=compile_timeout,
    )
    if compile_result["returncode"] != 0 or not vmfb.exists():
        raise RuntimeError(f"Failed to compile helper {name}: {compile_result['stderr_tail']}")
    return {
        "name": name,
        "status": "compiled",
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "vmfb": vmfb.as_posix(),
        "vmfb_size_bytes": vmfb.stat().st_size,
        "compile": compile_result,
    }


def ensure_helper_vmfbs(frames: int, compile_timeout: int) -> list[dict[str, Any]]:
    return [
        export_helper_if_missing(
            f"s3_flow_transpose_c256_t{frames}",
            Transpose12(),
            (torch.randn(1, 256, frames, dtype=torch.float32),),
            compile_timeout,
        ),
        export_helper_if_missing(
            f"s3_flow_transpose_t{frames}_c256",
            Transpose12(),
            (torch.randn(1, frames, 256, dtype=torch.float32),),
            compile_timeout,
        ),
        export_helper_if_missing(
            f"s3_flow_cat_channel_256_256_t{frames}",
            CatChannel(),
            (
                torch.randn(1, 256, frames, dtype=torch.float32),
                torch.randn(1, 256, frames, dtype=torch.float32),
            ),
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
    array = np.load(distinct_npy(name, filename))
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def load_input_array(path: Path, dtype: np.dtype | None = np.float32) -> np.ndarray:
    array = np.load(path)
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def module_names(frames: int = 1210) -> dict[str, str]:
    names = {
        "down_resnet": f"s3_distinct_down_resnet0_t{frames}",
        "downsample": f"s3_distinct_downsample0_t{frames}",
        "up_resnet": f"s3_distinct_up_resnet0_t{frames}",
        "upsample": f"s3_distinct_upsample0_t{frames}",
        "final_block": f"s3_distinct_final_block_t{frames}",
        "final_proj": f"s3_distinct_final_proj_t{frames}",
    }
    names.update({f"down_transformer_{index}": f"s3_distinct_down_transformer{index}_t{frames}" for index in range(4)})
    names.update({f"up_transformer_{index}": f"s3_distinct_up_transformer{index}_t{frames}" for index in range(4)})
    names.update({f"mid_resnet_{index}": f"s3_distinct_mid_resnet{index}_t{frames}" for index in range(12)})
    names.update(
        {
            f"mid_transformer_{mid}_{block}": f"s3_distinct_mid{mid}_transformer{block}_t{frames}"
            for mid in range(12)
            for block in range(4)
        }
    )
    return names


def load_modules(frames: int = 1210, compile_timeout: int = 180) -> tuple[dict[str, Any], dict[str, Any]]:
    ensure_helper_vmfbs(frames, compile_timeout)
    names = module_names(frames)
    missing = [distinct_vmfb(name).as_posix() for name in names.values() if not distinct_vmfb(name).exists()]
    helper_names = {
        "c_to_t": f"s3_flow_transpose_c256_t{frames}",
        "t_to_c": f"s3_flow_transpose_t{frames}_c256",
        "cat": f"s3_flow_cat_channel_256_256_t{frames}",
    }
    missing.extend(helper_vmfb(name).as_posix() for name in helper_names.values() if not helper_vmfb(name).exists())
    if missing:
        raise SystemExit("Missing required VMFBs:\n" + "\n".join(missing))
    modules = {
        key: ireert.load_vm_flatbuffer_file(distinct_vmfb(name).as_posix(), driver="vulkan")
        for key, name in names.items()
    }
    helpers = {
        key: ireert.load_vm_flatbuffer_file(helper_vmfb(name).as_posix(), driver="vulkan")
        for key, name in helper_names.items()
    }
    return modules, helpers


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
    down_transformers = [TransformerWrapper(block).eval() for block in estimator.down_blocks[0][1]]
    mid_transformers = [
        [TransformerWrapper(block).eval() for block in estimator.mid_blocks[mid][1]]
        for mid in range(12)
    ]
    up_transformers = [TransformerWrapper(block).eval() for block in estimator.up_blocks[0][1]]

    x = torch.from_numpy(x_np)
    mask = torch.from_numpy(mask_np)
    attention_bias = torch.from_numpy(attention_bias_np)
    time_emb = torch.from_numpy(time_emb_np)

    with torch.inference_mode():
        x = estimator.down_blocks[0][0](x, mask, time_emb)
        x = x.transpose(1, 2).contiguous()
        for block in down_transformers:
            x = block(x, attention_bias, time_emb)
        x = x.transpose(1, 2).contiguous()
        skip = x
        x = estimator.down_blocks[0][2](x * mask)

        for mid in range(12):
            x = estimator.mid_blocks[mid][0](x, mask, time_emb)
            x = x.transpose(1, 2).contiguous()
            for block in mid_transformers[mid]:
                x = block(x, attention_bias, time_emb)
            x = x.transpose(1, 2).contiguous()

        x = torch.cat((x[:, :, : skip.shape[-1]], skip), dim=1)
        x = estimator.up_blocks[0][0](x, mask, time_emb)
        x = x.transpose(1, 2).contiguous()
        for block in up_transformers:
            x = block(x, attention_bias, time_emb)
        x = x.transpose(1, 2).contiguous()
        x = estimator.up_blocks[0][2](x * mask)
        x = estimator.final_block(x, mask)
        output = estimator.final_proj(x * mask)
        output = output * mask
    return output.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--frames", type=int, default=1210)
    parser.add_argument("--helper-compile-timeout", type=int, default=180)
    parser.add_argument("--skip-cpu-validation", action="store_true")
    parser.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Directory containing packed_x.npy, mask.npy, time_emb.npy, "
            "attention_bias.npy, and optionally cpu_output.npy."
        ),
    )
    parser.add_argument(
        "--expected-output",
        type=Path,
        help="Optional expected estimator output .npy; defaults to input-dir/cpu_output.npy when present.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        suffix = "" if args.frames == 1210 else f"_t{args.frames}"
        output = BASE / f"s3_estimator_distinct_iree_runtime_chain{suffix}_2026-07-08.json"

    input_source = "synthetic_distinct_fixture"
    expected_output = args.expected_output
    if args.input_dir is not None:
        input_source = args.input_dir.as_posix()
        x = load_input_array(args.input_dir / "packed_x.npy")
        mask = load_input_array(args.input_dir / "mask.npy")
        time_emb = load_input_array(args.input_dir / "time_emb.npy")
        attention_bias = load_input_array(args.input_dir / "attention_bias.npy")
        default_expected = args.input_dir / "cpu_output.npy"
        if expected_output is None and default_expected.exists():
            expected_output = default_expected
    else:
        x = load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_0.npy")
        mask = load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_1.npy")
        time_emb = load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_2.npy")
        attention_bias = load_array(f"s3_distinct_down_transformer0_t{args.frames}", "input_1.npy")

    if list(x.shape) != [1, 320, args.frames]:
        raise SystemExit(f"Expected packed_x shape [1, 320, {args.frames}], got {list(x.shape)}")
    if list(mask.shape) != [1, 1, args.frames]:
        raise SystemExit(f"Expected mask shape [1, 1, {args.frames}], got {list(mask.shape)}")
    if list(time_emb.shape) != [1, 1024]:
        raise SystemExit(f"Expected time_emb shape [1, 1024], got {list(time_emb.shape)}")
    if list(attention_bias.shape) != [1, 1, args.frames]:
        raise SystemExit(f"Expected attention_bias shape [1, 1, {args.frames}], got {list(attention_bias.shape)}")

    load_started = time.perf_counter()
    modules, helpers = load_modules(args.frames, compile_timeout=args.helper_compile_timeout)
    module_load_seconds = time.perf_counter() - load_started

    device = ireert.get_device("vulkan")
    device_x = ireert.asdevicearray(device, x, implicit_host_transfer=False)
    device_mask = ireert.asdevicearray(device, mask, implicit_host_transfer=False)
    device_time = ireert.asdevicearray(device, time_emb, implicit_host_transfer=False)
    device_attention_bias = ireert.asdevicearray(device, attention_bias, implicit_host_transfer=False)

    def chain_device_no_fetch():
        hidden = modules["down_resnet"]["forward"](device_x, device_mask, device_time)
        hidden = helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = modules[f"down_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = helpers["t_to_c"]["forward"](hidden)
        skip = hidden
        hidden = modules["downsample"]["forward"](hidden)

        for mid in range(12):
            hidden = modules[f"mid_resnet_{mid}"]["forward"](hidden, device_mask, device_time)
            hidden = helpers["c_to_t"]["forward"](hidden)
            for block in range(4):
                hidden = modules[f"mid_transformer_{mid}_{block}"]["forward"](
                    hidden,
                    device_attention_bias,
                    device_time,
                )
            hidden = helpers["t_to_c"]["forward"](hidden)

        hidden = helpers["cat"]["forward"](hidden, skip)
        hidden = modules["up_resnet"]["forward"](hidden, device_mask, device_time)
        hidden = helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = modules[f"up_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = helpers["t_to_c"]["forward"](hidden)
        hidden = modules["upsample"]["forward"](hidden)
        hidden = modules["final_block"]["forward"](hidden, device_mask)
        return modules["final_proj"]["forward"](hidden)

    first_output = chain_device_no_fetch()
    validation = None
    if not args.skip_cpu_validation:
        if expected_output is not None:
            expected = load_input_array(expected_output)
        else:
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

    no_fetch_seconds = timings["device_chain_no_fetch"]["mean_ms"] / 1000.0
    notes = [
        f"This validates the full distinct-weight estimator blocks for the fixed T={args.frames} bucket.",
        "The live API remains CPU-only.",
    ]
    if args.input_dir is not None:
        notes.insert(
            1,
            "This uses supplied estimator input tensors; it is still an estimator-chain benchmark, not a full API request.",
        )
    else:
        notes.insert(1, "This is still a synthetic estimator-chain benchmark, not a full API request.")

    report = {
        "description": "Full distinct-weight fixed-shape S3 estimator IREE Vulkan runtime chain benchmark.",
        "frames": args.frames,
        "input_source": input_source,
        "expected_output": expected_output.as_posix() if expected_output is not None else None,
        "shape": {
            "x": list(x.shape),
            "mask": list(mask.shape),
            "attention_bias": list(attention_bias.shape),
            "time_emb": list(time_emb.shape),
        },
        "component_counts": {
            "distinct_vmfb_modules": len(module_names(args.frames)),
            "helper_vmfb_modules": 3,
            "down_transformers": 4,
            "mid_transformers": 48,
            "up_transformers": 4,
            "mid_resnets": 12,
        },
        "module_load_seconds": module_load_seconds,
        "validation_against_distinct_cpu_chain": validation,
        "timings": timings,
        "device_chain_no_fetch_seconds": no_fetch_seconds,
        "two_estimator_calls_projected_seconds": no_fetch_seconds * 2.0,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": notes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={output}")
    print(f"module_load_seconds={module_load_seconds:.3f}")
    if validation is not None:
        print(
            "distinct_validation_allclose_1e_4="
            f"{validation['allclose_1e_4']} "
            f"max_abs={validation['max_abs_error']:.3e}"
        )
    for name, timing in timings.items():
        print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")


if __name__ == "__main__":
    main()
