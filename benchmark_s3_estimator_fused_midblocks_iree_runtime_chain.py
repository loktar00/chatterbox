#!/usr/bin/env python3
"""Benchmark stitched versus fused-midblock S3 estimator chains on IREE Vulkan.

This is a bounded fixed-shape benchmark. It does not generate audio, does not
start an API worker, and does not probe ROCm/HIP.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import iree.runtime as ireert
import numpy as np

from benchmark_s3_estimator_distinct_iree_runtime_chain import (
    BASE,
    ROOT,
    diff_summary,
    load_array,
    load_input_array,
    load_modules,
    module_names,
    rss_mb,
    time_call,
    to_host_array,
)


FUSED = BASE / "fused_estimator"


def fused_vmfb(frames: int, mid_index: int, variant: str) -> Path:
    name = f"s3_fused_mid{mid_index}_block_t{frames}"
    return FUSED / name / "compile_flag_matrix" / f"{name}_{variant}_vulkan_gfx1013.vmfb"


def load_fused_midblocks(frames: int, variant: str) -> dict[int, Any]:
    missing = [fused_vmfb(frames, mid, variant).as_posix() for mid in range(12) if not fused_vmfb(frames, mid, variant).exists()]
    if missing:
        raise SystemExit("Missing required fused midblock VMFBs:\n" + "\n".join(missing))
    return {
        mid: ireert.load_vm_flatbuffer_file(fused_vmfb(frames, mid, variant).as_posix(), driver="vulkan")
        for mid in range(12)
    }


def load_inputs(args: argparse.Namespace) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.input_dir is not None:
        return (
            args.input_dir.as_posix(),
            load_input_array(args.input_dir / "packed_x.npy"),
            load_input_array(args.input_dir / "mask.npy"),
            load_input_array(args.input_dir / "attention_bias.npy"),
            load_input_array(args.input_dir / "time_emb.npy"),
        )
    return (
        "synthetic_distinct_fixture",
        load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_0.npy"),
        load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_1.npy"),
        load_array(f"s3_distinct_down_transformer0_t{args.frames}", "input_1.npy"),
        load_array(f"s3_distinct_down_resnet0_t{args.frames}", "input_2.npy"),
    )


def validate_shapes(frames: int, x: np.ndarray, mask: np.ndarray, attention_bias: np.ndarray, time_emb: np.ndarray) -> None:
    expected = {
        "packed_x": ([1, 320, frames], list(x.shape)),
        "mask": ([1, 1, frames], list(mask.shape)),
        "attention_bias": ([1, 1, frames], list(attention_bias.shape)),
        "time_emb": ([1, 1024], list(time_emb.shape)),
    }
    bad = [f"{name}: expected {want}, got {got}" for name, (want, got) in expected.items() if want != got]
    if bad:
        raise SystemExit("Input shape mismatch:\n" + "\n".join(bad))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1222)
    parser.add_argument("--variant", default="split8")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--helper-compile-timeout", type=int, default=180)
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Optional directory with packed_x.npy, mask.npy, attention_bias.npy, and time_emb.npy.",
    )
    parser.add_argument("--expected-output", type=Path, help="Optional expected estimator output .npy.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_estimator_fused_midblocks_t1222_split8_chain_2026-07-08.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    input_source, x, mask, attention_bias, time_emb = load_inputs(args)
    validate_shapes(args.frames, x, mask, attention_bias, time_emb)

    load_started = time.perf_counter()
    modules, helpers = load_modules(args.frames, compile_timeout=args.helper_compile_timeout)
    fused_midblocks = load_fused_midblocks(args.frames, args.variant)
    module_load_seconds = time.perf_counter() - load_started

    device = ireert.get_device("vulkan")
    device_x = ireert.asdevicearray(device, x, implicit_host_transfer=False)
    device_mask = ireert.asdevicearray(device, mask, implicit_host_transfer=False)
    device_attention_bias = ireert.asdevicearray(device, attention_bias, implicit_host_transfer=False)
    device_time = ireert.asdevicearray(device, time_emb, implicit_host_transfer=False)

    def run_prefix():
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
        return hidden, skip

    def run_suffix(hidden: Any, skip: Any):
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

    def stitched_chain_no_fetch():
        hidden, skip = run_prefix()
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
        return run_suffix(hidden, skip)

    def fused_chain_no_fetch():
        hidden, skip = run_prefix()
        for mid in range(12):
            hidden = fused_midblocks[mid]["forward"](
                hidden,
                device_mask,
                device_attention_bias,
                device_time,
            )
        return run_suffix(hidden, skip)

    stitched_first = stitched_chain_no_fetch()
    fused_first = fused_chain_no_fetch()
    stitched_first_host = to_host_array(stitched_first)
    fused_vs_stitched = diff_summary(fused_first, stitched_first_host)

    expected_validation = None
    if args.expected_output is not None:
        expected = load_input_array(args.expected_output)
        expected_validation = {
            "stitched": diff_summary(stitched_first_host, expected),
            "fused": diff_summary(fused_first, expected),
        }

    rss_before = rss_mb()
    timings = {
        "stitched_no_fetch": time_call(stitched_chain_no_fetch, args.iterations, args.warmup),
        "stitched_fetch_final_output": time_call(
            lambda: to_host_array(stitched_chain_no_fetch()),
            args.iterations,
            args.warmup,
        ),
        "fused_no_fetch": time_call(fused_chain_no_fetch, args.iterations, args.warmup),
        "fused_fetch_final_output": time_call(
            lambda: to_host_array(fused_chain_no_fetch()),
            args.iterations,
            args.warmup,
        ),
    }
    rss_after = rss_mb()

    stitched_fetch_ms = timings["stitched_fetch_final_output"]["mean_ms"]
    fused_fetch_ms = timings["fused_fetch_final_output"]["mean_ms"]
    stitched_no_fetch_ms = timings["stitched_no_fetch"]["mean_ms"]
    fused_no_fetch_ms = timings["fused_no_fetch"]["mean_ms"]
    fetch_save_seconds_per_estimator = (stitched_fetch_ms - fused_fetch_ms) / 1000.0
    no_fetch_save_seconds_per_estimator = (stitched_no_fetch_ms - fused_no_fetch_ms) / 1000.0

    report = {
        "description": "Fixed-shape S3 estimator chain benchmark comparing stitched midblocks with fused split8 midblocks.",
        "frames": args.frames,
        "variant": args.variant,
        "input_source": input_source,
        "expected_output": args.expected_output.as_posix() if args.expected_output else None,
        "shape": {
            "x": list(x.shape),
            "mask": list(mask.shape),
            "attention_bias": list(attention_bias.shape),
            "time_emb": list(time_emb.shape),
        },
        "component_counts": {
            "stitched_distinct_vmfb_modules": len(module_names(args.frames)),
            "fused_midblock_vmfb_modules": 12,
            "helper_vmfb_modules": 3,
        },
        "module_load_seconds": module_load_seconds,
        "validation": {
            "fused_vs_stitched": fused_vs_stitched,
            "expected_output": expected_validation,
        },
        "timings": timings,
        "savings": {
            "fetch_output_seconds_per_estimator_call": fetch_save_seconds_per_estimator,
            "fetch_output_seconds_for_two_s3_estimator_calls": fetch_save_seconds_per_estimator * 2.0,
            "no_fetch_seconds_per_estimator_call": no_fetch_save_seconds_per_estimator,
            "no_fetch_seconds_for_two_s3_estimator_calls": no_fetch_save_seconds_per_estimator * 2.0,
        },
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "This is an estimator-chain benchmark only, not a full API request.",
            "Production credit requires integrating the fused chain into the S3 runtime and measuring a guarded API benchmark.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={args.output}")
    print(f"module_load_seconds={module_load_seconds:.3f}")
    print(
        "fused_vs_stitched_allclose_1e_4="
        f"{fused_vs_stitched['allclose_1e_4']} max_abs={fused_vs_stitched['max_abs_error']:.3e}"
    )
    print(f"stitched_fetch_ms={stitched_fetch_ms:.3f}")
    print(f"fused_fetch_ms={fused_fetch_ms:.3f}")
    print(f"fetch_save_per_estimator_ms={(stitched_fetch_ms - fused_fetch_ms):.3f}")
    print(f"fetch_save_two_s3_calls_seconds={fetch_save_seconds_per_estimator * 2.0:.3f}")
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
