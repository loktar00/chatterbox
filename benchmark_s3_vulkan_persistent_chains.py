#!/usr/bin/env python3
"""Persistent-process soak for S3 IREE Vulkan encoder/estimator chains."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import iree.runtime as ireert
import numpy as np

from benchmark_s3_encoder_iree_runtime_chain import (
    load_array as load_encoder_array,
    load_modules as load_encoder_modules,
)
from benchmark_s3_estimator_distinct_iree_runtime_chain import (
    load_array as load_estimator_array,
    load_modules as load_estimator_modules,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"


def to_host_array(value: Any) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--fetch-every", type=int, default=0)
    parser.add_argument("--gc-each-cycle", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_vulkan_persistent_chains_soak_2026-07-08.json",
    )
    args = parser.parse_args()

    device = ireert.get_device("vulkan")

    encoder_token_hidden = load_encoder_array("s3_encoder_embed_fmask_t605", "input_0.npy", np.float32)
    encoder_token_mask = load_encoder_array("s3_encoder_embed_fmask_t605", "input_1.npy", np.float32)
    encoder_up_mask = load_encoder_array("s3_encoder_up_embed_fmask_t1210", "input_1.npy", np.float32)
    encoder_lengths = load_encoder_array("s3_encoder_up_layer_t605", "input_1.npy", np.int64)

    estimator_x = load_estimator_array("s3_distinct_down_resnet0_t1210", "input_0.npy")
    estimator_mask = load_estimator_array("s3_distinct_down_resnet0_t1210", "input_1.npy")
    estimator_time = load_estimator_array("s3_distinct_down_resnet0_t1210", "input_2.npy")
    estimator_attention_bias = load_estimator_array("s3_distinct_down_transformer0_t1210", "input_1.npy")

    load_started = time.perf_counter()
    encoder_modules = load_encoder_modules()
    estimator_modules, estimator_helpers = load_estimator_modules()
    module_load_seconds = time.perf_counter() - load_started

    e_token_hidden = ireert.asdevicearray(device, encoder_token_hidden, implicit_host_transfer=False)
    e_token_mask = ireert.asdevicearray(device, encoder_token_mask, implicit_host_transfer=False)
    e_up_mask = ireert.asdevicearray(device, encoder_up_mask, implicit_host_transfer=False)
    e_lengths = ireert.asdevicearray(device, encoder_lengths, implicit_host_transfer=False)

    s_x = ireert.asdevicearray(device, estimator_x, implicit_host_transfer=False)
    s_mask = ireert.asdevicearray(device, estimator_mask, implicit_host_transfer=False)
    s_time = ireert.asdevicearray(device, estimator_time, implicit_host_transfer=False)
    s_attention_bias = ireert.asdevicearray(device, estimator_attention_bias, implicit_host_transfer=False)

    def encoder_chain():
        lower_hidden, lower_pos, lower_mask = encoder_modules["embed"]["forward"](
            e_token_hidden,
            e_token_mask,
        )
        hidden = encoder_modules["pre_lookahead"]["forward"](lower_hidden)
        for index in range(6):
            hidden = encoder_modules[f"lower_layer_{index}"]["forward"](
                hidden,
                lower_mask,
                lower_pos,
                lower_mask,
            )
        hidden_ct = encoder_modules["lower_transpose"]["forward"](hidden)
        up_hidden_ct = encoder_modules["up_layer"]["forward"](hidden_ct, e_lengths)
        up_hidden = encoder_modules["up_transpose"]["forward"](up_hidden_ct)
        upper_hidden, upper_pos, upper_mask = encoder_modules["up_embed"]["forward"](up_hidden, e_up_mask)
        hidden = upper_hidden
        for index in range(4):
            hidden = encoder_modules[f"upper_layer_{index}"]["forward"](
                hidden,
                upper_mask,
                upper_pos,
                upper_mask,
            )
        return encoder_modules["after_norm"]["forward"](hidden)

    def estimator_chain():
        hidden = estimator_modules["down_resnet"]["forward"](s_x, s_mask, s_time)
        hidden = estimator_helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = estimator_modules[f"down_transformer_{index}"]["forward"](
                hidden,
                s_attention_bias,
                s_time,
            )
        hidden = estimator_helpers["t_to_c"]["forward"](hidden)
        skip = hidden
        hidden = estimator_modules["downsample"]["forward"](hidden)
        for mid in range(12):
            hidden = estimator_modules[f"mid_resnet_{mid}"]["forward"](hidden, s_mask, s_time)
            hidden = estimator_helpers["c_to_t"]["forward"](hidden)
            for block in range(4):
                hidden = estimator_modules[f"mid_transformer_{mid}_{block}"]["forward"](
                    hidden,
                    s_attention_bias,
                    s_time,
                )
            hidden = estimator_helpers["t_to_c"]["forward"](hidden)
        hidden = estimator_helpers["cat"]["forward"](hidden, skip)
        hidden = estimator_modules["up_resnet"]["forward"](hidden, s_mask, s_time)
        hidden = estimator_helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = estimator_modules[f"up_transformer_{index}"]["forward"](
                hidden,
                s_attention_bias,
                s_time,
            )
        hidden = estimator_helpers["t_to_c"]["forward"](hidden)
        hidden = estimator_modules["upsample"]["forward"](hidden)
        hidden = estimator_modules["final_block"]["forward"](hidden, s_mask)
        return estimator_modules["final_proj"]["forward"](hidden)

    def run_cycle(fetch: bool) -> dict[str, Any]:
        started = time.perf_counter()
        encoder_output = encoder_chain()
        estimator_output_0 = estimator_chain()
        estimator_output_1 = estimator_chain()
        fetch_shapes = None
        if fetch:
            fetch_shapes = {
                "encoder": list(to_host_array(encoder_output).shape),
                "estimator_0": list(to_host_array(estimator_output_0).shape),
                "estimator_1": list(to_host_array(estimator_output_1).shape),
            }
        elapsed = time.perf_counter() - started
        return {
            "seconds": elapsed,
            "fetch": fetch,
            "fetch_shapes": fetch_shapes,
        }

    warmup_results = [run_cycle(fetch=False) for _ in range(args.warmup)]
    if args.gc_each_cycle:
        gc.collect()

    rss_start = rss_mb()
    cycle_results = []
    for index in range(args.cycles):
        fetch = args.fetch_every > 0 and ((index + 1) % args.fetch_every == 0)
        before = rss_mb()
        result = run_cycle(fetch=fetch)
        if args.gc_each_cycle:
            gc.collect()
        after = rss_mb()
        result.update(
            {
                "cycle": index + 1,
                "rss_before_mb": before,
                "rss_after_mb": after,
                "rss_delta_mb": after - before,
            }
        )
        cycle_results.append(result)
        print(
            f"cycle {index + 1}/{args.cycles}: {result['seconds']:.3f}s "
            f"rss={after:.1f}MB delta={after - before:.3f}MB fetch={fetch}"
        )
    rss_end = rss_mb()

    seconds = [item["seconds"] for item in cycle_results]
    rss_values = [item["rss_after_mb"] for item in cycle_results]
    report = {
        "description": "Persistent-process soak for fixed-shape S3 encoder + two full distinct S3 estimator IREE Vulkan chains.",
        "cycles": args.cycles,
        "warmup": args.warmup,
        "fetch_every": args.fetch_every,
        "gc_each_cycle": args.gc_each_cycle,
        "module_load_seconds": module_load_seconds,
        "warmup_results": warmup_results,
        "cycle_results": cycle_results,
        "summary": {
            "mean_cycle_seconds": float(np.mean(seconds)) if seconds else None,
            "min_cycle_seconds": float(np.min(seconds)) if seconds else None,
            "max_cycle_seconds": float(np.max(seconds)) if seconds else None,
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_end,
            "rss_delta_mb": rss_end - rss_start,
            "rss_min_after_cycle_mb": float(np.min(rss_values)) if rss_values else None,
            "rss_max_after_cycle_mb": float(np.max(rss_values)) if rss_values else None,
        },
        "notes": [
            "Each cycle runs one S3 encoder chain and two full distinct-weight estimator chains.",
            "This is a synthetic fixed-bucket persistent-process soak, not a full API request.",
            "The live CPU API remains untouched.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")
    print(
        f"mean_cycle_seconds={report['summary']['mean_cycle_seconds']:.3f} "
        f"rss_delta_mb={report['summary']['rss_delta_mb']:.3f}"
    )


if __name__ == "__main__":
    main()
