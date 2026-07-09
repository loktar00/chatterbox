#!/usr/bin/env python3
"""Validate and benchmark two chained T3 KV chunks through IREE Vulkan."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable

import iree.runtime as ireert
import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS
from export_t3_cached_stack_kv_probe import CachedManualStackKVWrapper


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
VARIANT = "split_matmul4_split_reduction"
CHUNKS = (
    ("s0", 0, "gpt2_cached_stack_kv_s0_l4_p128_t1"),
    ("s4", 4, "gpt2_cached_stack_kv_s4_l4_p128_t1"),
)
CACHE_UPDATE_VMFB = (
    EXPORT_DIR
    / "cache_update"
    / "t3_kv_cache_roll_append_p128_t1_vulkan_gfx1013.vmfb"
)


def vmfb_for(subgraph: str) -> Path:
    return EXPORT_DIR / "flag_variants" / subgraph / VARIANT / f"{subgraph}_{VARIANT}.vmfb"


def load_input(subgraph: str, index: int) -> np.ndarray:
    return np.load(EXPORT_DIR / f"{subgraph}_input_{index}.npy").astype(np.float32, copy=False)


def to_host_array(value) -> np.ndarray:
    if hasattr(value, "to_host"):
        return np.asarray(value.to_host())
    return np.asarray(value)


def compare(actual, expected: torch.Tensor | np.ndarray) -> dict:
    actual_np = to_host_array(actual)
    expected_np = expected.detach().cpu().numpy() if hasattr(expected, "detach") else np.asarray(expected)
    diff = np.abs(actual_np - expected_np)
    return {
        "shape": list(actual_np.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_exact": bool(np.array_equal(actual_np, expected_np)),
        "allclose_1e_6": bool(np.allclose(actual_np, expected_np, atol=1e-6, rtol=1e-6)),
        "allclose_1e_4": bool(np.allclose(actual_np, expected_np, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual_np, expected_np, atol=1e-3, rtol=1e-3)),
    }


def build_cpu_expected() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    wrappers = {
        start: CachedManualStackKVWrapper(
            [t3.tfmr.h[index].eval() for index in range(start, start + 4)]
        ).eval()
        for _, start, _ in CHUNKS
    }
    s0_inputs = tuple(torch.from_numpy(load_input(CHUNKS[0][2], index)) for index in range(9))
    s4_pasts = tuple(torch.from_numpy(load_input(CHUNKS[1][2], index)) for index in range(1, 9))

    with torch.inference_mode():
        s0_outputs = wrappers[0](*s0_inputs)
        s4_outputs = wrappers[4](s0_outputs[0], *s4_pasts)
    return s0_outputs, s4_outputs


def time_call(fn: Callable[[], object], iterations: int, warmup: int) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip-cpu-validation", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_DIR / "t3_two_chunk_iree_runtime_latest.json",
    )
    args = parser.parse_args()

    for _, _, subgraph in CHUNKS:
        vmfb = vmfb_for(subgraph)
        if not vmfb.exists():
            raise SystemExit(f"Missing VMFB: {vmfb}")

    load_started = time.perf_counter()
    modules = {
        name: ireert.load_vm_flatbuffer_file(vmfb_for(subgraph).as_posix(), driver="vulkan")
        for name, _, subgraph in CHUNKS
    }
    cache_update_module = None
    if CACHE_UPDATE_VMFB.exists():
        cache_update_module = ireert.load_vm_flatbuffer_file(CACHE_UPDATE_VMFB.as_posix(), driver="vulkan")
    module_load_seconds = time.perf_counter() - load_started

    host_inputs = {
        name: [load_input(subgraph, index) for index in range(9)]
        for name, _, subgraph in CHUNKS
    }
    device = ireert.get_device("vulkan")
    device_s0_inputs = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in host_inputs["s0"]
    ]
    device_s4_pasts = [
        ireert.asdevicearray(device, array, implicit_host_transfer=False)
        for array in host_inputs["s4"][1:]
    ]

    s0_outputs = modules["s0"]["forward"](*device_s0_inputs)
    s4_outputs = modules["s4"]["forward"](s0_outputs[0], *device_s4_pasts)

    validation = None
    if not args.skip_cpu_validation:
        expected_s0, expected_s4 = build_cpu_expected()
        validation = {
            "s0": [compare(actual, expected) for actual, expected in zip(s0_outputs, expected_s0)],
            "s4_chained": [
                compare(actual, expected) for actual, expected in zip(s4_outputs, expected_s4)
            ],
        }
        validation["s0_allclose_1e_4"] = all(item["allclose_1e_4"] for item in validation["s0"])
        validation["s4_chained_allclose_1e_4"] = all(
            item["allclose_1e_4"] for item in validation["s4_chained"]
        )

    def chain_device_no_fetch():
        out0 = modules["s0"]["forward"](*device_s0_inputs)
        out4 = modules["s4"]["forward"](out0[0], *device_s4_pasts)
        return out4

    def chain_device_fetch_final_hidden():
        out4 = chain_device_no_fetch()
        return to_host_array(out4[0])

    def chain_device_fetch_all_outputs():
        out0 = modules["s0"]["forward"](*device_s0_inputs)
        out4 = modules["s4"]["forward"](out0[0], *device_s4_pasts)
        return [to_host_array(output) for output in (*out0, *out4)]

    def update_chunk_cache(outputs, past_tensors):
        assert cache_update_module is not None
        updates = []
        for layer in range(4):
            updates.append(
                cache_update_module["forward"](
                    past_tensors[layer * 2],
                    past_tensors[layer * 2 + 1],
                    outputs[1 + layer * 2],
                    outputs[2 + layer * 2],
                )
            )
        return updates

    def chain_device_plus_cache_updates_no_fetch():
        out0 = modules["s0"]["forward"](*device_s0_inputs)
        out4 = modules["s4"]["forward"](out0[0], *device_s4_pasts)
        updates0 = update_chunk_cache(out0, device_s0_inputs[1:])
        updates4 = update_chunk_cache(out4, device_s4_pasts)
        return out4, updates0, updates4

    def chain_device_plus_cache_updates_fetch_updated_cache():
        _, updates0, updates4 = chain_device_plus_cache_updates_no_fetch()
        return [
            (to_host_array(key), to_host_array(value))
            for key, value in (*updates0, *updates4)
        ]

    cache_update_validation = None
    if cache_update_module is not None:
        updates0 = update_chunk_cache(s0_outputs, device_s0_inputs[1:])
        updates4 = update_chunk_cache(s4_outputs, device_s4_pasts)
        cache_update_validation = []
        for chunk_name, outputs, updates, past_arrays in (
            ("s0", s0_outputs, updates0, host_inputs["s0"][1:]),
            ("s4", s4_outputs, updates4, host_inputs["s4"][1:]),
        ):
            for layer in range(4):
                new_key = to_host_array(outputs[1 + layer * 2])
                new_value = to_host_array(outputs[2 + layer * 2])
                expected_key = np.concatenate((past_arrays[layer * 2][:, :, 1:, :], new_key), axis=2)
                expected_value = np.concatenate(
                    (past_arrays[layer * 2 + 1][:, :, 1:, :], new_value),
                    axis=2,
                )
                updated_key, updated_value = updates[layer]
                cache_update_validation.append(
                    {
                        "chunk": chunk_name,
                        "layer": layer,
                        "key": compare(updated_key, expected_key),
                        "value": compare(updated_value, expected_value),
                    }
                )

    rss_before = rss_mb()
    timings = {
        "device_chain_no_fetch": time_call(chain_device_no_fetch, args.iterations, args.warmup),
        "device_chain_fetch_final_hidden": time_call(
            chain_device_fetch_final_hidden,
            args.iterations,
            args.warmup,
        ),
        "device_chain_fetch_all_outputs": time_call(
            chain_device_fetch_all_outputs,
            args.iterations,
            args.warmup,
        ),
    }
    if cache_update_module is not None:
        timings["device_chain_plus_eight_cache_updates_no_fetch"] = time_call(
            chain_device_plus_cache_updates_no_fetch,
            args.iterations,
            args.warmup,
        )
        timings["device_chain_plus_eight_cache_updates_fetch_updated_cache"] = time_call(
            chain_device_plus_cache_updates_fetch_updated_cache,
            args.iterations,
            args.warmup,
        )
    rss_after = rss_mb()

    report = {
        "chunks": [
            {
                "name": name,
                "start_layer": start,
                "subgraph": subgraph,
                "vmfb": vmfb_for(subgraph).as_posix(),
                "vmfb_size_bytes": vmfb_for(subgraph).stat().st_size,
            }
            for name, start, subgraph in CHUNKS
        ],
        "module_load_seconds": module_load_seconds,
        "cache_update_vmfb": CACHE_UPDATE_VMFB.as_posix() if CACHE_UPDATE_VMFB.exists() else None,
        "validation": validation,
        "cache_update_validation": cache_update_validation,
        "cache_update_allclose_1e_6": None
        if cache_update_validation is None
        else all(
            item["key"]["allclose_1e_6"] and item["value"]["allclose_1e_6"]
            for item in cache_update_validation
        ),
        "cache_update_exact": None
        if cache_update_validation is None
        else all(
            item["key"]["allclose_exact"] and item["value"]["allclose_exact"]
            for item in cache_update_validation
        ),
        "timings": timings,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "The hidden state output from layers 0-3 is passed directly as a DeviceArray input to layers 4-7.",
            "When the cache-update VMFB exists, this also rolls/appends K/V cache tensors on device for all eight layers covered by these two chunks.",
            "The cache update is a fixed-size rolling-window operation, not exact growing-cache generation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"module_load_seconds={module_load_seconds:.3f}")
    if validation is not None:
        print(f"s0_allclose_1e_4={validation['s0_allclose_1e_4']}")
        print(f"s4_chained_allclose_1e_4={validation['s4_chained_allclose_1e_4']}")
    if cache_update_validation is not None:
        print(
            "cache_update_allclose_1e_6="
            f"{all(item['key']['allclose_1e_6'] and item['value']['allclose_1e_6'] for item in cache_update_validation)}"
        )
        print(
            "cache_update_exact="
            f"{all(item['key']['allclose_exact'] and item['value']['allclose_exact'] for item in cache_update_validation)}"
        )
    for name, timing in timings.items():
        print(f"{name}: {timing['mean_ms']:.3f} ms ({timing['items_per_second']:.2f}/s)")
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")


if __name__ == "__main__":
    main()
