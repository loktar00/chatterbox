#!/usr/bin/env python3
"""Run a small masked-cache T3 chunk loop with on-device slot updates."""

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
from probe_t3_kv_cache_slot_update_vulkan import slot_mask
from probe_t3_masked_cache_block_vulkan import compare_arrays, make_attention_bias, to_host_array
from probe_t3_masked_cache_stack_kv_vulkan import DynamicCacheStackKVWrapper


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
MASKED_DIR = EXPORT_DIR / "masked_cache"
CACHE_UPDATE_DIR = EXPORT_DIR / "cache_update"
MASKED_STACK_SUBGRAPH = "gpt2_masked_cache_stack_s0_l4_p128_valid42_t1"
MASKED_STACK_VMFB = MASKED_DIR / f"{MASKED_STACK_SUBGRAPH}_vulkan_gfx1013.vmfb"
CACHE_UPDATE_VMFB = CACHE_UPDATE_DIR / "t3_kv_cache_slot_update_p128_t1_vulkan_gfx1013.vmfb"


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


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


def load_masked_input(index: int) -> np.ndarray:
    return np.load(MASKED_DIR / f"{MASKED_STACK_SUBGRAPH}_input_{index}.npy").astype(
        np.float32,
        copy=False,
    )


def make_hidden_sequence(steps: int) -> list[np.ndarray]:
    base = load_masked_input(0)
    sequence = [base]
    rng = np.random.default_rng(1234)
    for _ in range(steps - 1):
        sequence.append(rng.standard_normal(base.shape, dtype=np.float32))
    return sequence


def zero_cache_like(index: int) -> np.ndarray:
    return np.zeros_like(load_masked_input(index), dtype=np.float32)


def build_cpu_reference(hidden_sequence: list[np.ndarray]) -> list[dict]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"
    dynamic = DynamicCacheStackKVWrapper([t3.tfmr.h[index].eval() for index in range(4)]).eval()

    empty_key = torch.zeros(1, 16, 0, 64, dtype=torch.float32)
    empty_value = torch.zeros(1, 16, 0, 64, dtype=torch.float32)
    past: list[torch.Tensor] = []
    for _ in range(4):
        past.extend((empty_key.clone(), empty_value.clone()))

    references = []
    with torch.inference_mode():
        for step, hidden_np in enumerate(hidden_sequence):
            outputs = dynamic(torch.from_numpy(hidden_np), *past)
            references.append(
                {
                    "step": step,
                    "valid_len": step,
                    "outputs": [output.detach().cpu().numpy() for output in outputs],
                }
            )
            for layer in range(4):
                past[layer * 2] = torch.cat((past[layer * 2], outputs[1 + layer * 2]), dim=2)
                past[layer * 2 + 1] = torch.cat(
                    (past[layer * 2 + 1], outputs[2 + layer * 2]),
                    dim=2,
                )
    return references


class VulkanMaskedChunkLoop:
    def __init__(self, max_len: int = 128) -> None:
        if not MASKED_STACK_VMFB.exists():
            raise FileNotFoundError(MASKED_STACK_VMFB)
        if not CACHE_UPDATE_VMFB.exists():
            raise FileNotFoundError(CACHE_UPDATE_VMFB)

        self.max_len = max_len
        self.stack = ireert.load_vm_flatbuffer_file(MASKED_STACK_VMFB.as_posix(), driver="vulkan")
        self.cache_update = ireert.load_vm_flatbuffer_file(CACHE_UPDATE_VMFB.as_posix(), driver="vulkan")
        self.device = ireert.get_device("vulkan")
        self.reset_cache()

    def as_device(self, array: np.ndarray):
        return ireert.asdevicearray(self.device, array, implicit_host_transfer=False)

    def reset_cache(self) -> None:
        self.cache = []
        for layer in range(4):
            self.cache.append(self.as_device(zero_cache_like(2 + layer * 2)))
            self.cache.append(self.as_device(zero_cache_like(3 + layer * 2)))

    def step(self, hidden_np: np.ndarray, valid_len: int, update_cache: bool = True):
        hidden = self.as_device(hidden_np)
        bias = self.as_device(make_attention_bias(self.max_len, valid_len).numpy())
        outputs = self.stack["forward"](hidden, bias, *self.cache)
        if update_cache:
            mask = self.as_device(slot_mask(self.max_len, valid_len))
            updated = []
            for layer in range(4):
                updated_key, updated_value = self.cache_update["forward"](
                    self.cache[layer * 2],
                    self.cache[layer * 2 + 1],
                    outputs[1 + layer * 2],
                    outputs[2 + layer * 2],
                    mask,
                )
                updated.extend((updated_key, updated_value))
            self.cache = updated
        return outputs

    def run_sequence(self, hidden_sequence: list[np.ndarray], fetch_outputs: bool = False):
        self.reset_cache()
        results = []
        for step, hidden_np in enumerate(hidden_sequence):
            outputs = self.step(hidden_np, valid_len=step, update_cache=True)
            if fetch_outputs:
                results.append([to_host_array(output) for output in outputs])
        return results


def validate_loop(hidden_sequence: list[np.ndarray], references: list[dict]) -> dict:
    loop = VulkanMaskedChunkLoop(max_len=128)
    comparisons = []
    for step, hidden_np in enumerate(hidden_sequence):
        outputs = loop.step(hidden_np, valid_len=step, update_cache=True)
        step_comparisons = [
            compare_arrays(to_host_array(output), references[step]["outputs"][index])
            for index, output in enumerate(outputs)
        ]
        comparisons.append(
            {
                "step": step,
                "valid_len": step,
                "comparisons": step_comparisons,
                "max_abs_error": max(item["max_abs_error"] for item in step_comparisons),
                "hidden_max_abs_error": step_comparisons[0]["max_abs_error"],
                "hidden_mean_abs_error": step_comparisons[0]["mean_abs_error"],
                "hidden_p95_abs_error": step_comparisons[0]["p95_abs_error"],
                "allclose_1e_4": all(item["allclose_1e_4"] for item in step_comparisons),
                "allclose_1e_3": all(item["allclose_1e_3"] for item in step_comparisons),
            }
        )
    return {
        "steps": len(hidden_sequence),
        "comparisons": comparisons,
        "all_steps_allclose_1e_4": all(item["allclose_1e_4"] for item in comparisons),
        "all_steps_allclose_1e_3": all(item["allclose_1e_3"] for item in comparisons),
        "max_abs_error": max(item["max_abs_error"] for item in comparisons),
        "max_hidden_abs_error": max(item["hidden_max_abs_error"] for item in comparisons),
    }


def benchmark_loop(hidden_sequence: list[np.ndarray], iterations: int, warmup: int) -> dict:
    loop = VulkanMaskedChunkLoop(max_len=128)

    def run_no_fetch():
        return loop.run_sequence(hidden_sequence, fetch_outputs=False)

    def run_fetch_outputs():
        return loop.run_sequence(hidden_sequence, fetch_outputs=True)

    rss_before = rss_mb()
    timings = {
        "loop_no_fetch": time_call(run_no_fetch, iterations, warmup),
        "loop_fetch_outputs": time_call(run_fetch_outputs, iterations, warmup),
    }
    rss_after = rss_mb()
    for timing in timings.values():
        timing["mean_ms_per_step"] = timing["mean_ms"] / len(hidden_sequence)
    return {
        "timings": timings,
        "rss_mb": {
            "before_timing": rss_before,
            "after_timing": rss_after,
            "delta": rss_after - rss_before,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORT_DIR / "masked_cache" / "t3_masked_chunk_cache_loop_latest.json",
    )
    args = parser.parse_args()
    if args.steps > 128:
        raise SystemExit("--steps cannot exceed the compiled cache length 128")

    started = time.perf_counter()
    hidden_sequence = make_hidden_sequence(args.steps)
    references = build_cpu_reference(hidden_sequence)
    validation = validate_loop(hidden_sequence, references)
    benchmark = benchmark_loop(hidden_sequence, iterations=args.iterations, warmup=args.warmup)

    report = {
        "seconds": time.perf_counter() - started,
        "steps": args.steps,
        "masked_stack_vmfb": MASKED_STACK_VMFB.as_posix(),
        "cache_update_vmfb": CACHE_UPDATE_VMFB.as_posix(),
        "validation": validation,
        "benchmark": benchmark,
        "notes": [
            "This runs one 4-layer masked T3 chunk over several token steps.",
            "The preallocated K/V cache is updated in-place logically by passing DeviceArray outputs from the fixed-slot cache updater into the next step.",
            "Hidden inputs are synthetic; this validates chunk/cache runtime mechanics, not full T3 sampling behavior.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"steps={args.steps}")
    print(f"all_steps_allclose_1e_4={validation['all_steps_allclose_1e_4']}")
    print(f"all_steps_allclose_1e_3={validation['all_steps_allclose_1e_3']}")
    print(f"max_abs_error={validation['max_abs_error']:.3e}")
    print(f"max_hidden_abs_error={validation['max_hidden_abs_error']:.3e}")
    for name, timing in benchmark["timings"].items():
        print(
            f"{name}: {timing['mean_ms']:.3f} ms/run "
            f"({timing['mean_ms_per_step']:.3f} ms/step)"
        )
    print(f"rss_delta_mb={benchmark['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
