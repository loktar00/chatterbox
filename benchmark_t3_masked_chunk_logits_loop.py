#!/usr/bin/env python3
"""Run a small masked-cache T3 chunk loop through the Vulkan speech head."""

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
from benchmark_t3_masked_chunk_cache_loop import (
    CACHE_UPDATE_VMFB,
    MASKED_STACK_VMFB,
    VulkanMaskedChunkLoop,
    make_hidden_sequence,
)
from probe_t3_masked_cache_block_vulkan import compare_arrays, to_host_array
from probe_t3_masked_cache_stack_kv_vulkan import DynamicCacheStackKVWrapper


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
SPEECH_HEAD_VMFB = EXPORT_DIR / "speech_head_t1_vulkan_gfx1013.vmfb"


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


def logits_summary(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(actual - expected)
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    actual_argmax = int(np.argmax(actual_flat))
    expected_argmax = int(np.argmax(expected_flat))
    actual_top5 = set(np.argsort(actual_flat)[-5:].tolist())
    expected_top5 = set(np.argsort(expected_flat)[-5:].tolist())
    return {
        **compare_arrays(actual, expected),
        "argmax_match": actual_argmax == expected_argmax,
        "actual_argmax": actual_argmax,
        "expected_argmax": expected_argmax,
        "actual_argmax_logit": float(actual_flat[actual_argmax]),
        "expected_argmax_logit": float(expected_flat[expected_argmax]),
        "top5_overlap": len(actual_top5 & expected_top5),
        "max_abs_error_index": int(np.argmax(diff.reshape(-1))),
    }


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
            logits = t3.speech_head(outputs[0])
            references.append(
                {
                    "step": step,
                    "valid_len": step,
                    "hidden": outputs[0].detach().cpu().numpy(),
                    "logits": logits.detach().cpu().numpy(),
                }
            )
            for layer in range(4):
                past[layer * 2] = torch.cat((past[layer * 2], outputs[1 + layer * 2]), dim=2)
                past[layer * 2 + 1] = torch.cat(
                    (past[layer * 2 + 1], outputs[2 + layer * 2]),
                    dim=2,
                )
    return references


class VulkanMaskedChunkLogitsLoop(VulkanMaskedChunkLoop):
    def __init__(self, max_len: int = 128) -> None:
        super().__init__(max_len=max_len)
        if not SPEECH_HEAD_VMFB.exists():
            raise FileNotFoundError(SPEECH_HEAD_VMFB)
        self.speech_head = ireert.load_vm_flatbuffer_file(SPEECH_HEAD_VMFB.as_posix(), driver="vulkan")

    def step_logits(self, hidden_np: np.ndarray, valid_len: int, update_cache: bool = True):
        outputs = self.step(hidden_np, valid_len=valid_len, update_cache=update_cache)
        logits = self.speech_head["forward"](outputs[0])
        return outputs, logits

    def run_sequence_logits(self, hidden_sequence: list[np.ndarray], fetch_logits: bool = False):
        self.reset_cache()
        results = []
        for step, hidden_np in enumerate(hidden_sequence):
            outputs, logits = self.step_logits(hidden_np, valid_len=step, update_cache=True)
            if fetch_logits:
                results.append((to_host_array(outputs[0]), to_host_array(logits)))
        return results


def validate_logits_loop(hidden_sequence: list[np.ndarray], references: list[dict]) -> dict:
    loop = VulkanMaskedChunkLogitsLoop(max_len=128)
    comparisons = []
    for step, hidden_np in enumerate(hidden_sequence):
        outputs, logits = loop.step_logits(hidden_np, valid_len=step, update_cache=True)
        hidden_compare = compare_arrays(to_host_array(outputs[0]), references[step]["hidden"])
        logits_compare = logits_summary(to_host_array(logits), references[step]["logits"])
        comparisons.append(
            {
                "step": step,
                "valid_len": step,
                "hidden": hidden_compare,
                "logits": logits_compare,
            }
        )
    return {
        "steps": len(hidden_sequence),
        "comparisons": comparisons,
        "hidden_allclose_1e_4": all(item["hidden"]["allclose_1e_4"] for item in comparisons),
        "logits_allclose_1e_4": all(item["logits"]["allclose_1e_4"] for item in comparisons),
        "logits_allclose_1e_3": all(item["logits"]["allclose_1e_3"] for item in comparisons),
        "argmax_matches": sum(1 for item in comparisons if item["logits"]["argmax_match"]),
        "argmax_total": len(comparisons),
        "min_top5_overlap": min(item["logits"]["top5_overlap"] for item in comparisons),
        "max_hidden_abs_error": max(item["hidden"]["max_abs_error"] for item in comparisons),
        "max_logits_abs_error": max(item["logits"]["max_abs_error"] for item in comparisons),
    }


def benchmark_logits_loop(hidden_sequence: list[np.ndarray], iterations: int, warmup: int) -> dict:
    loop = VulkanMaskedChunkLogitsLoop(max_len=128)

    def run_no_fetch():
        return loop.run_sequence_logits(hidden_sequence, fetch_logits=False)

    def run_fetch_logits():
        return loop.run_sequence_logits(hidden_sequence, fetch_logits=True)

    rss_before = rss_mb()
    timings = {
        "loop_logits_no_fetch": time_call(run_no_fetch, iterations, warmup),
        "loop_logits_fetch": time_call(run_fetch_logits, iterations, warmup),
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
        default=EXPORT_DIR / "masked_cache" / "t3_masked_chunk_logits_loop_latest.json",
    )
    args = parser.parse_args()
    if args.steps > 128:
        raise SystemExit("--steps cannot exceed the compiled cache length 128")
    if not MASKED_STACK_VMFB.exists() or not CACHE_UPDATE_VMFB.exists() or not SPEECH_HEAD_VMFB.exists():
        raise SystemExit("Required VMFB is missing")

    started = time.perf_counter()
    hidden_sequence = make_hidden_sequence(args.steps)
    references = build_cpu_reference(hidden_sequence)
    validation = validate_logits_loop(hidden_sequence, references)
    benchmark = benchmark_logits_loop(hidden_sequence, iterations=args.iterations, warmup=args.warmup)

    report = {
        "seconds": time.perf_counter() - started,
        "steps": args.steps,
        "masked_stack_vmfb": MASKED_STACK_VMFB.as_posix(),
        "cache_update_vmfb": CACHE_UPDATE_VMFB.as_posix(),
        "speech_head_vmfb": SPEECH_HEAD_VMFB.as_posix(),
        "validation": validation,
        "benchmark": benchmark,
        "notes": [
            "This adds the existing Vulkan speech head to the one-chunk masked-cache runtime loop.",
            "The loop covers one 4-layer T3 chunk plus speech logits, not the full 24-layer T3 model.",
            "Hidden inputs are synthetic; this validates logits drift and argmax stability for the assembled sub-runtime.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"steps={args.steps}")
    print(f"hidden_allclose_1e_4={validation['hidden_allclose_1e_4']}")
    print(f"logits_allclose_1e_4={validation['logits_allclose_1e_4']}")
    print(f"logits_allclose_1e_3={validation['logits_allclose_1e_3']}")
    print(f"argmax_matches={validation['argmax_matches']}/{validation['argmax_total']}")
    print(f"min_top5_overlap={validation['min_top5_overlap']}")
    print(f"max_logits_abs_error={validation['max_logits_abs_error']:.3e}")
    for name, timing in benchmark["timings"].items():
        print(
            f"{name}: {timing['mean_ms']:.3f} ms/run "
            f"({timing['mean_ms_per_step']:.3f} ms/step)"
        )
    print(f"rss_delta_mb={benchmark['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
