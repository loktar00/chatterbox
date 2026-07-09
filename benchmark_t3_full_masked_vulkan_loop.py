#!/usr/bin/env python3
"""Run the full 24-layer masked-cache T3 body with Vulkan chunks."""

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

from benchmark_t3_masked_chunk_logits_loop import logits_summary
from probe_t3_kv_cache_slot_update_vulkan import slot_mask
from probe_t3_masked_cache_block_vulkan import compare_arrays, make_attention_bias, to_host_array
from probe_t3_masked_cache_stack_kv_vulkan import DynamicCacheStackKVWrapper

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports" / "t3_exportability"
MASKED_DIR = EXPORT_DIR / "masked_cache"
CACHE_UPDATE_VMFB = EXPORT_DIR / "cache_update" / "t3_kv_cache_slot_update_p128_t1_vulkan_gfx1013.vmfb"
SPEECH_HEAD_VMFB = EXPORT_DIR / "speech_head_t1_vulkan_gfx1013.vmfb"
FINAL_NORM_SPEECH_HEAD_VMFB = EXPORT_DIR / "t3_final_norm_speech_head_t1_vulkan_gfx1013.vmfb"
CHUNK_STARTS = (0, 4, 8, 12, 16, 20)
DEFAULT_MAX_LEN = 128
DEFAULT_COMPILED_VALID_LEN = 42


def cache_update_vmfb(max_len: int) -> Path:
    return EXPORT_DIR / "cache_update" / f"t3_kv_cache_slot_update_p{max_len}_t1_vulkan_gfx1013.vmfb"


def chunk_name(start_layer: int, max_len: int, compiled_valid_len: int) -> str:
    return f"gpt2_masked_cache_stack_s{start_layer}_l4_p{max_len}_valid{compiled_valid_len}_t1"


def chunk_vmfb(start_layer: int, max_len: int, compiled_valid_len: int) -> Path:
    return MASKED_DIR / f"{chunk_name(start_layer, max_len, compiled_valid_len)}_vulkan_gfx1013.vmfb"


def chunk_input(start_layer: int, index: int, max_len: int, compiled_valid_len: int) -> np.ndarray:
    return np.load(MASKED_DIR / f"{chunk_name(start_layer, max_len, compiled_valid_len)}_input_{index}.npy").astype(
        np.float32,
        copy=False,
    )


def zero_cache_like(start_layer: int, index: int, max_len: int, compiled_valid_len: int) -> np.ndarray:
    return np.zeros_like(chunk_input(start_layer, index, max_len, compiled_valid_len), dtype=np.float32)


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


def make_hidden_sequence(steps: int, max_len: int, compiled_valid_len: int) -> list[np.ndarray]:
    base = chunk_input(0, 0, max_len, compiled_valid_len)
    sequence = [base]
    rng = np.random.default_rng(1234)
    for _ in range(steps - 1):
        sequence.append(rng.standard_normal(base.shape, dtype=np.float32))
    return sequence


def build_cpu_reference(hidden_sequence: list[np.ndarray]) -> dict:
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

    dynamic = DynamicCacheStackKVWrapper([t3.tfmr.h[index].eval() for index in range(24)]).eval()
    empty_key = torch.zeros(1, 16, 0, 64, dtype=torch.float32)
    empty_value = torch.zeros(1, 16, 0, 64, dtype=torch.float32)
    past: list[torch.Tensor] = []
    for _ in range(24):
        past.extend((empty_key.clone(), empty_value.clone()))

    steps = []
    with torch.inference_mode():
        for step, hidden_np in enumerate(hidden_sequence):
            outputs = dynamic(torch.from_numpy(hidden_np), *past)
            logits = t3.speech_head(t3.tfmr.ln_f(outputs[0]))
            steps.append(
                {
                    "step": step,
                    "valid_len": step,
                    "hidden": outputs[0].detach().cpu().numpy(),
                    "logits": logits.detach().cpu().numpy(),
                }
            )
            for layer in range(24):
                past[layer * 2] = torch.cat((past[layer * 2], outputs[1 + layer * 2]), dim=2)
                past[layer * 2 + 1] = torch.cat(
                    (past[layer * 2 + 1], outputs[2 + layer * 2]),
                    dim=2,
                )

    return {
        "steps": steps,
        "final_past": [tensor.detach().cpu().numpy() for tensor in past],
    }


class VulkanFullMaskedT3Loop:
    def __init__(
        self,
        max_len: int = DEFAULT_MAX_LEN,
        compiled_valid_len: int = DEFAULT_COMPILED_VALID_LEN,
    ) -> None:
        missing = [
            path
            for path in [
                *(chunk_vmfb(start, max_len, compiled_valid_len) for start in CHUNK_STARTS),
                cache_update_vmfb(max_len),
                FINAL_NORM_SPEECH_HEAD_VMFB,
            ]
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(", ".join(path.as_posix() for path in missing))

        self.max_len = max_len
        self.compiled_valid_len = compiled_valid_len
        self.device = ireert.get_device("vulkan")
        self.chunks = {
            start: ireert.load_vm_flatbuffer_file(
                chunk_vmfb(start, max_len, compiled_valid_len).as_posix(),
                driver="vulkan",
            )
            for start in CHUNK_STARTS
        }
        self.cache_update = ireert.load_vm_flatbuffer_file(cache_update_vmfb(max_len).as_posix(), driver="vulkan")
        self.final_norm_speech_head = ireert.load_vm_flatbuffer_file(
            FINAL_NORM_SPEECH_HEAD_VMFB.as_posix(),
            driver="vulkan",
        )
        self.biases = [
            self.as_device(make_attention_bias(max_len, valid_len).numpy())
            for valid_len in range(max_len)
        ]
        self.masks = [self.as_device(slot_mask(max_len, valid_len)) for valid_len in range(max_len)]
        self.reset_cache()

    def as_device(self, array: np.ndarray):
        return ireert.asdevicearray(self.device, array, implicit_host_transfer=False)

    def reset_cache(self) -> None:
        self.cache = []
        for start in CHUNK_STARTS:
            for local_layer in range(4):
                self.cache.append(
                    self.as_device(
                        zero_cache_like(start, 2 + local_layer * 2, self.max_len, self.compiled_valid_len)
                    )
                )
                self.cache.append(
                    self.as_device(
                        zero_cache_like(start, 3 + local_layer * 2, self.max_len, self.compiled_valid_len)
                    )
                )

    def step(self, hidden_np: np.ndarray, valid_len: int, update_cache: bool = True):
        hidden = self.as_device(hidden_np)
        bias = self.biases[valid_len]
        mask = self.masks[valid_len]

        for chunk_index, start in enumerate(CHUNK_STARTS):
            cache_offset = chunk_index * 8
            chunk_cache = self.cache[cache_offset : cache_offset + 8]
            outputs = self.chunks[start]["forward"](hidden, bias, *chunk_cache)
            hidden = outputs[0]
            if update_cache:
                for local_layer in range(4):
                    layer = chunk_index * 4 + local_layer
                    updated_key, updated_value = self.cache_update["forward"](
                        self.cache[layer * 2],
                        self.cache[layer * 2 + 1],
                        outputs[1 + local_layer * 2],
                        outputs[2 + local_layer * 2],
                        mask,
                    )
                    self.cache[layer * 2] = updated_key
                    self.cache[layer * 2 + 1] = updated_value

        logits = self.final_norm_speech_head["forward"](hidden)
        return hidden, logits

    def run_sequence(self, hidden_sequence: list[np.ndarray], fetch_logits: bool = False):
        self.reset_cache()
        results = []
        for step, hidden_np in enumerate(hidden_sequence):
            hidden, logits = self.step(hidden_np, valid_len=step, update_cache=True)
            if fetch_logits:
                results.append((to_host_array(hidden), to_host_array(logits)))
        return results


def padded_reference_cache(cpu_cache: np.ndarray, max_len: int) -> np.ndarray:
    padded = np.zeros((cpu_cache.shape[0], cpu_cache.shape[1], max_len, cpu_cache.shape[3]), dtype=np.float32)
    padded[:, :, : cpu_cache.shape[2], :] = cpu_cache
    return padded


def validate_loop(
    hidden_sequence: list[np.ndarray],
    reference: dict,
    compare_caches: bool,
    max_len: int,
    compiled_valid_len: int,
) -> dict:
    loop = VulkanFullMaskedT3Loop(max_len=max_len, compiled_valid_len=compiled_valid_len)
    comparisons = []
    for step, hidden_np in enumerate(hidden_sequence):
        hidden, logits = loop.step(hidden_np, valid_len=step, update_cache=True)
        expected = reference["steps"][step]
        comparisons.append(
            {
                "step": step,
                "valid_len": step,
                "hidden": compare_arrays(to_host_array(hidden), expected["hidden"]),
                "logits": logits_summary(to_host_array(logits), expected["logits"]),
            }
        )

    cache_comparisons = []
    if compare_caches:
        for index, cache in enumerate(loop.cache):
            cache_comparisons.append(
                {
                    "index": index,
                    "layer": index // 2,
                    "kind": "key" if index % 2 == 0 else "value",
                    "comparison": compare_arrays(
                        to_host_array(cache),
                        padded_reference_cache(reference["final_past"][index], max_len),
                    ),
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
        "cache_compared": compare_caches,
        "cache_allclose_1e_4": (
            all(item["comparison"]["allclose_1e_4"] for item in cache_comparisons)
            if cache_comparisons
            else None
        ),
        "max_cache_abs_error": (
            max(item["comparison"]["max_abs_error"] for item in cache_comparisons)
            if cache_comparisons
            else None
        ),
        "cache_comparisons": cache_comparisons,
    }


def benchmark_loop(
    hidden_sequence: list[np.ndarray],
    iterations: int,
    warmup: int,
    max_len: int,
    compiled_valid_len: int,
) -> dict:
    loop = VulkanFullMaskedT3Loop(max_len=max_len, compiled_valid_len=compiled_valid_len)

    def run_no_fetch():
        return loop.run_sequence(hidden_sequence, fetch_logits=False)

    def run_fetch_logits():
        return loop.run_sequence(hidden_sequence, fetch_logits=True)

    rss_before = rss_mb()
    timings = {
        "full_t3_logits_no_fetch": time_call(run_no_fetch, iterations, warmup),
        "full_t3_logits_fetch": time_call(run_fetch_logits, iterations, warmup),
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
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--compiled-valid-len", type=int, default=DEFAULT_COMPILED_VALID_LEN)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--skip-cache-compare", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=MASKED_DIR / "t3_full_masked_vulkan_loop_latest.json",
    )
    args = parser.parse_args()
    if args.steps > args.max_len:
        raise SystemExit(f"--steps cannot exceed the compiled cache length {args.max_len}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""

    started = time.perf_counter()
    hidden_sequence = make_hidden_sequence(args.steps, args.max_len, args.compiled_valid_len)
    reference = build_cpu_reference(hidden_sequence)
    validation = validate_loop(
        hidden_sequence,
        reference,
        compare_caches=not args.skip_cache_compare,
        max_len=args.max_len,
        compiled_valid_len=args.compiled_valid_len,
    )
    benchmark = benchmark_loop(
        hidden_sequence,
        iterations=args.iterations,
        warmup=args.warmup,
        max_len=args.max_len,
        compiled_valid_len=args.compiled_valid_len,
    )

    report = {
        "seconds": time.perf_counter() - started,
        "steps": args.steps,
        "max_len": args.max_len,
        "compiled_valid_len": args.compiled_valid_len,
        "chunk_starts": list(CHUNK_STARTS),
        "chunk_vmfbs": [
            chunk_vmfb(start, args.max_len, args.compiled_valid_len).as_posix()
            for start in CHUNK_STARTS
        ],
        "cache_update_vmfb": cache_update_vmfb(args.max_len).as_posix(),
        "final_norm_speech_head_vmfb": FINAL_NORM_SPEECH_HEAD_VMFB.as_posix(),
        "validation": validation,
        "benchmark": benchmark,
        "notes": [
            "This runs all six 4-layer masked-cache T3 Vulkan chunks as a full 24-layer transformer body.",
            "The test uses synthetic hidden states; embeddings, sampler control flow, and real prompt conditioning are still outside this loop.",
            "All 24 layer K/V caches are updated through the fixed-slot Vulkan cache updater.",
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
    print(f"max_hidden_abs_error={validation['max_hidden_abs_error']:.3e}")
    print(f"max_logits_abs_error={validation['max_logits_abs_error']:.3e}")
    if validation["cache_compared"]:
        print(f"cache_allclose_1e_4={validation['cache_allclose_1e_4']}")
        print(f"max_cache_abs_error={validation['max_cache_abs_error']:.3e}")
    for name, timing in benchmark["timings"].items():
        print(
            f"{name}: {timing['mean_ms']:.3f} ms/run "
            f"({timing['mean_ms_per_step']:.3f} ms/step)"
        )
    print(f"rss_delta_mb={benchmark['rss_mb']['delta']:.3f}")


if __name__ == "__main__":
    main()
