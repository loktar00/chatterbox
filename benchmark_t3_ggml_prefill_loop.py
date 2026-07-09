#!/usr/bin/env python3
"""Benchmark T3 prefill through the existing one-token ggml/Vulkan bridge."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def diff_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    diff = np.abs(actual - expected)
    actual_top10 = np.argsort(actual)[-10:][::-1].astype(int).tolist()
    expected_top10 = np.argsort(expected)[-10:][::-1].astype(int).tolist()
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "argmax_actual": int(actual.argmax()),
        "argmax_expected": int(expected.argmax()),
        "argmax_matches": bool(int(actual.argmax()) == int(expected.argmax())),
        "top10_actual": actual_top10,
        "top10_expected": expected_top10,
        "top10_overlap": len(set(actual_top10).intersection(expected_top10)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
        "allclose_1e_2": bool(np.allclose(actual, expected, atol=1e-2, rtol=1e-2)),
    }


def zero_runtime_cache(runtime: T3GGMLVulkanRuntime) -> float:
    started = time.perf_counter()
    key_cache = np.zeros((runtime.heads, runtime.max_len, runtime.head_dim), dtype=np.float32)
    value_cache = np.zeros((runtime.heads, runtime.max_len, runtime.head_dim), dtype=np.float32)
    for layer in range(runtime.layers):
        runtime.set_layer_cache_arrays(layer, key_cache, value_cache)
    return time.perf_counter() - started


@torch.inference_mode()
def build_prefill_inputs(model: ChatterboxTurboTTS, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    t3 = model.t3
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    return embeds, speech_start_token


@torch.inference_mode()
def torch_prefill_logits(model: ChatterboxTurboTTS, embeds: torch.Tensor) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    outputs = model.t3.tfmr(inputs_embeds=embeds, use_cache=True)
    logits = model.t3.speech_head(outputs[0][:, -1:])[:, -1, :]
    elapsed = time.perf_counter() - started
    return logits.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False), elapsed


@torch.inference_mode()
def ggml_prefill_logits(
    model: ChatterboxTurboTTS,
    runtime: T3GGMLVulkanRuntime,
    embeds: torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    t3 = model.t3
    context_len = int(embeds.shape[1])
    if context_len >= runtime.max_len:
        raise ValueError(f"context length {context_len} exceeds runtime max_len {runtime.max_len}")

    started = time.perf_counter()
    position_embeddings = t3.tfmr.wpe.weight[:context_len].detach().to(device="cpu", dtype=torch.float32).numpy()
    embed_np = embeds[0].detach().to(device="cpu", dtype=torch.float32).numpy()
    hidden_np = np.ascontiguousarray(embed_np + position_embeddings, dtype=np.float32)
    masks = runtime.masks_for_valid_lens(0, context_len)
    setup_seconds = time.perf_counter() - started

    zero_cache_seconds = zero_runtime_cache(runtime)

    wall_times: list[float] = []
    reported_ms: list[float] = []
    logits = None
    loop_started = time.perf_counter()
    for position in range(context_len):
        step_started = time.perf_counter()
        logits, elapsed_ms = runtime.run_step_with_mask(hidden_np[position], masks[position], position)
        wall_times.append(time.perf_counter() - step_started)
        reported_ms.append(elapsed_ms)
    loop_seconds = time.perf_counter() - loop_started
    if logits is None:
        raise RuntimeError("ggml prefill loop did not run")

    return logits.astype(np.float32, copy=False), {
        "setup_seconds": setup_seconds,
        "zero_cache_seconds": zero_cache_seconds,
        "loop_wall_seconds": loop_seconds,
        "loop_reported_seconds": float(sum(reported_ms) / 1000.0),
        "total_seconds": setup_seconds + zero_cache_seconds + loop_seconds,
        "steps": context_len,
        "wall_mean_ms": float(np.mean(wall_times) * 1000.0),
        "wall_min_ms": float(np.min(wall_times) * 1000.0),
        "wall_max_ms": float(np.max(wall_times) * 1000.0),
        "reported_mean_ms": float(np.mean(reported_ms)),
        "reported_min_ms": float(np.min(reported_ms)),
        "reported_max_ms": float(np.max(reported_ms)),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
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
    torch.manual_seed(args.seed)

    text = punc_norm(select_text(args.case, args.text))
    model_load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - model_load_started

    runtime_load_started = time.perf_counter()
    runtime = T3GGMLVulkanRuntime()
    runtime_load_seconds = time.perf_counter() - runtime_load_started

    prepare_started = time.perf_counter()
    embeds, _speech_start_token = build_prefill_inputs(model, text)
    prepare_seconds = time.perf_counter() - prepare_started

    reports = []
    for index in range(args.requests):
        torch_logits, torch_seconds = torch_prefill_logits(model, embeds)
        ggml_logits, ggml_timing = ggml_prefill_logits(model, runtime, embeds)
        comparison = diff_summary(ggml_logits, torch_logits)
        reports.append(
            {
                "index": index,
                "torch_prefill_seconds": torch_seconds,
                "ggml_prefill": ggml_timing,
                "speedup_vs_torch": torch_seconds / ggml_timing["total_seconds"]
                if ggml_timing["total_seconds"] > 0.0
                else None,
                "comparison": comparison,
            }
        )

    return {
        "description": "Sequential T3 prefill through existing one-token ggml/Vulkan bridge",
        "case": args.case,
        "normalized_chars": len(text),
        "seed": args.seed,
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "model_load_seconds": model_load_seconds,
        "runtime_load_seconds": runtime_load_seconds,
        "prepare_embeds_seconds": prepare_seconds,
        "context_len": int(embeds.shape[1]),
        "device": runtime.device,
        "requests": args.requests,
        "results": reports,
        "summary": {
            "mean_torch_prefill_seconds": float(np.mean([item["torch_prefill_seconds"] for item in reports])),
            "mean_ggml_total_seconds": float(np.mean([item["ggml_prefill"]["total_seconds"] for item in reports])),
            "mean_ggml_loop_seconds": float(np.mean([item["ggml_prefill"]["loop_wall_seconds"] for item in reports])),
            "mean_speedup_vs_torch": float(np.mean([item["speedup_vs_torch"] for item in reports])),
            "all_argmax_match": all(item["comparison"]["argmax_matches"] for item in reports),
            "min_top10_overlap": int(min(item["comparison"]["top10_overlap"] for item in reports)),
            "max_abs_error": float(max(item["comparison"]["max_abs_error"] for item in reports)),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# T3 ggml/Vulkan Sequential Prefill Probe",
        "",
        f"- Case: `{report['case']}`, chars: `{report['normalized_chars']}`, context length: `{report['context_len']}`",
        f"- Device: `{report['device']}`",
        f"- Mean PyTorch CPU prefill: `{summary['mean_torch_prefill_seconds']:.3f}s`",
        f"- Mean sequential ggml/Vulkan prefill total: `{summary['mean_ggml_total_seconds']:.3f}s`",
        f"- Mean sequential ggml/Vulkan loop only: `{summary['mean_ggml_loop_seconds']:.3f}s`",
        f"- Speedup vs PyTorch: `{summary['mean_speedup_vs_torch']:.3f}x`",
        f"- Argmax matches all requests: `{summary['all_argmax_match']}`",
        f"- Minimum top-10 overlap: `{summary['min_top10_overlap']}/10`",
        f"- Max logits abs error: `{summary['max_abs_error']:.6g}`",
        "",
        "This uses the existing one-token bridge for every prefill token. It is a feasibility probe, not a full-sequence prefill implementation.",
        "",
        "| Request | PyTorch s | ggml total s | ggml loop s | mean ms/step | argmax | top10 | max abs |",
        "|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in report["results"]:
        cmp = item["comparison"]
        ggml = item["ggml_prefill"]
        lines.append(
            f"| {item['index']} | {item['torch_prefill_seconds']:.3f} | "
            f"{ggml['total_seconds']:.3f} | {ggml['loop_wall_seconds']:.3f} | "
            f"{ggml['wall_mean_ms']:.3f} | "
            f"{cmp['argmax_actual']}/{cmp['argmax_expected']} | "
            f"{cmp['top10_overlap']} | {cmp['max_abs_error']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=CHUNK_270)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument(
        "--out-prefix",
        default="t3_ggml_vulkan_prefill_loop_chunk270_2026-07-08",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = run_case(args)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)

    print(f"results={json_path}")
    print(f"markdown={md_path}")
    print(f"context_len={report['context_len']}")
    print(f"torch_prefill_mean={report['summary']['mean_torch_prefill_seconds']:.3f}")
    print(f"ggml_prefill_total_mean={report['summary']['mean_ggml_total_seconds']:.3f}")
    print(f"ggml_prefill_loop_mean={report['summary']['mean_ggml_loop_seconds']:.3f}")
    print(f"speedup_vs_torch={report['summary']['mean_speedup_vs_torch']:.3f}")
    print(f"all_argmax_match={report['summary']['all_argmax_match']}")
    print(f"min_top10_overlap={report['summary']['min_top10_overlap']}")


if __name__ == "__main__":
    main()
