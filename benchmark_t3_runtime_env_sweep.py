#!/usr/bin/env python3
"""Sweep current ggml/Vulkan T3 runtime knobs with token equality checks."""

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
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def parse_ints(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def run_t3(
    model: ChatterboxTurboTTS,
    runtime: T3GGMLVulkanRuntime,
    text_tokens: torch.Tensor,
    *,
    seed: int,
    max_gen_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    fast_loop: bool,
    fast_sampler: bool,
    prefill_threads: int,
    prefix_prefill: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rss_before = rss_mb()
    started = time.perf_counter()
    tokens = inference_turbo_vulkan(
        model.t3,
        runtime,
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_gen_len=max_gen_len,
        fast_loop=fast_loop,
        fast_sampler=fast_sampler,
        prefill_threads=prefill_threads,
        prefix_prefill=prefix_prefill,
    )
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_mb(),
        "prefill_info": getattr(runtime, "last_prefill_info", None),
        "loop_info": getattr(runtime, "last_loop_info", None),
        "token_count": int(tokens.numel()),
        "tokens": [int(item) for item in tokens.detach().cpu().reshape(-1).tolist()],
    }


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=CHUNK_270)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--prefill-threads", default="2,4,6,8,10,12")
    parser.add_argument("--prefix-prefill", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--lib-path", type=Path, default=ROOT / "libt3_ggml_vulkan_bridge_f16weights.so")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_runtime_env_sweep_chunk270_2026-07-08.json",
    )
    args = parser.parse_args()

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

    text = punc_norm(select_text(args.case, args.text))

    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - load_start
    rss_mb_after_model_load = rss_mb()
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)

    runtime_start = time.perf_counter()
    runtime = T3GGMLVulkanRuntime(lib_path=args.lib_path)
    runtime_load_seconds = time.perf_counter() - runtime_start
    rss_mb_after_runtime_load = rss_mb()

    prefill_values = parse_ints(args.prefill_threads)
    results = []
    reference_tokens = None
    for prefill_threads in prefill_values:
        runs = []
        for index in range(args.requests):
            run = run_t3(
                model,
                runtime,
                text_tokens,
                seed=args.seed,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                fast_loop=True,
                fast_sampler=True,
                prefill_threads=prefill_threads,
                prefix_prefill=args.prefix_prefill,
            )
            if reference_tokens is None:
                reference_tokens = run["tokens"]
            run["tokens_equal_reference"] = run["tokens"] == reference_tokens
            run["index"] = index
            runs.append(run)
            print(
                f"prefill_threads={prefill_threads} request={index} "
                f"seconds={run['seconds']:.3f} tokens={run['token_count']} "
                f"equal={run['tokens_equal_reference']}"
            )
        seconds = [float(run["seconds"]) for run in runs]
        results.append(
            {
                "prefill_threads": prefill_threads,
                "runs": runs,
                "summary": summarize(seconds),
                "all_tokens_equal_reference": all(bool(run["tokens_equal_reference"]) for run in runs),
            }
        )

    best = min(
        results,
        key=lambda item: (
            item["summary"]["mean"] if item["summary"]["mean"] is not None else float("inf")
        ),
    )
    report = {
        "description": "Sweep current ggml/Vulkan T3 runtime prefill thread setting using the real optimized runtime path.",
        "case": args.case,
        "normalized_chars": len(text),
        "text_token_count": int(text_tokens.numel()),
        "requests_per_setting": args.requests,
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_gen_len": args.max_gen_len,
            "fast_loop": True,
            "fast_sampler": True,
            "prefix_prefill": args.prefix_prefill,
        },
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "lib_path": args.lib_path.as_posix(),
        "model_load_seconds": model_load_seconds,
        "runtime_load_seconds": runtime_load_seconds,
        "rss_mb_after_model_load": rss_mb_after_model_load,
        "rss_mb_after_runtime_load": rss_mb_after_runtime_load,
        "rss_mb_after_runs": rss_mb(),
        "device": runtime.device,
        "results": results,
        "best": {
            "prefill_threads": best["prefill_threads"],
            "mean_seconds": best["summary"]["mean"],
            "min_seconds": best["summary"]["min"],
            "max_seconds": best["summary"]["max"],
            "all_tokens_equal_reference": best["all_tokens_equal_reference"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")
    print(
        f"best_prefill_threads={report['best']['prefill_threads']} "
        f"mean_seconds={report['best']['mean_seconds']:.3f}"
    )


if __name__ == "__main__":
    main()
