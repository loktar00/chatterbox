#!/usr/bin/env python3
"""Measure CPU thread sensitivity for the T3 PyTorch prefill stage."""

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


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


@torch.inference_mode()
def prepare_embeds(model: ChatterboxTurboTTS, text: str) -> torch.Tensor:
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
    return embeds


@torch.inference_mode()
def run_prefill(model: ChatterboxTurboTTS, embeds: torch.Tensor) -> torch.Tensor:
    outputs = model.t3.tfmr(inputs_embeds=embeds, use_cache=True)
    return model.t3.speech_head(outputs[0][:, -1:])[:, -1, :]


def time_prefill(
    model: ChatterboxTurboTTS,
    embeds: torch.Tensor,
    *,
    threads: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    for _ in range(warmup):
        run_prefill(model, embeds)
    times = []
    argmaxes = []
    for _ in range(iterations):
        started = time.perf_counter()
        logits = run_prefill(model, embeds)
        times.append(time.perf_counter() - started)
        argmaxes.append(int(logits.detach().cpu().numpy().reshape(-1).argmax()))
    return {
        "threads": threads,
        "warmup": warmup,
        "iterations": iterations,
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "std_seconds": float(np.std(times)),
        "times_seconds": times,
        "argmaxes": argmaxes,
        "argmax_stable": len(set(argmaxes)) == 1,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# T3 PyTorch Prefill Thread Sweep",
        "",
        f"- Case: `{report['case']}`, chars: `{report['normalized_chars']}`, context length: `{report['context_len']}`",
        f"- Interop threads: `{report['interop_threads']}`",
        f"- Best mean: `{report['summary']['best_threads']}` threads at `{report['summary']['best_mean_seconds']:.3f}s`",
        "",
        "| Threads | Mean s | Min s | Max s | Std s | Argmax stable |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['threads']} | {item['mean_seconds']:.3f} | {item['min_seconds']:.3f} | "
            f"{item['max_seconds']:.3f} | {item['std_seconds']:.3f} | {item['argmax_stable']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=CHUNK_270)
    parser.add_argument("--threads-list", default="1,2,3,4,6,8")
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--out-prefix", default="t3_prefill_thread_sweep_chunk270_2026-07-08")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    threads_list = [int(part) for part in args.threads_list.split(",") if part.strip()]
    text = punc_norm(select_text(args.case, args.text))

    model_load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - model_load_started

    prepare_started = time.perf_counter()
    embeds = prepare_embeds(model, text)
    prepare_seconds = time.perf_counter() - prepare_started

    results = [
        time_prefill(
            model,
            embeds,
            threads=threads,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for threads in threads_list
    ]
    best = min(results, key=lambda item: item["mean_seconds"])
    report = {
        "description": "CPU thread sweep for Chatterbox Turbo T3 PyTorch prefill",
        "case": args.case,
        "normalized_chars": len(text),
        "seed": args.seed,
        "interop_threads": args.interop_threads,
        "model_load_seconds": model_load_seconds,
        "prepare_embeds_seconds": prepare_seconds,
        "context_len": int(embeds.shape[1]),
        "threads_list": threads_list,
        "results": results,
        "summary": {
            "best_threads": best["threads"],
            "best_mean_seconds": best["mean_seconds"],
            "threads2_mean_seconds": next(
                (item["mean_seconds"] for item in results if item["threads"] == 2),
                None,
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, md_path)

    print(f"results={json_path}")
    print(f"markdown={md_path}")
    print(f"context_len={report['context_len']}")
    for item in results:
        print(
            f"threads={item['threads']} mean={item['mean_seconds']:.3f} "
            f"min={item['min_seconds']:.3f} max={item['max_seconds']:.3f}"
        )
    print(f"best_threads={best['threads']} best_mean={best['mean_seconds']:.3f}")


if __name__ == "__main__":
    main()
