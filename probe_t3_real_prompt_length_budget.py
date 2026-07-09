#!/usr/bin/env python3
"""Measure real T3 Turbo prompt/cache lengths for Vulkan bucket planning."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"


def next_bucket(length: int, buckets: tuple[int, ...]) -> int | None:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return None


def case_texts() -> dict[str, str]:
    return {
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }


def measure_case(model: ChatterboxTurboTTS, name: str, text: str, buckets: tuple[int, ...]) -> dict:
    normalized = punc_norm(text)
    text_tokens = model.tokenizer(normalized, return_tensors="pt", padding=True, truncation=True)
    text_ids = text_tokens.input_ids.to(model.device)
    speech_start_token = model.t3.hp.start_speech_token * torch.ones_like(text_ids[:, :1])

    with torch.inference_mode():
        embeds, len_cond = model.t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_ids,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

    initial_context_len = int(embeds.shape[1])
    p128_generation_capacity = max(0, 128 - initial_context_len)
    required_for_128_generated = initial_context_len + 128
    required_for_512_generated = initial_context_len + 512
    required_for_1000_generated = initial_context_len + 1000

    return {
        "case": name,
        "chars": len(text),
        "normalized_chars": len(normalized),
        "text_token_shape": list(text_ids.shape),
        "text_tokens": int(text_ids.shape[1]),
        "conditioning_tokens": int(len_cond),
        "initial_speech_tokens": 1,
        "initial_context_len": initial_context_len,
        "fits_existing_p128_initial_context": initial_context_len <= 128,
        "p128_generation_capacity_after_initial_context": p128_generation_capacity,
        "required_cache_for_128_generated": required_for_128_generated,
        "required_cache_for_512_generated": required_for_512_generated,
        "required_cache_for_1000_generated": required_for_1000_generated,
        "bucket_for_128_generated": next_bucket(required_for_128_generated, buckets),
        "bucket_for_512_generated": next_bucket(required_for_512_generated, buckets),
        "bucket_for_1000_generated": next_bucket(required_for_1000_generated, buckets),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--buckets", default="128,192,256,384,512,768,1024,1280")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_real_prompt_length_budget_2026-07-08.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    buckets = tuple(int(item) for item in args.buckets.split(",") if item.strip())

    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    if model.conds is None:
        raise SystemExit("Default Turbo conditionals are not loaded")

    cases = [measure_case(model, name, text, buckets) for name, text in case_texts().items()]
    report = {
        "seconds": time.perf_counter() - started,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "buckets": list(buckets),
        "current_compiled_vulkan_cache_bucket": 128,
        "cases": cases,
        "notes": [
            "This is CPU-only shape planning; it does not run ROCm/HIP or the live API.",
            "The current Vulkan T3 chunks are compiled for p128 total cache length.",
            "Real generation needs a cache bucket large enough for initial conditioning/text plus generated speech tokens.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    for case in cases:
        print(
            f"{case['case']}: chars={case['chars']} text_tokens={case['text_tokens']} "
            f"cond={case['conditioning_tokens']} initial_context={case['initial_context_len']} "
            f"p128_remaining={case['p128_generation_capacity_after_initial_context']} "
            f"bucket_512gen={case['bucket_for_512_generated']} "
            f"bucket_1000gen={case['bucket_for_1000_generated']}"
        )


if __name__ == "__main__":
    main()
