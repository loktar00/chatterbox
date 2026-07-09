#!/usr/bin/env python3
"""Validate and benchmark reusable T3 conditioning-prefix prefill cache."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from t3_ggml_vulkan_runtime import _legacy_past, _temporary_torch_threads


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def tensor_max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> int:
    a_top = set(torch.topk(a.detach().cpu().reshape(-1), k).indices.tolist())
    b_top = set(torch.topk(b.detach().cpu().reshape(-1), k).indices.tolist())
    return len(a_top & b_top)


def past_summary(a: Any, b: Any) -> dict[str, Any]:
    comparisons = []
    for layer, (layer_a, layer_b) in enumerate(zip(_legacy_past(a), _legacy_past(b), strict=True)):
        for index, name in enumerate(("key", "value")):
            tensor_a = layer_a[index].detach().cpu()
            tensor_b = layer_b[index].detach().cpu()
            diff = (tensor_a - tensor_b).abs()
            comparisons.append(
                {
                    "layer": layer,
                    "kind": name,
                    "shape": list(tensor_a.shape),
                    "max_abs_error": float(diff.max().item()),
                    "mean_abs_error": float(diff.mean().item()),
                    "allclose_1e_4": bool(torch.allclose(tensor_a, tensor_b, atol=1e-4, rtol=1e-4)),
                    "allclose_1e_3": bool(torch.allclose(tensor_a, tensor_b, atol=1e-3, rtol=1e-3)),
                }
            )
    return {
        "allclose_1e_4": all(item["allclose_1e_4"] for item in comparisons),
        "allclose_1e_3": all(item["allclose_1e_3"] for item in comparisons),
        "max_abs_error": max(item["max_abs_error"] for item in comparisons),
        "mean_abs_error": float(np.mean([item["mean_abs_error"] for item in comparisons])),
        "comparisons": comparisons,
    }


def run_full_prefill(t3, embeds: torch.Tensor, prefill_threads: int) -> tuple[Any, torch.Tensor, float]:
    started = time.perf_counter()
    with _temporary_torch_threads(prefill_threads):
        outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        logits = t3.speech_head(outputs[0][:, -1:])
    return outputs.past_key_values, logits, time.perf_counter() - started


def run_prefix_prefill(t3, embeds: torch.Tensor, len_cond: int, prefill_threads: int) -> tuple[Any, torch.Tensor, float, float]:
    cond_embeds = embeds[:, :len_cond].contiguous()
    suffix_embeds = embeds[:, len_cond:].contiguous()

    cond_started = time.perf_counter()
    with _temporary_torch_threads(prefill_threads):
        cond_outputs = t3.tfmr(inputs_embeds=cond_embeds, use_cache=True)
    cond_seconds = time.perf_counter() - cond_started

    suffix_started = time.perf_counter()
    with _temporary_torch_threads(prefill_threads):
        suffix_outputs = t3.tfmr(
            inputs_embeds=suffix_embeds,
            past_key_values=cond_outputs.past_key_values,
            use_cache=True,
        )
        logits = t3.speech_head(suffix_outputs[0][:, -1:])
    suffix_seconds = time.perf_counter() - suffix_started
    return suffix_outputs.past_key_values, logits, cond_seconds, suffix_seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=CHUNK_270)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--prefill-threads", type=int, default=8)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument(
        "--validate-cache",
        action="store_true",
        help="Also compare full/split KV caches. This is memory-heavy; logits validation is the default.",
    )
    parser.add_argument("--output", type=Path, default=OUT_DIR / "t3_prefix_prefill_cache_chunk270_2026-07-08.json")
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
    model_load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - model_load_started
    t3 = model.t3

    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    prepare_started = time.perf_counter()
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    prepare_seconds = time.perf_counter() - prepare_started

    # Warm once before recording timings.
    run_full_prefill(t3, embeds, args.prefill_threads)
    run_prefix_prefill(t3, embeds, len_cond, args.prefill_threads)

    runs = []
    reference_full_logits = None
    reference_split_logits = None
    reference_full_past = None
    reference_split_past = None
    for index in range(args.requests):
        full_past, full_logits, full_seconds = run_full_prefill(t3, embeds, args.prefill_threads)
        split_past, split_logits, cond_seconds, suffix_seconds = run_prefix_prefill(t3, embeds, len_cond, args.prefill_threads)
        if reference_full_logits is None:
            reference_full_logits = full_logits
            reference_split_logits = split_logits
            if args.validate_cache:
                reference_full_past = full_past
                reference_split_past = split_past
        logits_diff = tensor_max_abs(split_logits, full_logits)
        argmax_match = int(torch.argmax(split_logits[:, -1, :]).item()) == int(torch.argmax(full_logits[:, -1, :]).item())
        runs.append(
            {
                "index": index,
                "full_prefill_seconds": full_seconds,
                "prefix_build_seconds": cond_seconds,
                "suffix_prefill_seconds": suffix_seconds,
                "steady_state_speedup": full_seconds / suffix_seconds if suffix_seconds > 0 else None,
                "amortized_one_request_speedup": full_seconds / (cond_seconds + suffix_seconds)
                if (cond_seconds + suffix_seconds) > 0
                else None,
                "logits_max_abs_error": logits_diff,
                "logits_allclose_1e_4": bool(torch.allclose(split_logits, full_logits, atol=1e-4, rtol=1e-4)),
                "logits_allclose_1e_3": bool(torch.allclose(split_logits, full_logits, atol=1e-3, rtol=1e-3)),
                "argmax_match": argmax_match,
                "top10_overlap": topk_overlap(split_logits, full_logits, 10),
            }
        )
        print(
            f"request={index} full={full_seconds:.3f}s prefix_build={cond_seconds:.3f}s "
            f"suffix={suffix_seconds:.3f}s speedup={runs[-1]['steady_state_speedup']:.2f} "
            f"argmax={argmax_match} top10={runs[-1]['top10_overlap']}/10"
        )
        if not args.validate_cache or index > 0:
            del full_past, split_past
        del full_logits, split_logits
        gc.collect()

    assert reference_full_logits is not None
    assert reference_split_logits is not None
    cache = None
    if args.validate_cache:
        assert reference_full_past is not None
        assert reference_split_past is not None
        cache = past_summary(reference_split_past, reference_full_past)
    full_values = [item["full_prefill_seconds"] for item in runs]
    suffix_values = [item["suffix_prefill_seconds"] for item in runs]
    prefix_values = [item["prefix_build_seconds"] for item in runs]
    report = {
        "description": "Split T3 prefill into reusable conditioning prefix plus per-request text/speech-start suffix.",
        "case": args.case,
        "normalized_chars": len(text),
        "text_token_count": int(text_tokens.numel()),
        "conditioning_tokens": int(len_cond),
        "suffix_tokens": int(embeds.shape[1] - len_cond),
        "initial_context_len": int(embeds.shape[1]),
        "threads": args.threads,
        "interop_threads": args.interop_threads,
        "prefill_threads": args.prefill_threads,
        "validate_cache": args.validate_cache,
        "requests": args.requests,
        "model_load_seconds": model_load_seconds,
        "prepare_embeds_seconds": prepare_seconds,
        "runs": runs,
        "summary": {
            "mean_full_prefill_seconds": float(np.mean(full_values)),
            "mean_prefix_build_seconds": float(np.mean(prefix_values)),
            "mean_suffix_prefill_seconds": float(np.mean(suffix_values)),
            "steady_state_speedup": float(np.mean(full_values) / np.mean(suffix_values)),
            "one_request_amortized_speedup": float(np.mean(full_values) / (np.mean(prefix_values) + np.mean(suffix_values))),
            "all_logits_allclose_1e_4": all(item["logits_allclose_1e_4"] for item in runs),
            "all_logits_allclose_1e_3": all(item["logits_allclose_1e_3"] for item in runs),
            "all_argmax_match": all(item["argmax_match"] for item in runs),
            "min_top10_overlap": min(item["top10_overlap"] for item in runs),
        },
        "cache_validation": cache,
        "notes": [
            "The conditioning prefix is constant for the loaded voice/conditioning object.",
            "A served implementation could build the prefix cache once at startup, then run only the suffix prefill per request.",
            "Default validation compares final logits only to keep memory pressure lower. Use --validate-cache for the heavier KV-cache comparison.",
            "This is CPU-only prefill restructuring; it does not use ROCm/HIP and does not start an experimental API server.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")
    print(f"steady_state_speedup={report['summary']['steady_state_speedup']:.2f}")
    print(f"all_logits_allclose_1e_4={report['summary']['all_logits_allclose_1e_4']}")
    if cache is not None:
        print(f"cache_allclose_1e_4={cache['allclose_1e_4']}")
    raise SystemExit(0 if report["summary"]["all_logits_allclose_1e_4"] else 1)


if __name__ == "__main__":
    main()
