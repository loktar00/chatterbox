#!/usr/bin/env python3
"""Sequential persistent API-shaped Vulkan hybrid benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from benchmark_s3_flow_hybrid_vulkan_encoder_estimator import VulkanEncoderChain
from benchmark_s3_flow_hybrid_vulkan_estimator import VulkanEstimatorChain
from benchmark_t3_hybrid_s3_hift_vulkan_api_path import (
    parse_frame_sizes,
    run_hybrid_s3_and_watermark,
    select_text,
)
from benchmark_tts_servers import CHUNK_270
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from split_hift_vulkan_runtime import SplitHiFTVulkan
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def wav_to_file(wav: torch.Tensor, sample_rate: int, path: Path) -> float:
    arr = wav.detach().cpu().squeeze().numpy()
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    sf.write(path, arr, sample_rate, format="WAV")
    return float(arr.shape[0] / sample_rate)


def force_speech_token_count(tokens: torch.Tensor, target: int) -> torch.Tensor:
    tokens = tokens.reshape(-1)
    if int(tokens.numel()) > target:
        return tokens[:target].contiguous()
    if int(tokens.numel()) < target:
        pad = torch.full(
            (target - int(tokens.numel()),),
            S3GEN_SIL,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat([tokens, pad], dim=0).contiguous()
    return tokens.contiguous()


def parse_bucket_spec(spec: str) -> dict[int, tuple[int, int, int]]:
    """Parse encoder token/up frame pairs keyed by encoder token frames."""
    buckets: dict[int, tuple[int, int, int]] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) == 2:
            encoder_token_frames = int(pieces[0])
            encoder_up_frames = int(pieces[1])
            estimator_frames = encoder_up_frames
        elif len(pieces) == 3:
            encoder_token_frames = int(pieces[0])
            encoder_up_frames = int(pieces[1])
            estimator_frames = int(pieces[2])
        else:
            raise ValueError(f"Invalid bucket spec {part!r}; use token:up or token:up:estimator")
        buckets[encoder_token_frames] = (encoder_token_frames, encoder_up_frames, estimator_frames)
    if not buckets:
        raise ValueError("At least one S3 bucket is required")
    return buckets


def prompt_token_len(model: ChatterboxTurboTTS) -> int:
    value = model.conds.gen["prompt_token_len"]
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(value).reshape(-1)[0])


def infer_s3_bucket_key(model: ChatterboxTurboTTS, speech_tokens: torch.Tensor) -> dict[str, int]:
    valid_speech_tokens = int((speech_tokens.reshape(-1) < 6561).sum().item())
    speech_tokens_with_silence = valid_speech_tokens + 3
    prompt_tokens = prompt_token_len(model)
    encoder_token_frames = prompt_tokens + speech_tokens_with_silence
    return {
        "valid_speech_tokens": valid_speech_tokens,
        "speech_tokens_with_silence": speech_tokens_with_silence,
        "prompt_tokens": prompt_tokens,
        "encoder_token_frames": encoder_token_frames,
        "encoder_up_frames": encoder_token_frames * 2,
        "estimator_frames": encoder_token_frames * 2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=CHUNK_270)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument(
        "--force-speech-token-count",
        type=int,
        help="Optionally truncate/pad T3 speech tokens before S3 to hit a fixed S3 bucket.",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--same-seed-each-request", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--s3-timesteps", type=int, default=2)
    parser.add_argument("--hift-window-frames", type=int, default=128)
    parser.add_argument("--hift-center-frames", type=int, default=96)
    parser.add_argument("--hift-extra-frame-sizes", default="")
    parser.add_argument("--hift-exact-tail", action="store_true")
    parser.add_argument("--hift-compact-tail", action="store_true")
    parser.add_argument("--t3-lib-path", type=Path, help="Optional ggml T3 bridge library override.")
    parser.add_argument("--encoder-token-frames", type=int, default=605)
    parser.add_argument("--encoder-up-frames", type=int, default=1210)
    parser.add_argument("--estimator-frames", type=int, default=1210)
    parser.add_argument(
        "--auto-s3-bucket",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Infer the S3 bucket from generated token count and load a matching validated bucket.",
    )
    parser.add_argument(
        "--s3-buckets",
        default="605:1210,611:1222",
        help="Comma-separated validated buckets as encoder_token:encoder_up[:estimator].",
    )
    parser.add_argument(
        "--out-prefix",
        default="ggml_t3_hybrid_s3_hift_vulkan_persistent_chunk270_2req_2026-07-08",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.out_prefix}.json"

    selected_text = select_text(args.case, args.text)
    text = punc_norm(selected_text)
    hift_frame_sizes = parse_frame_sizes(args.hift_extra_frame_sizes, args.hift_window_frames)

    total_start = time.perf_counter()
    model_load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - model_load_start

    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    t3_load_start = time.perf_counter()
    t3_runtime_kwargs = {"lib_path": args.t3_lib_path} if args.t3_lib_path is not None else {}
    t3_runtime = T3GGMLVulkanRuntime(**t3_runtime_kwargs)
    t3_runtime_load_seconds = time.perf_counter() - t3_load_start

    encoder_chain = None
    estimator_chain = None
    split_hift = None
    s3_runtime_load_seconds = None
    bucket_specs = parse_bucket_spec(args.s3_buckets)
    bucket_cache: dict[tuple[int, int, int], tuple[VulkanEncoderChain, VulkanEstimatorChain]] = {}

    def ensure_s3_runtimes(bucket: tuple[int, int, int] | None) -> dict[str, Any]:
        nonlocal encoder_chain, estimator_chain, split_hift, s3_runtime_load_seconds
        if bucket is None:
            encoder_chain = None
            estimator_chain = None
            if split_hift is not None:
                return {
                    "loaded_this_request": False,
                    "selected_bucket": None,
                    "encoder_load_seconds": 0.0,
                    "estimator_load_seconds": 0.0,
                    "hift_load_seconds": 0.0,
                    "total_load_seconds": 0.0,
                }
            hift_start = time.perf_counter()
            split_hift = SplitHiFTVulkan(
                model.s3gen.mel2wav,
                frame_sizes=hift_frame_sizes,
                allow_padding=False,
            )
            hift_seconds = time.perf_counter() - hift_start
            return {
                "loaded_this_request": True,
                "selected_bucket": None,
                "encoder_load_seconds": 0.0,
                "estimator_load_seconds": 0.0,
                "hift_load_seconds": hift_seconds,
                "total_load_seconds": hift_seconds,
            }

        if bucket in bucket_cache and split_hift is not None:
            encoder_chain, estimator_chain = bucket_cache[bucket]
            return {
                "loaded_this_request": False,
                "selected_bucket": {
                    "encoder_token_frames": bucket[0],
                    "encoder_up_frames": bucket[1],
                    "estimator_frames": bucket[2],
                },
                "encoder_load_seconds": 0.0,
                "estimator_load_seconds": 0.0,
                "hift_load_seconds": 0.0,
                "total_load_seconds": 0.0,
            }

        started = time.perf_counter()
        encoder_seconds = 0.0
        estimator_seconds = 0.0
        if bucket in bucket_cache:
            encoder_chain, estimator_chain = bucket_cache[bucket]
        else:
            encoder_start = time.perf_counter()
            encoder_chain = VulkanEncoderChain(
                token_frames=bucket[0],
                up_frames=bucket[1],
            )
            encoder_seconds = time.perf_counter() - encoder_start
            estimator_start = time.perf_counter()
            estimator_chain = VulkanEstimatorChain(frames=bucket[2])
            estimator_seconds = time.perf_counter() - estimator_start
            bucket_cache[bucket] = (encoder_chain, estimator_chain)
        if split_hift is None:
            hift_start = time.perf_counter()
            split_hift = SplitHiFTVulkan(
                model.s3gen.mel2wav,
                frame_sizes=hift_frame_sizes,
                allow_padding=False,
            )
            hift_seconds = time.perf_counter() - hift_start
        else:
            hift_seconds = 0.0
        s3_runtime_load_seconds = time.perf_counter() - started
        return {
            "loaded_this_request": True,
            "selected_bucket": {
                "encoder_token_frames": bucket[0],
                "encoder_up_frames": bucket[1],
                "estimator_frames": bucket[2],
            },
            "encoder_load_seconds": encoder_seconds,
            "estimator_load_seconds": estimator_seconds,
            "hift_load_seconds": hift_seconds,
            "total_load_seconds": s3_runtime_load_seconds,
        }

    common_kwargs = {
        "t3_cond": model.conds.t3,
        "text_tokens": text_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_gen_len": args.max_gen_len,
    }

    request_results: list[dict[str, Any]] = []
    for index in range(args.requests):
        request_seed = args.seed if args.same_seed_each_request else args.seed + index
        torch.manual_seed(request_seed)
        had_s3_modules_before_t3 = encoder_chain is not None and estimator_chain is not None and split_hift is not None
        rss_before = rss_mb()
        request_start = time.perf_counter()

        t3_start = time.perf_counter()
        vulkan_tokens = inference_turbo_vulkan(model.t3, t3_runtime, **common_kwargs)
        t3_seconds = time.perf_counter() - t3_start
        raw_t3_tokens = int(vulkan_tokens.numel())
        token_adjustment = None
        if args.force_speech_token_count is not None:
            vulkan_tokens = force_speech_token_count(vulkan_tokens, args.force_speech_token_count)
            token_adjustment = {
                "raw_tokens": raw_t3_tokens,
                "target_tokens": args.force_speech_token_count,
                "adjusted_tokens": int(vulkan_tokens.numel()),
                "mode": "truncate" if raw_t3_tokens > args.force_speech_token_count else (
                    "pad" if raw_t3_tokens < args.force_speech_token_count else "unchanged"
                ),
                "pad_token": int(S3GEN_SIL),
            }

        bucket_inference = infer_s3_bucket_key(model, vulkan_tokens)
        if args.auto_s3_bucket:
            bucket = bucket_specs.get(bucket_inference["encoder_token_frames"])
        else:
            bucket = (args.encoder_token_frames, args.encoder_up_frames, args.estimator_frames)
        s3_runtime_load = ensure_s3_runtimes(bucket)
        s3_start = time.perf_counter()
        wav, s3_timings = run_hybrid_s3_and_watermark(
            model,
            vulkan_tokens,
            encoder_chain=encoder_chain,
            estimator_chain=estimator_chain,
            split_hift=split_hift,
            hift_window_frames=args.hift_window_frames,
            hift_center_frames=args.hift_center_frames,
            hift_exact_tail=args.hift_exact_tail,
            hift_compact_tail=args.hift_compact_tail,
            s3_timesteps=args.s3_timesteps,
        )
        s3_seconds = time.perf_counter() - s3_start

        wav_path = OUT_DIR / f"{args.out_prefix}_request{index}.wav"
        audio_seconds = wav_to_file(wav, model.sr, wav_path)
        request_seconds = time.perf_counter() - request_start
        rss_after = rss_mb()

        request_results.append(
            {
                "index": index,
                "seed": request_seed,
                "had_s3_modules_before_t3": had_s3_modules_before_t3,
                "vulkan_t3_seconds": t3_seconds,
                "raw_vulkan_t3_tokens": raw_t3_tokens,
                "vulkan_t3_tokens": int(vulkan_tokens.numel()),
                "token_adjustment": token_adjustment,
                "bucket_inference": bucket_inference,
                "s3_runtime_load": s3_runtime_load,
                "s3_and_watermark_seconds": s3_seconds,
                "s3_stage_breakdown": s3_timings,
                "warm_pipeline_seconds_excluding_runtime_load": t3_seconds + s3_seconds,
                "request_seconds_including_lazy_runtime_load": request_seconds,
                "audio_seconds": audio_seconds,
                "wav_path": wav_path.as_posix(),
                "rss_mb_before": rss_before,
                "rss_mb_after": rss_after,
                "rss_mb_delta": rss_after - rss_before,
            }
        )
        print(
            f"request={index} had_s3_before_t3={had_s3_modules_before_t3} "
            f"t3={t3_seconds:.3f}s s3={s3_seconds:.3f}s "
            f"warm={t3_seconds + s3_seconds:.3f}s total={request_seconds:.3f}s "
            f"rss_delta={rss_after - rss_before:.3f}MB"
        )

    total_seconds = time.perf_counter() - total_start
    warm_values = [row["warm_pipeline_seconds_excluding_runtime_load"] for row in request_results]
    t3_values = [row["vulkan_t3_seconds"] for row in request_results]
    report = {
        "description": "Sequential persistent API-shaped benchmark with Vulkan T3, hybrid Vulkan S3 encoder+estimator, and Vulkan HiFT.",
        "case": args.case,
        "normalized_chars": len(text),
        "requests": args.requests,
        "same_seed_each_request": args.same_seed_each_request,
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_gen_len": args.max_gen_len,
            "force_speech_token_count": args.force_speech_token_count,
            "s3_timesteps": args.s3_timesteps,
        },
        "bucket": {
            "auto_s3_bucket": args.auto_s3_bucket,
            "validated_s3_buckets": [
                {
                    "encoder_token_frames": value[0],
                    "encoder_up_frames": value[1],
                    "estimator_frames": value[2],
                }
                for value in bucket_specs.values()
            ],
            "encoder_token_frames": args.encoder_token_frames,
            "encoder_up_frames": args.encoder_up_frames,
            "estimator_frames": args.estimator_frames,
        },
        "hift": {
            "window_frames": args.hift_window_frames,
            "center_frames": args.hift_center_frames,
            "extra_frame_sizes": hift_frame_sizes,
            "exact_tail": args.hift_exact_tail,
            "compact_tail": args.hift_compact_tail,
        },
        "device": t3_runtime.device,
        "model_load_seconds": model_load_seconds,
        "t3_runtime_load_seconds": t3_runtime_load_seconds,
        "t3_lib_path": str(args.t3_lib_path) if args.t3_lib_path is not None else None,
        "s3_runtime_load_seconds": s3_runtime_load_seconds,
        "request_results": request_results,
        "summary": {
            "mean_warm_pipeline_seconds": float(np.mean(warm_values)),
            "min_warm_pipeline_seconds": float(np.min(warm_values)),
            "max_warm_pipeline_seconds": float(np.max(warm_values)),
            "mean_t3_seconds": float(np.mean(t3_values)),
            "request0_to_request1_t3_ratio": (
                t3_values[1] / t3_values[0] if len(t3_values) > 1 and t3_values[0] > 0 else None
            ),
            "final_rss_mb": rss_mb(),
            "total_process_seconds": total_seconds,
        },
        "notes": [
            "The live API remains CPU-only.",
            "S3/HiFT Vulkan modules are lazily loaded after request 0 T3, then remain resident for later requests.",
            "No ROCm/HIP path is used.",
        ],
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={json_path}")
    print(f"mean_warm_pipeline_seconds={report['summary']['mean_warm_pipeline_seconds']:.3f}")
    print(f"request0_to_request1_t3_ratio={report['summary']['request0_to_request1_t3_ratio']}")
    print(f"total_process_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
