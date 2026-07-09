#!/usr/bin/env python3
"""API-shaped run with Vulkan T3, hybrid Vulkan S3, and Vulkan HiFT."""

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

from benchmark_s3_flow_hybrid_vulkan_encoder_estimator import (
    VulkanEncoderChain,
    patch_encoder,
)
from benchmark_s3_flow_hybrid_vulkan_estimator import (
    VulkanEstimatorChain,
    patch_estimator,
)
from benchmark_t3_ggml_vulkan_api_path import (
    parse_frame_sizes,
    source_from_speech_feat,
    vulkan_hift_decode,
)
from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from split_hift_vulkan_runtime import SplitHiFTVulkan
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"
DEFAULT_TEXT = "Hello world, this is a short Vulkan hybrid S3 smoke test."


def wav_to_file(wav: torch.Tensor, sample_rate: int, path: Path) -> float:
    arr = wav.detach().cpu().squeeze().numpy()
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    sf.write(path, arr, sample_rate, format="WAV")
    return float(arr.shape[0] / sample_rate)


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def run_hybrid_s3_and_watermark(
    model: ChatterboxTurboTTS,
    speech_tokens: torch.Tensor,
    *,
    encoder_chain: VulkanEncoderChain | None,
    estimator_chain: VulkanEstimatorChain | None,
    split_hift: SplitHiFTVulkan | None,
    hift_window_frames: int,
    hift_center_frames: int,
    hift_exact_tail: bool = False,
    hift_compact_tail: bool = False,
    s3_timesteps: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    timings: dict[str, Any] = {}
    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    encoder_records: list[dict[str, Any]] = []
    estimator_records: list[dict[str, Any]] = []
    encoder = None
    estimator = None
    original_encoder_forward = None
    original_estimator_forward = None
    if encoder_chain is not None:
        encoder, original_encoder_forward = patch_encoder(model, encoder_chain, encoder_records)
    if estimator_chain is not None:
        estimator, original_estimator_forward = patch_estimator(model, estimator_chain, estimator_records)

    flow_start = time.perf_counter()
    try:
        mels = model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=s3_timesteps,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)
    finally:
        if encoder is not None and original_encoder_forward is not None:
            encoder.forward = original_encoder_forward
        if estimator is not None and original_estimator_forward is not None:
            estimator.forward = original_estimator_forward
    timings["s3_flow_seconds"] = time.perf_counter() - flow_start
    timings["encoder_patch_enabled"] = encoder_chain is not None
    timings["estimator_patch_enabled"] = estimator_chain is not None
    timings["mel_frames"] = int(mels.shape[-1])
    timings["s3_timesteps"] = s3_timesteps
    timings["speech_token_count_with_silence"] = int(speech_tokens.numel())
    timings["encoder_calls"] = {
        "total": len(encoder_records),
        "vulkan": sum(1 for record in encoder_records if not record["fallback_cpu"]),
        "fallback_cpu": sum(1 for record in encoder_records if record["fallback_cpu"]),
        "vulkan_chain_fetch_seconds": sum(
            float(record.get("vulkan_chain_fetch_seconds", 0.0)) for record in encoder_records
        ),
        "cache_stats": getattr(encoder_chain, "cache_stats", None),
        "records": encoder_records,
    }
    timings["estimator_calls"] = {
        "total": len(estimator_records),
        "vulkan": sum(1 for record in estimator_records if not record["fallback_cpu"]),
        "fallback_cpu": sum(1 for record in estimator_records if record["fallback_cpu"]),
        "vulkan_chain_fetch_seconds": sum(
            float(record.get("vulkan_chain_fetch_seconds", 0.0)) for record in estimator_records
        ),
        "cache_stats": getattr(estimator_chain, "cache_stats", None),
        "records": estimator_records,
    }

    source_start = time.perf_counter()
    if split_hift is not None:
        source = split_hift._source_from_speech_feat(mels)
    else:
        source = source_from_speech_feat(model.s3gen.mel2wav, mels)
    timings["source_seconds"] = time.perf_counter() - source_start

    hift_start = time.perf_counter()
    if split_hift is not None:
        wav = vulkan_hift_decode(
            split_hift,
            mels,
            source,
            window_frames=hift_window_frames,
            center_frames=hift_center_frames,
            exact_tail=hift_exact_tail,
            compact_tail=hift_compact_tail,
        )
        timings["hift_backend"] = "vulkan"
        timings["hift_window_frames"] = hift_window_frames
        timings["hift_center_frames"] = hift_center_frames
        timings["hift_exact_tail"] = hift_exact_tail
        timings["hift_compact_tail"] = hift_compact_tail
    else:
        with torch.inference_mode():
            wav = model.s3gen.mel2wav.decode(mels, source)
        timings["hift_backend"] = "cpu"
    timings["hift_decode_seconds"] = time.perf_counter() - hift_start
    wav = wav.clone()
    wav[:, : len(model.s3gen.trim_fade)] *= model.s3gen.trim_fade

    watermark_start = time.perf_counter()
    wav_np = wav.squeeze(0).detach().cpu().numpy()
    watermarked_wav = model.watermarker.apply_watermark(wav_np, sample_rate=model.sr)
    timings["watermark_seconds"] = time.perf_counter() - watermark_start
    timings["s3_and_watermark_seconds"] = (
        timings["s3_flow_seconds"]
        + timings["source_seconds"]
        + timings["hift_decode_seconds"]
        + timings["watermark_seconds"]
    )
    return torch.from_numpy(watermarked_wav).unsqueeze(0), timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--s3-timesteps", type=int, default=2)
    parser.add_argument("--vulkan-hift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hift-window-frames", type=int, default=128)
    parser.add_argument("--hift-center-frames", type=int, default=96)
    parser.add_argument("--hift-extra-frame-sizes", default="")
    parser.add_argument("--hift-exact-tail", action="store_true")
    parser.add_argument("--hift-compact-tail", action="store_true")
    parser.add_argument("--encoder-token-frames", type=int, default=605)
    parser.add_argument("--encoder-up-frames", type=int, default=1210)
    parser.add_argument("--estimator-frames", type=int, default=1210)
    parser.add_argument(
        "--load-s3-before-t3",
        action="store_true",
        help="Load S3/HIFT Vulkan modules before T3 generation instead of after it.",
    )
    parser.add_argument("--out-prefix", default="ggml_t3_hybrid_s3_hift_vulkan_api_path_chunk270_2026-07-08")
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
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    wav_path = OUT_DIR / f"{args.out_prefix}.wav"

    total_start = time.perf_counter()
    model_load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - model_load_start

    selected_text = select_text(args.case, args.text)
    text = punc_norm(selected_text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    t3_runtime_start = time.perf_counter()
    t3_runtime = T3GGMLVulkanRuntime()
    t3_runtime_load_seconds = time.perf_counter() - t3_runtime_start

    common_kwargs = {
        "t3_cond": model.conds.t3,
        "text_tokens": text_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_gen_len": args.max_gen_len,
    }

    encoder_chain = None
    estimator_chain = None
    split_hift = None
    encoder_runtime_load_seconds = None
    estimator_runtime_load_seconds = None
    hift_runtime_load_seconds = None

    def load_s3_runtimes() -> None:
        nonlocal encoder_chain, estimator_chain, split_hift
        nonlocal encoder_runtime_load_seconds, estimator_runtime_load_seconds, hift_runtime_load_seconds
        encoder_start = time.perf_counter()
        encoder_chain = VulkanEncoderChain(
            token_frames=args.encoder_token_frames,
            up_frames=args.encoder_up_frames,
        )
        encoder_runtime_load_seconds = time.perf_counter() - encoder_start
        estimator_start = time.perf_counter()
        estimator_chain = VulkanEstimatorChain(frames=args.estimator_frames)
        estimator_runtime_load_seconds = time.perf_counter() - estimator_start
        if args.vulkan_hift:
            hift_start = time.perf_counter()
            split_hift = SplitHiFTVulkan(
                model.s3gen.mel2wav,
                frame_sizes=parse_frame_sizes(args.hift_extra_frame_sizes, args.hift_window_frames),
                allow_padding=False,
            )
            hift_runtime_load_seconds = time.perf_counter() - hift_start

    if args.load_s3_before_t3:
        load_s3_runtimes()

    t3_start = time.perf_counter()
    vulkan_tokens = inference_turbo_vulkan(model.t3, t3_runtime, **common_kwargs)
    vulkan_t3_seconds = time.perf_counter() - t3_start

    if encoder_chain is None or estimator_chain is None:
        load_s3_runtimes()

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
    audio_seconds = wav_to_file(wav, model.sr, wav_path)
    total_seconds = time.perf_counter() - total_start
    warm_pipeline_seconds = vulkan_t3_seconds + s3_seconds

    result = {
        "description": "API-shaped isolated run with Vulkan T3, hybrid Vulkan S3 encoder+estimator, and Vulkan HiFT.",
        "case": args.case,
        "text": selected_text,
        "normalized_chars": len(text),
        "max_gen_len": args.max_gen_len,
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "bucket": {
            "encoder_token_frames": args.encoder_token_frames,
            "encoder_up_frames": args.encoder_up_frames,
            "estimator_frames": args.estimator_frames,
        },
        "device": t3_runtime.device,
        "model_load_seconds": model_load_seconds,
        "t3_runtime_load_seconds": t3_runtime_load_seconds,
        "s3_encoder_runtime_load_seconds": encoder_runtime_load_seconds,
        "s3_estimator_runtime_load_seconds": estimator_runtime_load_seconds,
        "hift_runtime_load_seconds": hift_runtime_load_seconds,
        "load_s3_before_t3": args.load_s3_before_t3,
        "vulkan_t3_seconds": vulkan_t3_seconds,
        "vulkan_t3_tokens": int(vulkan_tokens.numel()),
        "s3_and_watermark_seconds": s3_seconds,
        "s3_stage_breakdown": s3_timings,
        "warm_pipeline_seconds_excluding_load": warm_pipeline_seconds,
        "audio_seconds": audio_seconds,
        "total_seconds": total_seconds,
        "wav_path": wav_path.as_posix(),
        "notes": [
            "The live API remains CPU-only.",
            "This run uses Vulkan/RADV paths only for GPU acceleration.",
            "S3 encoder and estimator wrappers fall back to CPU if shapes differ from the fixed chunk270 bucket.",
        ],
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# ggml Vulkan T3 + Hybrid Vulkan S3 + Vulkan HiFT API-Shape Run",
        "",
        f"- Case: `{args.case}`",
        f"- Text chars: `{len(text)}`",
        f"- Max generated tokens: `{args.max_gen_len}`",
        f"- Device: `{t3_runtime.device}`",
        f"- Model load: `{model_load_seconds:.3f}s`",
        f"- T3 runtime load: `{t3_runtime_load_seconds:.3f}s`",
        f"- S3 encoder runtime load: `{encoder_runtime_load_seconds:.3f}s`",
        f"- S3 estimator runtime load: `{estimator_runtime_load_seconds:.3f}s`",
    ]
    if hift_runtime_load_seconds is not None:
        lines.append(f"- HiFT runtime load: `{hift_runtime_load_seconds:.3f}s`")
    lines.extend(
        [
            f"- Vulkan T3: `{vulkan_t3_seconds:.3f}s` for `{int(vulkan_tokens.numel())}` tokens",
            f"- S3 flow: `{s3_timings['s3_flow_seconds']:.3f}s`",
            f"- S3 encoder calls: `{s3_timings['encoder_calls']['vulkan']}` Vulkan, `{s3_timings['encoder_calls']['fallback_cpu']}` CPU fallback",
            f"- S3 estimator calls: `{s3_timings['estimator_calls']['vulkan']}` Vulkan, `{s3_timings['estimator_calls']['fallback_cpu']}` CPU fallback",
            f"- Source/F0: `{s3_timings['source_seconds']:.3f}s`",
            f"- HiFT backend: `{s3_timings['hift_backend']}`",
            f"- HiFT decode: `{s3_timings['hift_decode_seconds']:.3f}s`",
            f"- Watermark: `{s3_timings['watermark_seconds']:.3f}s`",
            f"- S3 + watermark: `{s3_seconds:.3f}s`",
            f"- Warm pipeline estimate excluding load: `{warm_pipeline_seconds:.3f}s`",
            f"- Audio duration: `{audio_seconds:.3f}s`",
            f"- Total isolated wall: `{total_seconds:.3f}s`",
            f"- WAV: `{wav_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"results={json_path}")
    print(f"wav={wav_path}")
    print(f"device={t3_runtime.device}")
    print(f"vulkan_t3_seconds={vulkan_t3_seconds:.3f}")
    print(f"s3_flow_seconds={s3_timings['s3_flow_seconds']:.3f}")
    print(
        "s3_encoder_calls="
        f"{s3_timings['encoder_calls']['vulkan']} vulkan/"
        f"{s3_timings['encoder_calls']['fallback_cpu']} fallback"
    )
    print(
        "s3_estimator_calls="
        f"{s3_timings['estimator_calls']['vulkan']} vulkan/"
        f"{s3_timings['estimator_calls']['fallback_cpu']} fallback"
    )
    print(f"source_seconds={s3_timings['source_seconds']:.3f}")
    print(f"hift_decode_seconds={s3_timings['hift_decode_seconds']:.3f}")
    print(f"watermark_seconds={s3_timings['watermark_seconds']:.3f}")
    print(f"s3_and_watermark_seconds={s3_seconds:.3f}")
    print(f"warm_pipeline_seconds_excluding_load={warm_pipeline_seconds:.3f}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
