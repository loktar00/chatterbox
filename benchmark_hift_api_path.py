#!/usr/bin/env python3
"""Benchmark CPU HiFT vs the API's opt-in Vulkan HiFT path on identical mels."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import chatterbox_api
from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from split_hift_vulkan_runtime import diff_summary


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "split_hift_vulkan"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000.0


def time_inference_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    with torch.inference_mode():
        value = fn()
    return value, (time.perf_counter() - started) * 1000.0


def trimmed(model: ChatterboxTurboTTS, wav: torch.Tensor) -> torch.Tensor:
    wav = wav.clone()
    wav[:, : len(model.s3gen.trim_fade)] *= model.s3gen.trim_fade
    return wav


def watermark(model: ChatterboxTurboTTS, wav: torch.Tensor) -> np.ndarray:
    wav_np = wav.squeeze(0).detach().cpu().numpy()
    return model.watermarker.apply_watermark(wav_np, sample_rate=model.sr)


def benchmark_case(text: str, seed: int, threads: int, interop_threads: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    torch.manual_seed(seed)

    model, load_ms = time_call(lambda: ChatterboxTurboTTS.from_pretrained("cpu"))
    split_hift, split_load_ms = time_call(lambda: chatterbox_api.get_vulkan_hift(model))

    normalized = punc_norm(text)
    text_tokens = model.tokenizer(normalized, return_tensors="pt", padding=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    speech_tokens, t3_ms = time_call(
        lambda: model.t3.inference_turbo(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            temperature=0.8,
            top_k=1000,
            top_p=0.95,
            repetition_penalty=1.2,
        )
    )
    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    mels, flow_ms = time_call(
        lambda: model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)
    )

    source, source_ms = time_inference_call(lambda: split_hift._source_from_speech_feat(mels))
    cpu_wav, cpu_decode_ms = time_inference_call(lambda: model.s3gen.mel2wav.decode(mels, source))
    vulkan_wav, vulkan_decode_ms = time_inference_call(
        lambda: chatterbox_api.vulkan_hift_decode(split_hift, mels, source)
    )

    cpu_trimmed = trimmed(model, cpu_wav)
    vulkan_trimmed = trimmed(model, vulkan_wav)
    _watermark_warmup, watermark_warmup_ms = time_call(lambda: watermark(model, cpu_trimmed))
    _cpu_watermarked, cpu_watermark_ms = time_call(lambda: watermark(model, cpu_trimmed))
    _vulkan_watermarked, vulkan_watermark_ms = time_call(lambda: watermark(model, vulkan_trimmed))

    cpu_hift_total_ms = source_ms + cpu_decode_ms + cpu_watermark_ms
    vulkan_hift_total_ms = source_ms + vulkan_decode_ms + vulkan_watermark_ms
    cpu_pipeline_ms = t3_ms + flow_ms + cpu_hift_total_ms
    vulkan_pipeline_ms = t3_ms + flow_ms + vulkan_hift_total_ms

    return {
        "seed": seed,
        "chars": len(text),
        "normalized_text": normalized,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "model_load_ms": load_ms,
        "vulkan_hift_load_ms": split_load_ms,
        "speech_token_count_with_silence": int(speech_tokens.numel()),
        "mel_shape": list(mels.shape),
        "mel_frames": int(mels.shape[-1]),
        "samples_per_frame": split_hift.samples_per_mel_frame,
        "audio_samples": int(cpu_trimmed.shape[-1]),
        "audio_seconds_at_24k": float(cpu_trimmed.shape[-1] / 24000.0),
        "stage_ms": {
            "t3": t3_ms,
            "s3_flow": flow_ms,
            "source_f0": source_ms,
            "cpu_hift_decode_same_source": cpu_decode_ms,
            "vulkan_hift_decode_same_source": vulkan_decode_ms,
            "watermark_warmup": watermark_warmup_ms,
            "cpu_watermark": cpu_watermark_ms,
            "vulkan_watermark": vulkan_watermark_ms,
        },
        "totals_ms": {
            "cpu_hift_stage": cpu_hift_total_ms,
            "vulkan_hift_stage": vulkan_hift_total_ms,
            "cpu_pipeline_estimate": cpu_pipeline_ms,
            "vulkan_hift_pipeline_estimate": vulkan_pipeline_ms,
            "estimated_wall_savings_ms": cpu_pipeline_ms - vulkan_pipeline_ms,
            "estimated_pipeline_speedup": cpu_pipeline_ms / vulkan_pipeline_ms,
            "hift_stage_speedup": cpu_hift_total_ms / vulkan_hift_total_ms,
            "decode_only_speedup": cpu_decode_ms / vulkan_decode_ms,
        },
        "vulkan_vs_cpu_trimmed_diff": diff_summary(vulkan_trimmed, cpu_trimmed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["short", "chunk270"], default="short")
    parser.add_argument("--text")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "vulkan_hift_stage_benchmark_latest.json",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.case == "chunk270":
        text = CHUNK_270
    else:
        text = SHORT_TEXT

    result = benchmark_case(text, args.seed, args.threads, args.interop_threads)
    result["case"] = args.case
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    totals = result["totals_ms"]
    stage = result["stage_ms"]
    print(
        f"{args.case}: frames={result['mel_frames']} audio={result['audio_seconds_at_24k']:.3f}s "
        f"t3={stage['t3'] / 1000:.3f}s flow={stage['s3_flow'] / 1000:.3f}s "
        f"cpu_hift={totals['cpu_hift_stage'] / 1000:.3f}s "
        f"vulkan_hift={totals['vulkan_hift_stage'] / 1000:.3f}s "
        f"pipeline_speedup={totals['estimated_pipeline_speedup']:.3f}x"
    )
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
