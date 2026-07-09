#!/usr/bin/env python3
"""Probe chunked HiFT decoding accuracy for real Chatterbox Turbo mels."""

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
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from split_hift_vulkan_runtime import SplitHiFTVulkan, diff_summary


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "hift_chunking"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_call(fn) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000.0


def generate_mels(model: ChatterboxTurboTTS, text: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
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

    meta = {
        "normalized_text": text,
        "text_token_shape": list(text_tokens.shape),
        "speech_token_count": int(speech_tokens.numel()),
        "mel_shape": list(mels.shape),
        "mel_frames": int(mels.shape[-1]),
        "t3_ms": t3_ms,
        "flow_ms": flow_ms,
    }
    return mels, speech_tokens, meta


def choose_window(total_frames: int, center_start: int, center_end: int, window_frames: int) -> tuple[int, int]:
    if total_frames <= window_frames:
        return 0, total_frames
    start = center_start - (window_frames - (center_end - center_start)) // 2
    start = max(0, min(start, total_frames - window_frames))
    return start, start + window_frames


def chunked_decode(
    mels: torch.Tensor,
    source: torch.Tensor,
    decode_fn,
    *,
    center_frames: int,
    window_frames: int,
    samples_per_frame: int,
) -> tuple[torch.Tensor, float, list[dict[str, int]]]:
    total_frames = int(mels.shape[-1])
    outputs = []
    chunks = []
    total_ms = 0.0

    for center_start in range(0, total_frames, center_frames):
        center_end = min(center_start + center_frames, total_frames)
        win_start, win_end = choose_window(total_frames, center_start, center_end, window_frames)
        mel_win = mels[..., win_start:win_end].contiguous()
        source_win = source[..., win_start * samples_per_frame : win_end * samples_per_frame].contiguous()
        wav_win, elapsed_ms = time_call(lambda: decode_fn(mel_win, source_win))
        total_ms += elapsed_ms

        crop_start = (center_start - win_start) * samples_per_frame
        crop_end = (center_end - win_start) * samples_per_frame
        outputs.append(wav_win[..., crop_start:crop_end].contiguous())
        chunks.append(
            {
                "center_start": center_start,
                "center_end": center_end,
                "window_start": win_start,
                "window_end": win_end,
                "window_frames": win_end - win_start,
                "crop_start_samples": crop_start,
                "crop_end_samples": crop_end,
            }
        )

    return torch.cat(outputs, dim=-1), total_ms, chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["short", "chunk270"], default="chunk270")
    parser.add_argument("--window-frames", type=int, default=128)
    parser.add_argument("--center-frames", default="128,64", help="Comma-separated center step sizes.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "hift_chunking_latest.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    text = SHORT_TEXT if args.case == "short" else CHUNK_270
    center_sizes = [int(part) for part in args.center_frames.split(",") if part.strip()]

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_ms = (time.perf_counter() - load_started) * 1000.0
    mel2wav = model.s3gen.mel2wav
    split = SplitHiFTVulkan(mel2wav, frame_sizes=[args.window_frames])
    samples_per_frame = split.samples_per_mel_frame

    with torch.inference_mode():
        mels, _speech_tokens, gen_meta = generate_mels(model, text)
        source, source_ms = time_call(lambda: split._source_from_speech_feat(mels))
        full_cpu, full_cpu_ms = time_call(lambda: mel2wav.decode(mels, source))

        cases = []
        for center_frames in center_sizes:
            cpu_chunked, cpu_chunk_ms, cpu_chunks = chunked_decode(
                mels,
                source,
                lambda mel_win, source_win: mel2wav.decode(mel_win, source_win),
                center_frames=center_frames,
                window_frames=args.window_frames,
                samples_per_frame=samples_per_frame,
            )
            split_chunked, split_chunk_ms, split_chunks = chunked_decode(
                mels,
                source,
                lambda mel_win, source_win: split.decode_from_source(mel_win, source_win),
                center_frames=center_frames,
                window_frames=args.window_frames,
                samples_per_frame=samples_per_frame,
            )
            cases.append(
                {
                    "center_frames": center_frames,
                    "window_frames": args.window_frames,
                    "chunk_count": len(cpu_chunks),
                    "cpu_chunked_decode_ms": cpu_chunk_ms,
                    "split_chunked_decode_ms": split_chunk_ms,
                    "cpu_chunked_vs_full": diff_summary(cpu_chunked, full_cpu),
                    "split_chunked_vs_full": diff_summary(split_chunked, full_cpu),
                    "split_vs_cpu_chunked": diff_summary(split_chunked, cpu_chunked),
                    "chunks": split_chunks,
                }
            )

    results = {
        "case": args.case,
        "chars": len(text),
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "model_load_ms": load_ms,
        "generation": gen_meta,
        "samples_per_frame": samples_per_frame,
        "source_generation_ms": source_ms,
        "full_cpu_decode_ms": full_cpu_ms,
        "full_audio_samples": int(full_cpu.shape[-1]),
        "full_audio_seconds_at_24k": float(full_cpu.shape[-1] / 24000.0),
        "chunk_cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"{args.case}: frames={gen_meta['mel_frames']} samples={results['full_audio_samples']} "
        f"full_cpu_decode={full_cpu_ms:.1f}ms"
    )
    for case in cases:
        split_diff = case["split_chunked_vs_full"]
        print(
            f"center={case['center_frames']} window={case['window_frames']} "
            f"chunks={case['chunk_count']} cpu_chunk={case['cpu_chunked_decode_ms']:.1f}ms "
            f"split_chunk={case['split_chunked_decode_ms']:.1f}ms "
            f"split_max={split_diff['max_abs_error']:.3e} "
            f"split_mean={split_diff['mean_abs_error']:.3e}"
        )
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
