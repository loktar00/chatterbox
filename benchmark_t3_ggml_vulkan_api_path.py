#!/usr/bin/env python3
"""Short isolated API-shape smoke benchmark for experimental ggml/Vulkan T3."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from split_hift_vulkan_runtime import SplitHiFTVulkan
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"
DEFAULT_TEXT = "Hello world, this is a short Vulkan T3 smoke test."


def wav_bytes_to_file(wav: torch.Tensor, sample_rate: int, path: Path) -> float:
    arr = wav.detach().cpu().squeeze().numpy()
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    sf.write(path, arr, sample_rate, format="WAV")
    return float(arr.shape[0] / sample_rate)


def source_from_speech_feat(mel2wav, speech_feat: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        f0 = mel2wav.f0_predictor(speech_feat)
        source = mel2wav.f0_upsamp(f0[:, None]).transpose(1, 2)
        source, _, _ = mel2wav.m_source(source)
        return source.transpose(1, 2).contiguous()


def choose_hift_window(total_frames: int, center_start: int, window_frames: int, center_frames: int) -> tuple[int, int]:
    center_end = min(center_start + center_frames, total_frames)
    if total_frames <= window_frames:
        return 0, total_frames
    start = center_start - (window_frames - (center_end - center_start)) // 2
    start = max(0, min(start, total_frames - window_frames))
    return start, start + window_frames


def parse_frame_sizes(spec: str, required: int) -> list[int]:
    sizes = {required}
    for part in spec.split(","):
        part = part.strip()
        if part:
            sizes.add(int(part))
    return sorted(sizes)


def vulkan_hift_decode(
    split_hift: SplitHiFTVulkan,
    mels: torch.Tensor,
    source: torch.Tensor,
    *,
    window_frames: int,
    center_frames: int,
    exact_tail: bool = False,
    compact_tail: bool = False,
) -> torch.Tensor:
    total_frames = int(mels.shape[-1])
    if total_frames <= window_frames and total_frames != window_frames:
        if not (exact_tail and total_frames in split_hift.supported_frame_sizes):
            raise ValueError(
                f"No exact Vulkan HiFT VMFB for {total_frames} frames; window={window_frames}"
            )

    samples_per_frame = split_hift.samples_per_mel_frame
    outputs = []
    for center_start in range(0, total_frames, center_frames):
        center_end = min(center_start + center_frames, total_frames)
        remaining_frames = total_frames - center_start
        if (
            exact_tail
            and remaining_frames < window_frames
            and remaining_frames in split_hift.supported_frame_sizes
        ):
            win_start, win_end = center_start, total_frames
        elif compact_tail and center_start > 0 and remaining_frames < window_frames:
            tail_candidates = [
                frames
                for frames in split_hift.supported_frame_sizes
                if remaining_frames <= frames < window_frames
            ]
            if tail_candidates:
                tail_frames = min(tail_candidates)
                win_start, win_end = total_frames - tail_frames, total_frames
            else:
                win_start, win_end = choose_hift_window(total_frames, center_start, window_frames, center_frames)
        else:
            win_start, win_end = choose_hift_window(total_frames, center_start, window_frames, center_frames)
        mel_win = mels[..., win_start:win_end].contiguous()
        source_win = source[..., win_start * samples_per_frame : win_end * samples_per_frame].contiguous()
        wav_win = split_hift.decode_from_source(mel_win, source_win)

        crop_start = (center_start - win_start) * samples_per_frame
        crop_end = (center_end - win_start) * samples_per_frame
        outputs.append(wav_win[..., crop_start:crop_end].contiguous())
    return torch.cat(outputs, dim=-1)


def run_s3_and_watermark(
    model: ChatterboxTurboTTS,
    speech_tokens: torch.Tensor,
    *,
    split_hift: SplitHiFTVulkan | None = None,
    hift_window_frames: int = 128,
    hift_center_frames: int = 96,
    hift_exact_tail: bool = False,
    hift_compact_tail: bool = False,
) -> tuple[torch.Tensor, dict]:
    timings = {}
    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    flow_start = time.perf_counter()
    mels = model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=2,
        finalize=True,
    ).to(dtype=model.s3gen.dtype)
    timings["s3_flow_seconds"] = time.perf_counter() - flow_start
    timings["mel_frames"] = int(mels.shape[-1])
    timings["speech_token_count_with_silence"] = int(speech_tokens.numel())

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
    parser.add_argument("--case", choices=("custom", "hello", "short", "chunk270"), default="custom")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--max-gen-len", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--vulkan-hift", action="store_true")
    parser.add_argument("--hift-window-frames", type=int, default=128)
    parser.add_argument("--hift-center-frames", type=int, default=96)
    parser.add_argument("--hift-extra-frame-sizes", default="")
    parser.add_argument("--hift-exact-tail", action="store_true")
    parser.add_argument("--hift-compact-tail", action="store_true")
    parser.add_argument("--out-prefix", default="ggml_t3_api_path_smoke_2026-07-08")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"
    wav_path = OUT_DIR / f"{args.out_prefix}.wav"

    total_start = time.perf_counter()
    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_seconds = time.perf_counter() - load_start

    selected_text = {
        "custom": args.text,
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[args.case]
    text = punc_norm(selected_text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    common_kwargs = {
        "t3_cond": model.conds.t3,
        "text_tokens": text_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_gen_len": args.max_gen_len,
    }

    cpu_tokens = None
    cpu_t3_seconds = None
    if args.compare_cpu:
        cpu_start = time.perf_counter()
        cpu_tokens = model.t3.inference_turbo(**common_kwargs)
        cpu_t3_seconds = time.perf_counter() - cpu_start

    runtime_start = time.perf_counter()
    runtime = T3GGMLVulkanRuntime()
    runtime_load_seconds = time.perf_counter() - runtime_start

    vulkan_start = time.perf_counter()
    vulkan_tokens = inference_turbo_vulkan(model.t3, runtime, **common_kwargs)
    vulkan_t3_seconds = time.perf_counter() - vulkan_start

    split_hift = None
    hift_runtime_load_seconds = None
    if args.vulkan_hift:
        hift_load_start = time.perf_counter()
        split_hift = SplitHiFTVulkan(
            model.s3gen.mel2wav,
            frame_sizes=parse_frame_sizes(args.hift_extra_frame_sizes, args.hift_window_frames),
            allow_padding=False,
        )
        hift_runtime_load_seconds = time.perf_counter() - hift_load_start

    s3_start = time.perf_counter()
    wav, s3_timings = run_s3_and_watermark(
        model,
        vulkan_tokens,
        split_hift=split_hift,
        hift_window_frames=args.hift_window_frames,
        hift_center_frames=args.hift_center_frames,
        hift_exact_tail=args.hift_exact_tail,
        hift_compact_tail=args.hift_compact_tail,
    )
    s3_seconds = time.perf_counter() - s3_start
    audio_seconds = wav_bytes_to_file(wav, model.sr, wav_path)
    total_seconds = time.perf_counter() - total_start
    warm_pipeline_seconds = vulkan_t3_seconds + s3_seconds

    tokens_match = None
    cpu_token_count = None
    if cpu_tokens is not None:
        cpu_token_count = int(cpu_tokens.numel())
        tokens_match = bool(torch.equal(cpu_tokens.cpu(), vulkan_tokens.cpu()))

    result = {
        "description": "Short isolated API-shape smoke benchmark for experimental ggml Vulkan T3.",
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
        "device": runtime.device,
        "load_seconds": load_seconds,
        "runtime_load_seconds": runtime_load_seconds,
        "hift_runtime_load_seconds": hift_runtime_load_seconds,
        "cpu_t3_seconds": cpu_t3_seconds,
        "vulkan_t3_seconds": vulkan_t3_seconds,
        "vulkan_t3_tokens": int(vulkan_tokens.numel()),
        "cpu_t3_tokens": cpu_token_count,
        "tokens_match": tokens_match,
        "s3_and_watermark_seconds": s3_seconds,
        "s3_stage_breakdown": s3_timings,
        "warm_pipeline_seconds_excluding_load": warm_pipeline_seconds,
        "audio_seconds": audio_seconds,
        "total_seconds": total_seconds,
        "wav_path": str(wav_path),
    }
    if cpu_t3_seconds is not None and vulkan_t3_seconds > 0:
        result["t3_speedup_cpu_vs_vulkan"] = cpu_t3_seconds / vulkan_t3_seconds

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# ggml Vulkan T3 API-Shape Smoke",
        "",
        f"- Case: `{args.case}`",
        f"- Text chars: `{len(text)}`",
        f"- Max generated tokens: `{args.max_gen_len}`",
        f"- Sampling: `temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}`",
        f"- Device: `{runtime.device}`",
        f"- Model load: `{load_seconds:.3f}s`",
        f"- Runtime load: `{runtime_load_seconds:.3f}s`",
        f"- HiFT backend: `{s3_timings['hift_backend']}`",
    ]
    if hift_runtime_load_seconds is not None:
        lines.append(f"- HiFT runtime load: `{hift_runtime_load_seconds:.3f}s`")
    if cpu_t3_seconds is not None:
        lines.append(f"- CPU T3: `{cpu_t3_seconds:.3f}s` for `{cpu_token_count}` tokens")
        lines.append(f"- Vulkan T3: `{vulkan_t3_seconds:.3f}s` for `{int(vulkan_tokens.numel())}` tokens")
        lines.append(f"- T3 speedup: `{cpu_t3_seconds / vulkan_t3_seconds:.2f}x`")
        lines.append(f"- Tokens match: `{tokens_match}`")
    else:
        lines.append(f"- Vulkan T3: `{vulkan_t3_seconds:.3f}s` for `{int(vulkan_tokens.numel())}` tokens")
    lines.extend(
        [
            f"- S3 + watermark: `{s3_seconds:.3f}s`",
            f"- S3 flow: `{s3_timings['s3_flow_seconds']:.3f}s`",
            f"- Source/F0: `{s3_timings['source_seconds']:.3f}s`",
            f"- HiFT decode: `{s3_timings['hift_decode_seconds']:.3f}s`",
            f"- Watermark: `{s3_timings['watermark_seconds']:.3f}s`",
            f"- Warm pipeline estimate excluding load: `{warm_pipeline_seconds:.3f}s`",
            f"- Audio duration: `{audio_seconds:.3f}s`",
            f"- Total isolated wall: `{total_seconds:.3f}s`",
            f"- WAV: `{wav_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"device={runtime.device}")
    print(f"vulkan_t3_seconds={vulkan_t3_seconds:.3f}")
    if cpu_t3_seconds is not None:
        print(f"cpu_t3_seconds={cpu_t3_seconds:.3f}")
        print(f"t3_speedup={cpu_t3_seconds / vulkan_t3_seconds:.2f}x")
        print(f"tokens_match={tokens_match}")
    print(f"s3_and_watermark_seconds={s3_seconds:.3f}")
    print(f"s3_flow_seconds={s3_timings['s3_flow_seconds']:.3f}")
    print(f"source_seconds={s3_timings['source_seconds']:.3f}")
    print(f"hift_backend={s3_timings['hift_backend']}")
    print(f"hift_decode_seconds={s3_timings['hift_decode_seconds']:.3f}")
    print(f"watermark_seconds={s3_timings['watermark_seconds']:.3f}")
    print(f"warm_pipeline_seconds_excluding_load={warm_pipeline_seconds:.3f}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f"wrote={json_path}")
    print(f"wrote={md_path}")
    print(f"wav={wav_path}")


if __name__ == "__main__":
    main()
