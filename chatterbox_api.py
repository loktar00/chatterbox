import base64
import logging
import os
import tempfile
import threading
import time
from io import BytesIO
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = int(os.getenv("CHATTERBOX_MAX_INPUT_CHARS", "3000"))


def pick_device() -> str:
    requested = os.getenv("CHATTERBOX_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_frame_sizes(spec: str, required: int) -> list[int]:
    sizes = {required}
    for part in spec.split(","):
        part = part.strip()
        if part:
            sizes.add(int(part))
    return sorted(sizes)


THREADS = int(os.getenv("CHATTERBOX_TORCH_THREADS", "0") or "0")
if THREADS > 0:
    torch.set_num_threads(THREADS)

INTEROP_THREADS = int(os.getenv("CHATTERBOX_TORCH_INTEROP_THREADS", "0") or "0")
if INTEROP_THREADS > 0:
    torch.set_num_interop_threads(INTEROP_THREADS)

DEVICE = pick_device()
MODEL = None
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
VULKAN_HIFT = None
VULKAN_HIFT_LOCK = threading.Lock()
VULKAN_T3 = None
VULKAN_T3_LOCK = threading.Lock()
VULKAN_S3_CHAINS = {}
VULKAN_S3_LOCK = threading.Lock()
LAST_REQUEST_TIMING = None
LAST_REQUEST_TIMING_LOCK = threading.Lock()

EXPERIMENTAL_VULKAN_HIFT = os.getenv("CHATTERBOX_EXPERIMENTAL_VULKAN_HIFT", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUIRE_VULKAN_HIFT = os.getenv("CHATTERBOX_REQUIRE_VULKAN_HIFT", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HIFT_WINDOW_FRAMES = int(os.getenv("CHATTERBOX_HIFT_WINDOW_FRAMES", "128"))
HIFT_CENTER_FRAMES = int(os.getenv("CHATTERBOX_HIFT_CENTER_FRAMES", "96"))
HIFT_FRAME_SIZES = parse_frame_sizes(os.getenv("CHATTERBOX_HIFT_EXTRA_FRAME_SIZES", ""), HIFT_WINDOW_FRAMES)
HIFT_EXACT_TAIL = os.getenv("CHATTERBOX_HIFT_EXACT_TAIL", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HIFT_COMPACT_TAIL = os.getenv("CHATTERBOX_HIFT_COMPACT_TAIL", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
APPLY_WATERMARK = os.getenv("CHATTERBOX_APPLY_WATERMARK", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
T3_NATIVE_SAMPLER = os.getenv("CHATTERBOX_T3_NATIVE_SAMPLER", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
EXPERIMENTAL_VULKAN_T3 = os.getenv("CHATTERBOX_EXPERIMENTAL_VULKAN_T3", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUIRE_VULKAN_T3 = os.getenv("CHATTERBOX_REQUIRE_VULKAN_T3", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
T3_GGML_WEIGHTS_DIR = os.getenv("CHATTERBOX_T3_GGML_WEIGHTS_DIR", "")
T3_GGML_LIB_PATH = os.getenv("CHATTERBOX_T3_GGML_LIB_PATH", "")
T3_GGML_DEVICE_INDEX = int(os.getenv("CHATTERBOX_T3_GGML_DEVICE_INDEX", "0"))
T3_MAX_GEN_LEN = int(os.getenv("CHATTERBOX_T3_MAX_GEN_LEN", "1000"))
T3_PREFIX_PREFILL = os.getenv("CHATTERBOX_T3_PREFIX_PREFILL", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_SEED = int(os.getenv("CHATTERBOX_DEFAULT_SEED", "0") or "0")
EXPERIMENTAL_VULKAN_S3 = os.getenv("CHATTERBOX_EXPERIMENTAL_VULKAN_S3", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REQUIRE_VULKAN_S3 = os.getenv("CHATTERBOX_REQUIRE_VULKAN_S3", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ALLOW_VULKAN_S3_PADDING = os.getenv("CHATTERBOX_ALLOW_VULKAN_S3_PADDING", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VULKAN_S3_FUSED_MIDBLOCKS = os.getenv("CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VULKAN_S3_FUSED_VARIANT = os.getenv("CHATTERBOX_VULKAN_S3_FUSED_VARIANT", "split8")
VULKAN_S3_BUCKET_SPEC = os.getenv("CHATTERBOX_S3_BUCKETS", "605:1210,611:1222,615:1230,629:1258")
S3_TIMESTEPS = int(os.getenv("CHATTERBOX_S3_TIMESTEPS", "2"))
STARTUP_WARMUP = os.getenv("CHATTERBOX_STARTUP_WARMUP", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STARTUP_WARMUP_S3_BUCKETS = os.getenv("CHATTERBOX_STARTUP_WARMUP_S3_BUCKETS", "")
STARTUP_WARMUP_T3_PREFIX = os.getenv("CHATTERBOX_STARTUP_WARMUP_T3_PREFIX", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="Chatterbox Turbo API", version="0.1.0")


def set_last_request_timing(timing: dict[str, Any]) -> None:
    global LAST_REQUEST_TIMING
    with LAST_REQUEST_TIMING_LOCK:
        LAST_REQUEST_TIMING = timing


def get_last_request_timing() -> dict[str, Any] | None:
    with LAST_REQUEST_TIMING_LOCK:
        return dict(LAST_REQUEST_TIMING) if LAST_REQUEST_TIMING is not None else None


def warmup_s3_bucket_keys(spec: str) -> list[int]:
    if not spec.strip():
        return []
    if spec.strip().lower() == "all":
        return sorted(VULKAN_S3_BUCKETS)
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    audio_prompt_path: Optional[str] = None
    exaggeration: float = 0.0
    cfg_weight: float = 0.0
    temperature: float = 0.8
    min_p: float = 0.0
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True
    seed: int = 0


def get_model() -> ChatterboxTurboTTS:
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                MODEL = ChatterboxTurboTTS.from_pretrained(DEVICE)
    return MODEL


def wav_bytes(wav: torch.Tensor, sample_rate: int) -> bytes:
    arr = wav.detach().cpu().squeeze().numpy()
    if arr.ndim != 1:
        arr = np.asarray(arr).reshape(-1)
    buf = BytesIO()
    sf.write(buf, arr, sample_rate, format="WAV")
    return buf.getvalue()


def maybe_apply_watermark(
    model: ChatterboxTurboTTS,
    wav: torch.Tensor,
    timings: dict[str, Any] | None = None,
) -> torch.Tensor:
    watermark_start = time.perf_counter()
    wav_np = wav.squeeze(0).detach().cpu().numpy()
    if APPLY_WATERMARK:
        output_np = model.watermarker.apply_watermark(wav_np, sample_rate=model.sr)
    else:
        output_np = wav_np
    if timings is not None:
        timings["watermark_applied"] = APPLY_WATERMARK
        timings["watermark_seconds"] = time.perf_counter() - watermark_start
        timings["audio_samples"] = int(output_np.shape[-1])
        timings["audio_seconds"] = float(output_np.shape[-1] / model.sr)
    return torch.from_numpy(output_np).unsqueeze(0)


def get_vulkan_hift(model: ChatterboxTurboTTS):
    global VULKAN_HIFT
    if VULKAN_HIFT is None:
        with VULKAN_HIFT_LOCK:
            if VULKAN_HIFT is None:
                from split_hift_vulkan_runtime import SplitHiFTVulkan

                VULKAN_HIFT = SplitHiFTVulkan(
                    model.s3gen.mel2wav,
                    frame_sizes=HIFT_FRAME_SIZES,
                    allow_padding=False,
                )
    return VULKAN_HIFT


def get_vulkan_t3():
    global VULKAN_T3
    if VULKAN_T3 is None:
        with VULKAN_T3_LOCK:
            if VULKAN_T3 is None:
                from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime

                kwargs = {"device_index": T3_GGML_DEVICE_INDEX}
                if T3_GGML_WEIGHTS_DIR:
                    kwargs["weights_dir"] = T3_GGML_WEIGHTS_DIR
                if T3_GGML_LIB_PATH:
                    kwargs["lib_path"] = T3_GGML_LIB_PATH
                VULKAN_T3 = T3GGMLVulkanRuntime(**kwargs)
                logger.info("Loaded experimental ggml Vulkan T3 runtime on %s", VULKAN_T3.device)
    return VULKAN_T3


def parse_vulkan_s3_buckets(spec: str) -> dict[int, tuple[int, int, int]]:
    buckets: dict[int, tuple[int, int, int]] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) == 2:
            token_frames = int(pieces[0])
            up_frames = int(pieces[1])
            estimator_frames = up_frames
        elif len(pieces) == 3:
            token_frames = int(pieces[0])
            up_frames = int(pieces[1])
            estimator_frames = int(pieces[2])
        else:
            raise ValueError(f"Invalid CHATTERBOX_S3_BUCKETS entry {part!r}")
        buckets[token_frames] = (token_frames, up_frames, estimator_frames)
    return buckets


VULKAN_S3_BUCKETS = parse_vulkan_s3_buckets(VULKAN_S3_BUCKET_SPEC)


def prompt_token_len(model: ChatterboxTurboTTS) -> int:
    value = model.conds.gen["prompt_token_len"]
    if torch.is_tensor(value):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(value).reshape(-1)[0])


def infer_vulkan_s3_bucket(model: ChatterboxTurboTTS, speech_tokens: torch.Tensor) -> tuple[tuple[int, int, int] | None, dict]:
    valid_speech_tokens = int((speech_tokens.reshape(-1) < 6561).sum().item())
    speech_tokens_with_silence = valid_speech_tokens + 3
    prompt_tokens = prompt_token_len(model)
    encoder_token_frames = prompt_tokens + speech_tokens_with_silence
    required_up_frames = encoder_token_frames * 2
    if ALLOW_VULKAN_S3_PADDING:
        bucket = next(
            (
                candidate
                for candidate in sorted(VULKAN_S3_BUCKETS.values(), key=lambda item: (item[0], item[1], item[2]))
                if (
                    candidate[0] >= encoder_token_frames
                    and candidate[1] >= required_up_frames
                    and candidate[2] >= required_up_frames
                )
            ),
            None,
        )
    else:
        bucket = VULKAN_S3_BUCKETS.get(encoder_token_frames)
    return bucket, {
        "valid_speech_tokens": valid_speech_tokens,
        "speech_tokens_with_silence": speech_tokens_with_silence,
        "prompt_tokens": prompt_tokens,
        "encoder_token_frames": encoder_token_frames,
        "encoder_up_frames": required_up_frames,
        "estimator_frames": required_up_frames,
        "selected_bucket": (
            {
                "encoder_token_frames": bucket[0],
                "encoder_up_frames": bucket[1],
                "estimator_frames": bucket[2],
                "padding": bucket[0] != encoder_token_frames or bucket[1] != required_up_frames,
            }
            if bucket is not None
            else None
        ),
    }


def missing_vulkan_s3_artifacts(bucket: tuple[int, int, int]) -> list[str]:
    from benchmark_s3_encoder_iree_runtime_chain import encoder_module_names, vmfb_path
    from benchmark_s3_estimator_distinct_iree_runtime_chain import distinct_vmfb, helper_vmfb, module_names
    from benchmark_s3_flow_hybrid_vulkan_estimator import fused_midblock_vmfb

    token_frames, up_frames, estimator_frames = bucket
    missing = [
        vmfb_path(name).as_posix()
        for name in encoder_module_names(token_frames, up_frames).values()
        if not vmfb_path(name).exists()
    ]
    missing.extend(
        distinct_vmfb(name).as_posix()
        for name in module_names(estimator_frames).values()
        if not distinct_vmfb(name).exists()
    )
    helper_names = (
        f"s3_flow_transpose_c256_t{estimator_frames}",
        f"s3_flow_transpose_t{estimator_frames}_c256",
        f"s3_flow_cat_channel_256_256_t{estimator_frames}",
    )
    missing.extend(helper_vmfb(name).as_posix() for name in helper_names if not helper_vmfb(name).exists())
    if VULKAN_S3_FUSED_MIDBLOCKS:
        missing.extend(
            fused_midblock_vmfb(estimator_frames, mid, VULKAN_S3_FUSED_VARIANT).as_posix()
            for mid in range(12)
            if not fused_midblock_vmfb(estimator_frames, mid, VULKAN_S3_FUSED_VARIANT).exists()
        )
    return missing


def get_vulkan_s3_chains(bucket: tuple[int, int, int]):
    missing = missing_vulkan_s3_artifacts(bucket)
    if missing:
        raise RuntimeError("Missing Vulkan S3 bucket artifacts:\n" + "\n".join(missing[:20]))

    with VULKAN_S3_LOCK:
        if bucket not in VULKAN_S3_CHAINS:
            from benchmark_s3_flow_hybrid_vulkan_encoder_estimator import VulkanEncoderChain
            from benchmark_s3_flow_hybrid_vulkan_estimator import (
                VulkanEstimatorChain,
                VulkanFusedMidblockEstimatorChain,
            )

            token_frames, up_frames, estimator_frames = bucket
            estimator_chain_cls = (
                VulkanFusedMidblockEstimatorChain
                if VULKAN_S3_FUSED_MIDBLOCKS
                else VulkanEstimatorChain
            )
            estimator_kwargs = (
                {"frames": estimator_frames, "variant": VULKAN_S3_FUSED_VARIANT}
                if VULKAN_S3_FUSED_MIDBLOCKS
                else {"frames": estimator_frames}
            )
            VULKAN_S3_CHAINS[bucket] = (
                VulkanEncoderChain(token_frames=token_frames, up_frames=up_frames),
                estimator_chain_cls(**estimator_kwargs),
            )
            logger.info(
                "Loaded experimental Vulkan S3 bucket token=%s up=%s estimator=%s fused_midblocks=%s variant=%s",
                token_frames,
                up_frames,
                estimator_frames,
                VULKAN_S3_FUSED_MIDBLOCKS,
                VULKAN_S3_FUSED_VARIANT if VULKAN_S3_FUSED_MIDBLOCKS else None,
            )
        return VULKAN_S3_CHAINS[bucket]


@app.on_event("startup")
def startup_warmup() -> None:
    if not STARTUP_WARMUP:
        return
    started = time.perf_counter()
    report: dict[str, Any] = {"startup_warmup": True}
    model = get_model()
    report["model_loaded"] = True
    if EXPERIMENTAL_VULKAN_T3:
        t3_start = time.perf_counter()
        t3_runtime = get_vulkan_t3()
        report["t3_runtime_load_seconds"] = time.perf_counter() - t3_start
        if T3_PREFIX_PREFILL and STARTUP_WARMUP_T3_PREFIX:
            prefix_start = time.perf_counter()
            try:
                from t3_ggml_vulkan_runtime import warm_prefix_prefill_cache

                text_tokens = model.tokenizer(
                    punc_norm("Warmup."),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(model.device)
                report["t3_prefix_warmup"] = warm_prefix_prefill_cache(
                    model.t3,
                    t3_runtime,
                    t3_cond=model.conds.t3,
                    text_tokens=text_tokens,
                )
            except Exception as exc:
                report["t3_prefix_warmup_error"] = str(exc)
                logger.exception("Startup T3 prefix warmup failed; continuing without prebuilt prefix cache")
            finally:
                report["t3_prefix_warmup_seconds"] = time.perf_counter() - prefix_start
    if EXPERIMENTAL_VULKAN_HIFT:
        hift_start = time.perf_counter()
        get_vulkan_hift(model)
        report["hift_runtime_load_seconds"] = time.perf_counter() - hift_start
    if EXPERIMENTAL_VULKAN_S3:
        loaded_buckets = []
        for key in warmup_s3_bucket_keys(STARTUP_WARMUP_S3_BUCKETS):
            bucket = VULKAN_S3_BUCKETS.get(key)
            if bucket is None:
                logger.warning("Startup warmup requested unknown S3 bucket key %s", key)
                continue
            bucket_start = time.perf_counter()
            get_vulkan_s3_chains(bucket)
            loaded_buckets.append(
                {
                    "encoder_token_frames": bucket[0],
                    "encoder_up_frames": bucket[1],
                    "estimator_frames": bucket[2],
                    "load_seconds": time.perf_counter() - bucket_start,
                }
            )
        report["s3_loaded_buckets"] = loaded_buckets
    report["total_seconds"] = time.perf_counter() - started
    logger.info("Startup warmup complete: %s", report)


def flow_inference_with_optional_vulkan_s3(
    model: ChatterboxTurboTTS,
    speech_tokens: torch.Tensor,
    bucket: tuple[int, int, int] | None,
    n_cfm_timesteps: int = S3_TIMESTEPS,
    timings: dict[str, Any] | None = None,
) -> torch.Tensor:
    if bucket is None:
        return model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)

    try:
        from benchmark_s3_flow_hybrid_vulkan_encoder_estimator import patch_encoder
        from benchmark_s3_flow_hybrid_vulkan_estimator import patch_estimator

        encoder_chain, estimator_chain = get_vulkan_s3_chains(bucket)
        encoder_records = []
        estimator_records = []
        encoder, original_encoder_forward = patch_encoder(model, encoder_chain, encoder_records)
        estimator, original_estimator_forward = patch_estimator(model, estimator_chain, estimator_records)
        try:
            mels = model.s3gen.flow_inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
                n_cfm_timesteps=n_cfm_timesteps,
                finalize=True,
            ).to(dtype=model.s3gen.dtype)
        finally:
            encoder.forward = original_encoder_forward
            estimator.forward = original_estimator_forward

        encoder_fallbacks = sum(1 for record in encoder_records if record.get("fallback_cpu"))
        estimator_fallbacks = sum(1 for record in estimator_records if record.get("fallback_cpu"))
        if timings is not None:
            timings["s3_encoder_calls"] = {
                "total": len(encoder_records),
                "vulkan": len(encoder_records) - encoder_fallbacks,
                "fallback_cpu": encoder_fallbacks,
                "vulkan_chain_fetch_seconds": sum(
                    float(record.get("vulkan_chain_fetch_seconds", 0.0))
                    for record in encoder_records
                ),
                "to_numpy_seconds": sum(
                    float(record.get("to_numpy_seconds", 0.0))
                    for record in encoder_records
                ),
                "cache_stats": getattr(encoder_chain, "cache_stats", None),
                "records": encoder_records,
            }
            timings["s3_estimator_calls"] = {
                "total": len(estimator_records),
                "vulkan": len(estimator_records) - estimator_fallbacks,
                "fallback_cpu": estimator_fallbacks,
                "frontend_seconds": sum(
                    float(record.get("frontend_seconds", 0.0))
                    for record in estimator_records
                ),
                "vulkan_chain_fetch_seconds": sum(
                    float(record.get("vulkan_chain_fetch_seconds", 0.0))
                    for record in estimator_records
                ),
                "to_numpy_seconds": sum(
                    float(record.get("to_numpy_seconds", 0.0))
                    for record in estimator_records
                ),
                "cache_stats": getattr(estimator_chain, "cache_stats", None),
                "records": estimator_records,
            }
        if encoder_fallbacks or estimator_fallbacks:
            raise RuntimeError(
                f"Vulkan S3 bucket fallback occurred: encoder={encoder_fallbacks}, estimator={estimator_fallbacks}"
            )
        return mels
    except Exception:
        if REQUIRE_VULKAN_S3:
            raise
        logger.exception("Experimental Vulkan S3 failed; falling back to CPU S3 flow")
        return model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=n_cfm_timesteps,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)


def choose_hift_window(total_frames: int, center_start: int) -> tuple[int, int]:
    center_end = min(center_start + HIFT_CENTER_FRAMES, total_frames)
    if total_frames <= HIFT_WINDOW_FRAMES:
        return 0, total_frames
    start = center_start - (HIFT_WINDOW_FRAMES - (center_end - center_start)) // 2
    start = max(0, min(start, total_frames - HIFT_WINDOW_FRAMES))
    return start, start + HIFT_WINDOW_FRAMES


def vulkan_hift_decode(split_hift, mels: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    total_frames = int(mels.shape[-1])
    if total_frames <= HIFT_WINDOW_FRAMES and total_frames != HIFT_WINDOW_FRAMES:
        if not (HIFT_EXACT_TAIL and total_frames in split_hift.supported_frame_sizes):
            raise ValueError(
                f"No exact Vulkan HiFT VMFB for {total_frames} frames; "
                f"window={HIFT_WINDOW_FRAMES}"
            )

    samples_per_frame = split_hift.samples_per_mel_frame
    outputs = []
    for center_start in range(0, total_frames, HIFT_CENTER_FRAMES):
        center_end = min(center_start + HIFT_CENTER_FRAMES, total_frames)
        remaining_frames = total_frames - center_start
        if (
            HIFT_EXACT_TAIL
            and remaining_frames < HIFT_WINDOW_FRAMES
            and remaining_frames in split_hift.supported_frame_sizes
        ):
            win_start, win_end = center_start, total_frames
        elif HIFT_COMPACT_TAIL and center_start > 0 and remaining_frames < HIFT_WINDOW_FRAMES:
            tail_candidates = [
                frames
                for frames in split_hift.supported_frame_sizes
                if remaining_frames <= frames < HIFT_WINDOW_FRAMES
            ]
            if tail_candidates:
                tail_frames = min(tail_candidates)
                win_start, win_end = total_frames - tail_frames, total_frames
            else:
                win_start, win_end = choose_hift_window(total_frames, center_start)
        else:
            win_start, win_end = choose_hift_window(total_frames, center_start)
        mel_win = mels[..., win_start:win_end].contiguous()
        source_win = source[..., win_start * samples_per_frame : win_end * samples_per_frame].contiguous()
        wav_win = split_hift.decode_from_source(mel_win, source_win)

        crop_start = (center_start - win_start) * samples_per_frame
        crop_end = (center_end - win_start) * samples_per_frame
        outputs.append(wav_win[..., crop_start:crop_end].contiguous())
    return torch.cat(outputs, dim=-1)


def generate_with_vulkan_hift(model: ChatterboxTurboTTS, req: TTSRequest) -> torch.Tensor:
    if req.audio_prompt_path:
        model.prepare_conditionals(
            req.audio_prompt_path,
            exaggeration=req.exaggeration,
            norm_loudness=req.norm_loudness,
        )
    else:
        assert model.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

    text = punc_norm(req.text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    speech_tokens = model.t3.inference_turbo(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_gen_len=T3_MAX_GEN_LEN,
    )

    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    mels = model.s3gen.flow_inference(
        speech_tokens=speech_tokens,
        ref_dict=model.conds.gen,
        n_cfm_timesteps=S3_TIMESTEPS,
        finalize=True,
    ).to(dtype=model.s3gen.dtype)

    split_hift = get_vulkan_hift(model)
    source = split_hift._source_from_speech_feat(mels)
    wav = vulkan_hift_decode(split_hift, mels, source)
    wav[:, : len(model.s3gen.trim_fade)] *= model.s3gen.trim_fade

    return maybe_apply_watermark(model, wav)


def generate_with_vulkan_t3(
    model: ChatterboxTurboTTS,
    req: TTSRequest,
    timings: dict[str, Any] | None = None,
) -> torch.Tensor:
    timings = timings if timings is not None else {}
    timings["generation_path"] = "vulkan_t3"
    if req.audio_prompt_path:
        prepare_start = time.perf_counter()
        model.prepare_conditionals(
            req.audio_prompt_path,
            exaggeration=req.exaggeration,
            norm_loudness=req.norm_loudness,
        )
        timings["prepare_conditionals_seconds"] = time.perf_counter() - prepare_start
    else:
        assert model.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

    tokenize_start = time.perf_counter()
    text = punc_norm(req.text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)
    timings["text_normalized_chars"] = len(text)
    timings["text_token_count"] = int(text_tokens.numel())
    timings["tokenize_seconds"] = time.perf_counter() - tokenize_start

    from t3_ggml_vulkan_runtime import inference_turbo_vulkan

    t3_start = time.perf_counter()
    t3_runtime = get_vulkan_t3()
    speech_tokens = inference_turbo_vulkan(
        model.t3,
        t3_runtime,
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_gen_len=T3_MAX_GEN_LEN,
        native_sampler_seed=req.seed or DEFAULT_SEED or None,
        prefix_prefill=T3_PREFIX_PREFILL,
    )
    timings["t3_seconds"] = time.perf_counter() - t3_start
    timings["t3_prefix_prefill"] = T3_PREFIX_PREFILL
    timings["t3_prefill_info"] = getattr(t3_runtime, "last_prefill_info", None)
    timings["t3_loop_info"] = getattr(t3_runtime, "last_loop_info", None)
    timings["raw_t3_tokens"] = int(speech_tokens.numel())

    s3_bucket = None
    s3_bucket_inference = None
    if EXPERIMENTAL_VULKAN_S3:
        s3_bucket, s3_bucket_inference = infer_vulkan_s3_bucket(model, speech_tokens)
        timings["s3_bucket_inference"] = s3_bucket_inference
        if s3_bucket is None:
            message = f"No validated Vulkan S3 bucket for {s3_bucket_inference}"
            if REQUIRE_VULKAN_S3:
                raise RuntimeError(message)
            logger.info("%s; falling back to CPU S3 flow", message)

    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])
    timings["speech_tokens_with_silence"] = int(speech_tokens.numel())

    if EXPERIMENTAL_VULKAN_HIFT or EXPERIMENTAL_VULKAN_S3:
        s3_start = time.perf_counter()
        timings["s3_timesteps"] = S3_TIMESTEPS
        mels = flow_inference_with_optional_vulkan_s3(
            model,
            speech_tokens,
            s3_bucket,
            S3_TIMESTEPS,
            timings,
        )
        timings["s3_flow_seconds"] = time.perf_counter() - s3_start
        timings["mel_frames"] = int(mels.shape[-1])

        if EXPERIMENTAL_VULKAN_HIFT:
            split_hift = get_vulkan_hift(model)
            timings["hift_supported_frame_sizes"] = list(split_hift.supported_frame_sizes)
            source_start = time.perf_counter()
            source = split_hift._source_from_speech_feat(mels)
            timings["source_seconds"] = time.perf_counter() - source_start
            hift_start = time.perf_counter()
            wav = vulkan_hift_decode(split_hift, mels, source)
            timings["hift_decode_seconds"] = time.perf_counter() - hift_start
            timings["hift_window_frames"] = HIFT_WINDOW_FRAMES
            timings["hift_center_frames"] = HIFT_CENTER_FRAMES
            timings["hift_frame_sizes"] = HIFT_FRAME_SIZES
            timings["hift_exact_tail"] = HIFT_EXACT_TAIL
            timings["hift_compact_tail"] = HIFT_COMPACT_TAIL
        else:
            hift_start = time.perf_counter()
            wav, _ = model.s3gen.hift_inference(mels, None)
            timings["cpu_hift_seconds"] = time.perf_counter() - hift_start
        wav[:, : len(model.s3gen.trim_fade)] *= model.s3gen.trim_fade
    else:
        s3_hift_start = time.perf_counter()
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=S3_TIMESTEPS,
        )
        timings["cpu_s3_hift_seconds"] = time.perf_counter() - s3_hift_start

    return maybe_apply_watermark(model, wav, timings)


def synthesize(req: TTSRequest) -> tuple[bytes, int]:
    total_start = time.perf_counter()
    timings: dict[str, Any] = {
        "input_chars": len(req.text),
        "seed": req.seed or DEFAULT_SEED,
        "experimental_vulkan_t3": EXPERIMENTAL_VULKAN_T3,
        "experimental_vulkan_s3": EXPERIMENTAL_VULKAN_S3,
        "experimental_vulkan_hift": EXPERIMENTAL_VULKAN_HIFT,
    }
    seed = req.seed or DEFAULT_SEED
    if seed:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model_start = time.perf_counter()
    model = get_model()
    timings["get_model_seconds"] = time.perf_counter() - model_start
    with INFER_LOCK:
        infer_start = time.perf_counter()
        if EXPERIMENTAL_VULKAN_T3:
            try:
                wav = generate_with_vulkan_t3(model, req, timings)
            except Exception:
                if REQUIRE_VULKAN_T3:
                    raise
                logger.exception("Experimental Vulkan T3 failed; falling back to CPU T3 path")
                if EXPERIMENTAL_VULKAN_HIFT:
                    try:
                        wav = generate_with_vulkan_hift(model, req)
                    except Exception:
                        if REQUIRE_VULKAN_HIFT:
                            raise
                        logger.exception("Experimental Vulkan HiFT failed; falling back to CPU HiFT")
                        wav = model.generate(
                            req.text,
                            audio_prompt_path=req.audio_prompt_path,
                            exaggeration=req.exaggeration,
                            cfg_weight=req.cfg_weight,
                            temperature=req.temperature,
                            min_p=req.min_p,
                            top_p=req.top_p,
                            top_k=req.top_k,
                            repetition_penalty=req.repetition_penalty,
                            norm_loudness=req.norm_loudness,
                            apply_watermark=APPLY_WATERMARK,
                        )
                else:
                    wav = model.generate(
                        req.text,
                        audio_prompt_path=req.audio_prompt_path,
                        exaggeration=req.exaggeration,
                        cfg_weight=req.cfg_weight,
                        temperature=req.temperature,
                        min_p=req.min_p,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                        norm_loudness=req.norm_loudness,
                        apply_watermark=APPLY_WATERMARK,
                    )
        elif EXPERIMENTAL_VULKAN_HIFT:
            try:
                wav = generate_with_vulkan_hift(model, req)
            except Exception:
                if REQUIRE_VULKAN_HIFT:
                    raise
                logger.exception("Experimental Vulkan HiFT failed; falling back to CPU HiFT")
                wav = model.generate(
                    req.text,
                    audio_prompt_path=req.audio_prompt_path,
                    exaggeration=req.exaggeration,
                    cfg_weight=req.cfg_weight,
                    temperature=req.temperature,
                    min_p=req.min_p,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty,
                    norm_loudness=req.norm_loudness,
                    apply_watermark=APPLY_WATERMARK,
                )
        else:
            wav = model.generate(
                req.text,
                audio_prompt_path=req.audio_prompt_path,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
                temperature=req.temperature,
                min_p=req.min_p,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                norm_loudness=req.norm_loudness,
                apply_watermark=APPLY_WATERMARK,
            )
        timings["infer_locked_seconds"] = time.perf_counter() - infer_start
    wav_encode_start = time.perf_counter()
    audio = wav_bytes(wav, model.sr)
    timings["wav_encode_seconds"] = time.perf_counter() - wav_encode_start
    timings["response_bytes"] = len(audio)
    timings["total_seconds"] = time.perf_counter() - total_start
    set_last_request_timing(timings)
    return audio, model.sr


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "max_input_chars": MAX_INPUT_CHARS,
        "model_loaded": MODEL is not None,
        "experimental_vulkan_hift": EXPERIMENTAL_VULKAN_HIFT,
        "vulkan_hift_loaded": VULKAN_HIFT is not None,
        "hift_window_frames": HIFT_WINDOW_FRAMES,
        "hift_center_frames": HIFT_CENTER_FRAMES,
        "hift_frame_sizes": HIFT_FRAME_SIZES,
        "hift_exact_tail": HIFT_EXACT_TAIL,
        "hift_compact_tail": HIFT_COMPACT_TAIL,
        "apply_watermark": APPLY_WATERMARK,
        "experimental_vulkan_t3": EXPERIMENTAL_VULKAN_T3,
        "vulkan_t3_loaded": VULKAN_T3 is not None,
        "vulkan_t3_device": getattr(VULKAN_T3, "device", None),
        "t3_prefix_cache_loaded": getattr(VULKAN_T3, "_t3_prefix_prefill_cache", None) is not None,
        "t3_ggml_prefix_resident": getattr(VULKAN_T3, "_t3_ggml_prefix_cache_resident_key", None) is not None,
        "t3_ggml_weights_dir": T3_GGML_WEIGHTS_DIR or None,
        "t3_max_gen_len": T3_MAX_GEN_LEN,
        "t3_native_sampler": T3_NATIVE_SAMPLER,
        "t3_prefix_prefill": T3_PREFIX_PREFILL,
        "default_seed": DEFAULT_SEED,
        "startup_warmup": STARTUP_WARMUP,
        "startup_warmup_s3_buckets": STARTUP_WARMUP_S3_BUCKETS,
        "startup_warmup_t3_prefix": STARTUP_WARMUP_T3_PREFIX,
        "s3_timesteps": S3_TIMESTEPS,
        "experimental_vulkan_s3": EXPERIMENTAL_VULKAN_S3,
        "require_vulkan_s3": REQUIRE_VULKAN_S3,
        "allow_vulkan_s3_padding": ALLOW_VULKAN_S3_PADDING,
        "vulkan_s3_fused_midblocks": VULKAN_S3_FUSED_MIDBLOCKS,
        "vulkan_s3_fused_variant": VULKAN_S3_FUSED_VARIANT if VULKAN_S3_FUSED_MIDBLOCKS else None,
        "vulkan_s3_buckets": [
            {
                "encoder_token_frames": bucket[0],
                "encoder_up_frames": bucket[1],
                "estimator_frames": bucket[2],
            }
            for bucket in VULKAN_S3_BUCKETS.values()
        ],
        "vulkan_s3_loaded_buckets": [
            {
                "encoder_token_frames": bucket[0],
                "encoder_up_frames": bucket[1],
                "estimator_frames": bucket[2],
            }
            for bucket in VULKAN_S3_CHAINS
        ],
    }


@app.get("/debug/last_request")
def debug_last_request():
    return {
        "ok": True,
        "last_request": get_last_request_timing(),
    }


@app.post("/v1/tts")
def tts_json(req: TTSRequest):
    try:
        audio, sample_rate = synthesize(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "sample_rate": sample_rate,
        "format": "wav",
        "audio_base64": base64.b64encode(audio).decode("ascii"),
    }


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)
    voice: str = "default"
    exaggeration: float = 0.0
    cfg_weight: float = 0.0
    temperature: float = 0.8
    min_p: float = 0.0
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True
    seed: int = 0


@app.get("/voices")
def voices():
    return ["default"]


@app.post("/audio/speech")
def audio_speech(req: SpeechRequest):
    try:
        audio, _sample_rate = synthesize(TTSRequest(
            text=req.input,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            min_p=req.min_p,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            norm_loudness=req.norm_loudness,
            seed=req.seed,
        ))
        return Response(content=audio, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/tts/file")
def tts_file(
    text: str = Form(..., min_length=1, max_length=MAX_INPUT_CHARS),
    audio_prompt: Optional[UploadFile] = File(None),
    temperature: float = Form(0.8),
    min_p: float = Form(0.0),
    top_p: float = Form(0.95),
    top_k: int = Form(1000),
    repetition_penalty: float = Form(1.2),
    norm_loudness: bool = Form(True),
    seed: int = Form(0),
):
    tmp_path = None
    try:
        if audio_prompt is not None:
            suffix = os.path.splitext(audio_prompt.filename or "")[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_prompt.file.read())
                tmp_path = tmp.name

        req = TTSRequest(
            text=text,
            audio_prompt_path=tmp_path,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            norm_loudness=norm_loudness,
            seed=seed,
        )
        audio, _sample_rate = synthesize(req)
        return Response(content=audio, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
