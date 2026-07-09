#!/usr/bin/env python3
"""Benchmark real S3 flow with encoder and estimator routed through Vulkan."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import iree.runtime as ireert
import numpy as np
import torch

from benchmark_s3_encoder_iree_runtime_chain import load_modules as load_encoder_modules
from benchmark_s3_estimator_distinct_iree_runtime_chain import to_host_array
from benchmark_s3_flow_hybrid_vulkan_estimator import (
    CAPTURE,
    BASE,
    VulkanEstimatorChain,
    diff_tensor,
    patch_estimator,
    rss_mb,
    run_s3_flow,
    tensor_to_float_np,
)
from chatterbox.tts_turbo import ChatterboxTurboTTS


class VulkanEncoderChain:
    def __init__(self, token_frames: int = 605, up_frames: int = 1210) -> None:
        started = time.perf_counter()
        self.token_frames = token_frames
        self.up_frames = up_frames
        self.modules = load_encoder_modules(token_frames=token_frames, up_frames=up_frames)
        self.device = ireert.get_device("vulkan")
        self._token_mask_cache: dict[int, Any] = {}
        self._up_mask_cache: dict[int, Any] = {}
        self._lengths_cache: dict[int, Any] = {}
        self.cache_stats = {
            "token_mask_hits": 0,
            "token_mask_misses": 0,
            "up_mask_hits": 0,
            "up_mask_misses": 0,
            "lengths_hits": 0,
            "lengths_misses": 0,
        }
        self.load_seconds = time.perf_counter() - started

    def _cached_device_array(self, cache: dict[int, Any], stats_prefix: str, key: int, array: np.ndarray):
        cached = cache.get(key)
        if cached is not None:
            self.cache_stats[f"{stats_prefix}_hits"] += 1
            return cached
        device_array = ireert.asdevicearray(self.device, array, implicit_host_transfer=False)
        cache[key] = device_array
        self.cache_stats[f"{stats_prefix}_misses"] += 1
        return device_array

    def run(self, token_hidden: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actual_token_frames = int(token_hidden.shape[1]) if token_hidden.ndim == 3 else -1
        if (
            list(token_hidden.shape[:1]) != [1]
            or token_hidden.ndim != 3
            or token_hidden.shape[2] != 512
            or actual_token_frames > self.token_frames
        ):
            raise ValueError(f"unsupported token_hidden shape {list(token_hidden.shape)}")
        if list(lengths.shape) != [1] or int(lengths[0]) != actual_token_frames:
            raise ValueError(f"unsupported lengths {lengths.tolist()}")
        actual_up_frames = actual_token_frames * 2
        if actual_up_frames > self.up_frames:
            raise ValueError(
                f"unsupported upsampled frames {actual_up_frames}; bucket supports {self.up_frames}"
            )

        if actual_token_frames == self.token_frames:
            bucket_token_hidden = token_hidden.astype(np.float32, copy=False)
        else:
            bucket_token_hidden = np.zeros((1, self.token_frames, 512), dtype=np.float32)
            bucket_token_hidden[:, :actual_token_frames, :] = token_hidden.astype(np.float32, copy=False)

        token_mask = np.zeros((1, 1, self.token_frames), dtype=np.float32)
        token_mask[:, :, :actual_token_frames] = 1.0
        up_mask = np.zeros((1, 1, self.up_frames), dtype=np.float32)
        up_mask[:, :, :actual_up_frames] = 1.0

        device_token_hidden = ireert.asdevicearray(
            self.device,
            bucket_token_hidden,
            implicit_host_transfer=False,
        )
        device_token_mask = self._cached_device_array(
            self._token_mask_cache,
            "token_mask",
            actual_token_frames,
            token_mask,
        )
        device_up_mask = self._cached_device_array(
            self._up_mask_cache,
            "up_mask",
            actual_token_frames,
            up_mask,
        )
        device_lengths = self._cached_device_array(
            self._lengths_cache,
            "lengths",
            actual_token_frames,
            lengths.astype(np.int64, copy=False),
        )

        lower_hidden, lower_pos, lower_mask = self.modules["embed"]["forward"](
            device_token_hidden,
            device_token_mask,
        )
        hidden = self.modules["pre_lookahead"]["forward"](lower_hidden)
        for index in range(6):
            hidden = self.modules[f"lower_layer_{index}"]["forward"](
                hidden,
                lower_mask,
                lower_pos,
                lower_mask,
            )
        hidden_ct = self.modules["lower_transpose"]["forward"](hidden)
        up_hidden_ct = self.modules["up_layer"]["forward"](hidden_ct, device_lengths)
        up_hidden = self.modules["up_transpose"]["forward"](up_hidden_ct)
        upper_hidden, upper_pos, upper_mask = self.modules["up_embed"]["forward"](
            up_hidden,
            device_up_mask,
        )
        hidden = upper_hidden
        for index in range(4):
            hidden = self.modules[f"upper_layer_{index}"]["forward"](
                hidden,
                upper_mask,
                upper_pos,
                upper_mask,
            )
        hidden = self.modules["after_norm"]["forward"](hidden)
        hidden_np = to_host_array(hidden).astype(np.float32, copy=False)
        mask_np = to_host_array(upper_mask).astype(np.float32, copy=False)
        return (
            hidden_np[:, :actual_up_frames, :],
            mask_np[:, :, :actual_up_frames],
        )


def patch_encoder(
    model: ChatterboxTurboTTS,
    chain: VulkanEncoderChain,
    records: list[dict[str, Any]],
):
    encoder = model.s3gen.flow.encoder
    original_forward = encoder.forward

    def wrapped_forward(
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record: dict[str, Any] = {
            "index": len(records),
            "input_shape": list(xs.shape),
            "lengths": xs_lens.detach().cpu().long().tolist(),
            "decoding_chunk_size": int(decoding_chunk_size),
            "num_decoding_left_chunks": int(num_decoding_left_chunks),
            "fallback_cpu": False,
        }
        overall_start = time.perf_counter()

        actual_token_frames = int(xs.shape[1]) if xs.ndim == 3 else -1
        if (
            list(xs.shape[:1]) != [1]
            or xs.ndim != 3
            or xs.shape[2] != 512
            or actual_token_frames > chain.token_frames
            or list(xs_lens.shape) != [1]
            or int(xs_lens.detach().cpu().long()[0]) != actual_token_frames
            or actual_token_frames * 2 > chain.up_frames
            or decoding_chunk_size != 0
            or num_decoding_left_chunks != -1
        ):
            record["fallback_cpu"] = True
            record["fallback_reason"] = "unsupported_shape_or_chunking"
            output = original_forward(xs, xs_lens, decoding_chunk_size, num_decoding_left_chunks)
            record["total_seconds"] = time.perf_counter() - overall_start
            records.append(record)
            return output
        record["bucket_shape"] = {
            "token_frames": chain.token_frames,
            "up_frames": chain.up_frames,
            "padded": actual_token_frames != chain.token_frames,
        }

        transfer_start = time.perf_counter()
        xs_np = tensor_to_float_np(xs)
        lengths_np = xs_lens.detach().cpu().long().numpy().astype(np.int64, copy=False)
        record["to_numpy_seconds"] = time.perf_counter() - transfer_start

        vulkan_start = time.perf_counter()
        hidden_np, mask_np = chain.run(xs_np, lengths_np)
        record["vulkan_chain_fetch_seconds"] = time.perf_counter() - vulkan_start

        hidden = torch.from_numpy(hidden_np).to(device=xs.device, dtype=xs.dtype)
        mask = torch.from_numpy(mask_np > 0.5).to(device=xs.device)
        record["output_shape"] = list(hidden.shape)
        record["mask_shape"] = list(mask.shape)
        record["total_seconds"] = time.perf_counter() - overall_start
        records.append(record)
        return hidden, mask

    encoder.forward = wrapped_forward
    return encoder, original_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-tokens", type=Path, default=CAPTURE / "speech_tokens.npy")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_flow_hybrid_vulkan_encoder_estimator_chunk270_2026-07-08.json",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--flow-seed", type=int, default=20261708)
    parser.add_argument("--encoder-token-frames", type=int, default=605)
    parser.add_argument("--encoder-up-frames", type=int, default=1210)
    parser.add_argument("--estimator-frames", type=int, default=1210)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--skip-cpu-reference", action="store_true")
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

    speech_tokens_np = np.load(args.speech_tokens)
    speech_tokens = torch.from_numpy(speech_tokens_np.astype(np.int64, copy=False))

    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    model_load_seconds = time.perf_counter() - load_start

    torch.manual_seed(args.seed)
    noised_mels = torch.randn(
        1,
        80,
        int(speech_tokens.numel()) * 2,
        dtype=model.s3gen.dtype,
        device=model.device,
    )

    cpu_mels = None
    cpu_flow_seconds = None
    if not args.skip_cpu_reference:
        torch.manual_seed(args.flow_seed)
        cpu_start = time.perf_counter()
        cpu_mels = run_s3_flow(model, speech_tokens, noised_mels.clone())
        cpu_flow_seconds = time.perf_counter() - cpu_start

    encoder_chain = VulkanEncoderChain(
        token_frames=args.encoder_token_frames,
        up_frames=args.encoder_up_frames,
    )
    estimator_chain = VulkanEstimatorChain(frames=args.estimator_frames)
    encoder_records: list[dict[str, Any]] = []
    estimator_records: list[dict[str, Any]] = []

    encoder, original_encoder_forward = patch_encoder(model, encoder_chain, encoder_records)
    estimator, original_estimator_forward = patch_estimator(model, estimator_chain, estimator_records)

    rss_before = rss_mb()
    torch.manual_seed(args.flow_seed)
    hybrid_start = time.perf_counter()
    try:
        hybrid_mels = run_s3_flow(model, speech_tokens, noised_mels.clone())
    finally:
        encoder.forward = original_encoder_forward
        estimator.forward = original_estimator_forward
    hybrid_flow_seconds = time.perf_counter() - hybrid_start
    rss_after = rss_mb()

    validation = diff_tensor(hybrid_mels, cpu_mels) if cpu_mels is not None else None
    encoder_fallbacks = [record for record in encoder_records if record["fallback_cpu"]]
    estimator_fallbacks = [record for record in estimator_records if record["fallback_cpu"]]
    encoder_chain_seconds = sum(float(record.get("vulkan_chain_fetch_seconds", 0.0)) for record in encoder_records)
    estimator_chain_seconds = sum(
        float(record.get("vulkan_chain_fetch_seconds", 0.0)) for record in estimator_records
    )

    report = {
        "description": "Real S3 flow benchmark with encoder and estimator routed through fixed-bucket IREE Vulkan chains.",
        "encoder_token_frames": args.encoder_token_frames,
        "encoder_up_frames": args.encoder_up_frames,
        "estimator_frames": args.estimator_frames,
        "speech_tokens": args.speech_tokens.as_posix(),
        "speech_token_count_with_silence": int(speech_tokens.numel()),
        "seed": args.seed,
        "flow_seed": args.flow_seed,
        "model_load_seconds": model_load_seconds,
        "vulkan_encoder_module_load_seconds": encoder_chain.load_seconds,
        "vulkan_estimator_module_load_seconds": estimator_chain.load_seconds,
        "cpu_flow_seconds": cpu_flow_seconds,
        "hybrid_flow_seconds": hybrid_flow_seconds,
        "mel_shape": list(hybrid_mels.shape),
        "validation_against_cpu_flow": validation,
        "encoder_calls": {
            "total": len(encoder_records),
            "vulkan": len(encoder_records) - len(encoder_fallbacks),
            "fallback_cpu": len(encoder_fallbacks),
            "cache_stats": encoder_chain.cache_stats,
            "records": encoder_records,
            "vulkan_chain_fetch_seconds": encoder_chain_seconds,
        },
        "estimator_calls": {
            "total": len(estimator_records),
            "vulkan": len(estimator_records) - len(estimator_fallbacks),
            "fallback_cpu": len(estimator_fallbacks),
            "cache_stats": estimator_chain.cache_stats,
            "records": estimator_records,
            "vulkan_chain_fetch_seconds": estimator_chain_seconds,
        },
        "rss_mb": {
            "before_hybrid": rss_before,
            "after_hybrid": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "This is a hybrid integration benchmark, not the live API.",
            "The encoder and estimator outputs are fetched back to CPU so the rest of S3 flow can continue unchanged.",
            "Both wrappers fall back to CPU for unsupported shapes.",
            "No ROCm/HIP path is used.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={args.output}")
    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"vulkan_encoder_module_load_seconds={encoder_chain.load_seconds:.3f}")
    print(f"vulkan_estimator_module_load_seconds={estimator_chain.load_seconds:.3f}")
    if cpu_flow_seconds is not None:
        print(f"cpu_flow_seconds={cpu_flow_seconds:.3f}")
    print(f"hybrid_flow_seconds={hybrid_flow_seconds:.3f}")
    print(f"mel_shape={list(hybrid_mels.shape)}")
    print(
        f"encoder_calls={len(encoder_records)} "
        f"vulkan={len(encoder_records) - len(encoder_fallbacks)} "
        f"fallback_cpu={len(encoder_fallbacks)}"
    )
    print(
        f"estimator_calls={len(estimator_records)} "
        f"vulkan={len(estimator_records) - len(estimator_fallbacks)} "
        f"fallback_cpu={len(estimator_fallbacks)}"
    )
    print(f"encoder_chain_fetch_seconds_total={encoder_chain_seconds:.3f}")
    print(f"estimator_chain_fetch_seconds_total={estimator_chain_seconds:.3f}")
    if validation is not None:
        print(
            "flow_validation_allclose_1e_4="
            f"{validation['allclose_1e_4']} "
            f"allclose_1e_3={validation['allclose_1e_3']} "
            f"max_abs={validation['max_abs_error']:.3e}"
        )
    print(f"rss_delta_mb={rss_after - rss_before:.3f}")


if __name__ == "__main__":
    main()
