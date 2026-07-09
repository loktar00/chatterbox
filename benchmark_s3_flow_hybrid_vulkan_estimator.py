#!/usr/bin/env python3
"""Benchmark real S3 flow with the estimator calls routed through Vulkan."""

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
from einops import pack, repeat

from benchmark_s3_estimator_distinct_iree_runtime_chain import (
    diff_summary,
    helper_vmfb,
    load_modules,
    module_names,
    to_host_array,
)
from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
CAPTURE = BASE / "real_estimator_inputs" / "chunk270_vulkan_t3_cpu_s3_2026-07-08"
FUSED = BASE / "fused_estimator"


def fused_midblock_vmfb(frames: int, mid_index: int, variant: str = "split8") -> Path:
    name = f"s3_fused_mid{mid_index}_block_t{frames}"
    return FUSED / name / "compile_flag_matrix" / f"{name}_{variant}_vulkan_gfx1013.vmfb"


class VulkanEstimatorChain:
    def __init__(self, frames: int = 1210) -> None:
        started = time.perf_counter()
        self.frames = frames
        self.modules, self.helpers = load_modules(frames=frames)
        self.device = ireert.get_device("vulkan")
        self._mask_cache: dict[int, Any] = {}
        self._attention_bias_cache: dict[int, Any] = {}
        self.cache_stats = {
            "mask_hits": 0,
            "mask_misses": 0,
            "attention_bias_hits": 0,
            "attention_bias_misses": 0,
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

    def run(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        time_emb: np.ndarray,
        attention_bias: np.ndarray,
    ) -> np.ndarray:
        actual_frames = int(x.shape[-1]) if x.ndim == 3 else -1
        if actual_frames > self.frames:
            raise ValueError(f"unsupported estimator frames {actual_frames}; bucket supports {self.frames}")
        if list(x.shape[:2]) != [1, 320] or list(mask.shape) != [1, 1, actual_frames]:
            raise ValueError(
                f"unsupported estimator input shapes x={list(x.shape)} mask={list(mask.shape)}"
            )
        if list(attention_bias.shape) != [1, 1, actual_frames] or list(time_emb.shape) != [1, 1024]:
            raise ValueError(
                "unsupported estimator conditioning shapes "
                f"attention_bias={list(attention_bias.shape)} time_emb={list(time_emb.shape)}"
            )

        if actual_frames == self.frames:
            bucket_x = x.astype(np.float32, copy=False)
            bucket_mask = mask.astype(np.float32, copy=False)
            bucket_attention_bias = attention_bias.astype(np.float32, copy=False)
        else:
            bucket_x = np.zeros((1, 320, self.frames), dtype=np.float32)
            bucket_x[:, :, :actual_frames] = x.astype(np.float32, copy=False)
            bucket_mask = np.zeros((1, 1, self.frames), dtype=np.float32)
            bucket_mask[:, :, :actual_frames] = mask.astype(np.float32, copy=False)
            bucket_attention_bias = np.full((1, 1, self.frames), -1.0e10, dtype=np.float32)
            bucket_attention_bias[:, :, :actual_frames] = attention_bias.astype(np.float32, copy=False)

        device_x = ireert.asdevicearray(self.device, bucket_x, implicit_host_transfer=False)
        cache_static_conditioning = bool(np.all(mask == 1.0) and np.all(attention_bias == 0.0))
        if cache_static_conditioning:
            device_mask = self._cached_device_array(self._mask_cache, "mask", actual_frames, bucket_mask)
            device_attention_bias = self._cached_device_array(
                self._attention_bias_cache,
                "attention_bias",
                actual_frames,
                bucket_attention_bias,
            )
        else:
            device_mask = ireert.asdevicearray(self.device, bucket_mask, implicit_host_transfer=False)
            device_attention_bias = ireert.asdevicearray(
                self.device,
                bucket_attention_bias,
                implicit_host_transfer=False,
            )
        device_time = ireert.asdevicearray(
            self.device,
            time_emb.astype(np.float32, copy=False),
            implicit_host_transfer=False,
        )

        hidden = self.modules["down_resnet"]["forward"](device_x, device_mask, device_time)
        hidden = self.helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = self.modules[f"down_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = self.helpers["t_to_c"]["forward"](hidden)
        skip = hidden
        hidden = self.modules["downsample"]["forward"](hidden)

        for mid in range(12):
            hidden = self.modules[f"mid_resnet_{mid}"]["forward"](hidden, device_mask, device_time)
            hidden = self.helpers["c_to_t"]["forward"](hidden)
            for block in range(4):
                hidden = self.modules[f"mid_transformer_{mid}_{block}"]["forward"](
                    hidden,
                    device_attention_bias,
                    device_time,
                )
            hidden = self.helpers["t_to_c"]["forward"](hidden)

        hidden = self.helpers["cat"]["forward"](hidden, skip)
        hidden = self.modules["up_resnet"]["forward"](hidden, device_mask, device_time)
        hidden = self.helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = self.modules[f"up_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = self.helpers["t_to_c"]["forward"](hidden)
        hidden = self.modules["upsample"]["forward"](hidden)
        hidden = self.modules["final_block"]["forward"](hidden, device_mask)
        output = self.modules["final_proj"]["forward"](hidden)
        return to_host_array(output).astype(np.float32, copy=False)[:, :, :actual_frames]


class VulkanFusedMidblockEstimatorChain(VulkanEstimatorChain):
    def __init__(self, frames: int = 1222, variant: str = "split8") -> None:
        started = time.perf_counter()
        super().__init__(frames=frames)
        self.variant = variant
        self.chain_type = "fused_midblocks"
        missing = [
            fused_midblock_vmfb(frames, mid, variant).as_posix()
            for mid in range(12)
            if not fused_midblock_vmfb(frames, mid, variant).exists()
        ]
        if missing:
            raise RuntimeError("Missing fused S3 estimator midblock VMFBs:\n" + "\n".join(missing))
        self.fused_midblocks = {
            mid: ireert.load_vm_flatbuffer_file(
                fused_midblock_vmfb(frames, mid, variant).as_posix(),
                driver="vulkan",
            )
            for mid in range(12)
        }
        self.load_seconds = time.perf_counter() - started

    def run(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        time_emb: np.ndarray,
        attention_bias: np.ndarray,
    ) -> np.ndarray:
        actual_frames = int(x.shape[-1]) if x.ndim == 3 else -1
        if actual_frames > self.frames:
            raise ValueError(f"unsupported estimator frames {actual_frames}; bucket supports {self.frames}")
        if list(x.shape[:2]) != [1, 320] or list(mask.shape) != [1, 1, actual_frames]:
            raise ValueError(
                f"unsupported estimator input shapes x={list(x.shape)} mask={list(mask.shape)}"
            )
        if list(attention_bias.shape) != [1, 1, actual_frames] or list(time_emb.shape) != [1, 1024]:
            raise ValueError(
                "unsupported estimator conditioning shapes "
                f"attention_bias={list(attention_bias.shape)} time_emb={list(time_emb.shape)}"
            )

        if actual_frames == self.frames:
            bucket_x = x.astype(np.float32, copy=False)
            bucket_mask = mask.astype(np.float32, copy=False)
            bucket_attention_bias = attention_bias.astype(np.float32, copy=False)
        else:
            bucket_x = np.zeros((1, 320, self.frames), dtype=np.float32)
            bucket_x[:, :, :actual_frames] = x.astype(np.float32, copy=False)
            bucket_mask = np.zeros((1, 1, self.frames), dtype=np.float32)
            bucket_mask[:, :, :actual_frames] = mask.astype(np.float32, copy=False)
            bucket_attention_bias = np.full((1, 1, self.frames), -1.0e10, dtype=np.float32)
            bucket_attention_bias[:, :, :actual_frames] = attention_bias.astype(np.float32, copy=False)

        device_x = ireert.asdevicearray(self.device, bucket_x, implicit_host_transfer=False)
        cache_static_conditioning = bool(np.all(mask == 1.0) and np.all(attention_bias == 0.0))
        if cache_static_conditioning:
            device_mask = self._cached_device_array(self._mask_cache, "mask", actual_frames, bucket_mask)
            device_attention_bias = self._cached_device_array(
                self._attention_bias_cache,
                "attention_bias",
                actual_frames,
                bucket_attention_bias,
            )
        else:
            device_mask = ireert.asdevicearray(self.device, bucket_mask, implicit_host_transfer=False)
            device_attention_bias = ireert.asdevicearray(
                self.device,
                bucket_attention_bias,
                implicit_host_transfer=False,
            )
        device_time = ireert.asdevicearray(
            self.device,
            time_emb.astype(np.float32, copy=False),
            implicit_host_transfer=False,
        )

        hidden = self.modules["down_resnet"]["forward"](device_x, device_mask, device_time)
        hidden = self.helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = self.modules[f"down_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = self.helpers["t_to_c"]["forward"](hidden)
        skip = hidden
        hidden = self.modules["downsample"]["forward"](hidden)

        for mid in range(12):
            hidden = self.fused_midblocks[mid]["forward"](
                hidden,
                device_mask,
                device_attention_bias,
                device_time,
            )

        hidden = self.helpers["cat"]["forward"](hidden, skip)
        hidden = self.modules["up_resnet"]["forward"](hidden, device_mask, device_time)
        hidden = self.helpers["c_to_t"]["forward"](hidden)
        for index in range(4):
            hidden = self.modules[f"up_transformer_{index}"]["forward"](
                hidden,
                device_attention_bias,
                device_time,
            )
        hidden = self.helpers["t_to_c"]["forward"](hidden)
        hidden = self.modules["upsample"]["forward"](hidden)
        hidden = self.modules["final_block"]["forward"](hidden, device_mask)
        output = self.modules["final_proj"]["forward"](hidden)
        return to_host_array(output).astype(np.float32, copy=False)[:, :, :actual_frames]


def rss_mb() -> float:
    with Path("/proc/self/status").open() as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def tensor_to_float_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def build_attention_bias(mask: torch.Tensor) -> torch.Tensor:
    return (1.0 - (mask > 0.5).to(torch.float32)) * -1.0e10


def run_s3_flow(
    model: ChatterboxTurboTTS,
    speech_tokens: torch.Tensor,
    noised_mels: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        return model.s3gen(
            speech_tokens=speech_tokens,
            ref_wav=None,
            ref_sr=None,
            ref_dict=model.conds.gen,
            finalize=True,
            skip_vocoder=True,
            n_cfm_timesteps=2,
            noised_mels=noised_mels,
        ).to(dtype=model.s3gen.dtype)


def patch_estimator(
    model: ChatterboxTurboTTS,
    chain: VulkanEstimatorChain,
    records: list[dict[str, Any]],
):
    estimator = model.s3gen.flow.decoder.estimator
    original_forward = estimator.forward

    def wrapped_forward(
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        record: dict[str, Any] = {
            "index": len(records),
            "chain_type": getattr(chain, "chain_type", "stitched_midblocks"),
            "chain_variant": getattr(chain, "variant", None),
            "external_shapes": {
                "x": list(x.shape),
                "mask": list(mask.shape),
                "mu": list(mu.shape),
                "t": list(t.shape),
                "spks": list(spks.shape) if spks is not None else None,
                "cond": list(cond.shape) if cond is not None else None,
                "r": list(r.shape) if r is not None else None,
            },
            "fallback_cpu": False,
        }
        overall_start = time.perf_counter()

        frames = int(x.shape[-1]) if x.ndim == 3 else -1
        if (
            list(x.shape[:2]) != [1, 80]
            or list(mask.shape) != [1, 1, frames]
            or list(mu.shape) != [1, 80, frames]
            or frames > chain.frames
        ):
            record["fallback_cpu"] = True
            record["fallback_reason"] = "unsupported_shape"
            output = original_forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            record["total_seconds"] = time.perf_counter() - overall_start
            records.append(record)
            return output
        record["bucket_shape"] = {
            "frames": chain.frames,
            "actual_frames": frames,
            "padded": frames != chain.frames,
        }

        if not bool(torch.all(mask == 1).item()):
            record["fallback_cpu"] = True
            record["fallback_reason"] = "non_all_ones_mask"
            output = original_forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            record["total_seconds"] = time.perf_counter() - overall_start
            records.append(record)
            return output

        frontend_start = time.perf_counter()
        time_emb = estimator.time_embeddings(t).to(t.dtype)
        time_emb = estimator.time_mlp(time_emb)
        if estimator.meanflow:
            if r is None:
                raise RuntimeError("meanflow estimator requires r")
            r_emb = estimator.time_embeddings(r).to(time_emb.dtype)
            r_emb = estimator.time_mlp(r_emb)
            time_emb = estimator.time_embed_mixer(torch.cat([time_emb, r_emb], dim=1))

        packed_x = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks_expanded = repeat(spks, "b c -> b c t", t=packed_x.shape[-1])
            packed_x = pack([packed_x, spks_expanded], "b * t")[0]
        if cond is not None:
            packed_x = pack([packed_x, cond], "b * t")[0]
        attention_bias = build_attention_bias(mask)
        record["frontend_seconds"] = time.perf_counter() - frontend_start

        if list(packed_x.shape) != [1, 320, frames] or list(time_emb.shape) != [1, 1024]:
            record["fallback_cpu"] = True
            record["fallback_reason"] = "unsupported_packed_shape"
            output = original_forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
            record["total_seconds"] = time.perf_counter() - overall_start
            records.append(record)
            return output

        transfer_start = time.perf_counter()
        packed_x_np = tensor_to_float_np(packed_x)
        mask_np = tensor_to_float_np(mask)
        time_emb_np = tensor_to_float_np(time_emb)
        attention_bias_np = tensor_to_float_np(attention_bias)
        record["to_numpy_seconds"] = time.perf_counter() - transfer_start

        vulkan_start = time.perf_counter()
        output_np = chain.run(packed_x_np, mask_np, time_emb_np, attention_bias_np)
        record["vulkan_chain_fetch_seconds"] = time.perf_counter() - vulkan_start

        output = torch.from_numpy(output_np).to(device=x.device, dtype=x.dtype)
        record["output_shape"] = list(output.shape)
        record["total_seconds"] = time.perf_counter() - overall_start
        records.append(record)
        return output

    estimator.forward = wrapped_forward
    return estimator, original_forward


def diff_tensor(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    return diff_summary(tensor_to_float_np(actual), tensor_to_float_np(expected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-tokens", type=Path, default=CAPTURE / "speech_tokens.npy")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "s3_flow_hybrid_vulkan_estimator_chunk270_2026-07-08.json",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--flow-seed", type=int, default=20261708)
    parser.add_argument("--frames", type=int, default=1210)
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

    chain = VulkanEstimatorChain(frames=args.frames)
    records: list[dict[str, Any]] = []
    estimator, original_forward = patch_estimator(model, chain, records)
    rss_before = rss_mb()
    torch.manual_seed(args.flow_seed)
    hybrid_start = time.perf_counter()
    try:
        hybrid_mels = run_s3_flow(model, speech_tokens, noised_mels.clone())
    finally:
        estimator.forward = original_forward
    hybrid_flow_seconds = time.perf_counter() - hybrid_start
    rss_after = rss_mb()

    validation = diff_tensor(hybrid_mels, cpu_mels) if cpu_mels is not None else None
    fallback_calls = [record for record in records if record["fallback_cpu"]]
    vulkan_calls = [record for record in records if not record["fallback_cpu"]]
    vulkan_chain_seconds = sum(float(record.get("vulkan_chain_fetch_seconds", 0.0)) for record in records)
    frontend_seconds = sum(float(record.get("frontend_seconds", 0.0)) for record in records)
    to_numpy_seconds = sum(float(record.get("to_numpy_seconds", 0.0)) for record in records)
    estimator_total_seconds = sum(float(record.get("total_seconds", 0.0)) for record in records)

    report = {
        "description": "Real S3 flow benchmark with estimator.forward routed through the fixed-bucket IREE Vulkan chain.",
        "frames": args.frames,
        "speech_tokens": args.speech_tokens.as_posix(),
        "speech_token_count_with_silence": int(speech_tokens.numel()),
        "seed": args.seed,
        "flow_seed": args.flow_seed,
        "model_load_seconds": model_load_seconds,
        "vulkan_estimator_module_load_seconds": chain.load_seconds,
        "cpu_flow_seconds": cpu_flow_seconds,
        "hybrid_flow_seconds": hybrid_flow_seconds,
        "mel_shape": list(hybrid_mels.shape),
        "validation_against_cpu_flow": validation,
        "estimator_calls": {
            "total": len(records),
            "vulkan": len(vulkan_calls),
            "fallback_cpu": len(fallback_calls),
            "records": records,
            "frontend_seconds": frontend_seconds,
            "to_numpy_seconds": to_numpy_seconds,
            "vulkan_chain_fetch_seconds": vulkan_chain_seconds,
            "wrapped_total_seconds": estimator_total_seconds,
        },
        "rss_mb": {
            "before_hybrid": rss_before,
            "after_hybrid": rss_after,
            "delta": rss_after - rss_before,
        },
        "notes": [
            "This is a hybrid integration benchmark, not the live API.",
            "The estimator outputs are fetched back to CPU after each Vulkan call so the rest of S3 flow can continue unchanged.",
            "The wrapper falls back to CPU for unsupported shapes or non-all-ones masks.",
            "No ROCm/HIP path is used.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"results={args.output}")
    print(f"model_load_seconds={model_load_seconds:.3f}")
    print(f"vulkan_estimator_module_load_seconds={chain.load_seconds:.3f}")
    if cpu_flow_seconds is not None:
        print(f"cpu_flow_seconds={cpu_flow_seconds:.3f}")
    print(f"hybrid_flow_seconds={hybrid_flow_seconds:.3f}")
    print(f"mel_shape={list(hybrid_mels.shape)}")
    print(f"estimator_calls={len(records)} vulkan={len(vulkan_calls)} fallback_cpu={len(fallback_calls)}")
    print(f"vulkan_chain_fetch_seconds_total={vulkan_chain_seconds:.3f}")
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
