#!/usr/bin/env python3
"""Split Chatterbox Turbo HiFT runner using CPU FFT/F0 and IREE Vulkan core."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable

import iree.runtime as ireert
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import parametrize

from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import HiFTGenerator


ROOT = Path(__file__).resolve().parent
CKPT = Path(
    "/root/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/"
    "snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd/s3gen_meanflow.safetensors"
)
VMFB_DIR = ROOT / "exports" / "iree_vulkan_real_hift_core"
OUT_DIR = ROOT / "exports" / "split_hift_vulkan"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def discover_compiled_frame_sizes() -> tuple[int, ...]:
    pattern = re.compile(r"real_hift_core_no_fft_t(\d+)_vulkan_gfx1013\.vmfb$")
    sizes = []
    for path in VMFB_DIR.glob("real_hift_core_no_fft_t*_vulkan_gfx1013.vmfb"):
        match = pattern.match(path.name)
        if match:
            sizes.append(int(match.group(1)))
    return tuple(sorted(set(sizes)))


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def load_real_mel2wav() -> HiFTGenerator:
    model = HiFTGenerator(
        sampling_rate=S3GEN_SR,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=ConvRNNF0Predictor(),
    )
    weights = load_file(CKPT.as_posix())
    mel2wav_state = {
        key[len("mel2wav.") :]: value
        for key, value in weights.items()
        if key.startswith("mel2wav.")
    }
    model.load_state_dict(mel2wav_state, strict=True)
    bake_parametrized_weights(model)
    return model.cpu().eval()


class SplitHiFTVulkan:
    """Runs the real HiFT convolutional core on Vulkan for fixed mel lengths."""

    def __init__(
        self,
        mel2wav: HiFTGenerator,
        frame_sizes: Iterable[int] | None = None,
        allow_padding: bool = False,
    ) -> None:
        self.mel2wav = mel2wav
        self.allow_padding = allow_padding
        self.modules = {}
        if frame_sizes is None:
            frame_sizes = discover_compiled_frame_sizes()
        for frames in frame_sizes:
            vmfb = VMFB_DIR / f"real_hift_core_no_fft_t{frames}_vulkan_gfx1013.vmfb"
            if not vmfb.exists():
                continue
            self.modules[frames] = ireert.load_vm_flatbuffer_file(vmfb.as_posix(), driver="vulkan")
        if not self.modules:
            raise RuntimeError(f"No Vulkan VMFB modules found in {VMFB_DIR}")

    @property
    def supported_frame_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self.modules))

    @property
    def samples_per_mel_frame(self) -> int:
        scale_factor = self.mel2wav.f0_upsamp.scale_factor
        if isinstance(scale_factor, (tuple, list)):
            scale_factor = scale_factor[-1]
        return int(scale_factor)

    def select_frame_size(self, requested_frames: int) -> int:
        if requested_frames in self.modules:
            return requested_frames
        if not self.allow_padding:
            raise ValueError(
                f"No exact compiled Vulkan HiFT core for {requested_frames} mel frames; "
                f"supported={self.supported_frame_sizes}. "
                "Padding is disabled because it changes waveform output."
            )
        for compiled_frames in self.supported_frame_sizes:
            if compiled_frames >= requested_frames:
                return compiled_frames
        raise ValueError(
            f"No compiled Vulkan HiFT core can handle {requested_frames} mel frames; "
            f"supported={self.supported_frame_sizes}"
        )

    def _pad_for_compiled_shape(
        self,
        speech_feat: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        requested_frames = int(speech_feat.shape[-1])
        compiled_frames = self.select_frame_size(requested_frames)
        frame_pad = compiled_frames - requested_frames
        if frame_pad:
            speech_feat = F.pad(speech_feat, (0, frame_pad))

        requested_source_len = int(source.shape[-1])
        compiled_source_len = compiled_frames * self.samples_per_mel_frame
        source_pad = compiled_source_len - requested_source_len
        if source_pad < 0:
            raise ValueError(
                f"Source length {requested_source_len} exceeds compiled length "
                f"{compiled_source_len} for {compiled_frames} frames"
            )
        if source_pad:
            source = F.pad(source, (0, source_pad))
        return speech_feat, source, requested_source_len

    def _source_from_speech_feat(self, speech_feat: torch.Tensor) -> torch.Tensor:
        g = self.mel2wav
        with torch.inference_mode():
            f0 = g.f0_predictor(speech_feat)
            source = g.f0_upsamp(f0[:, None]).transpose(1, 2)
            source, _, _ = g.m_source(source)
            return source.transpose(1, 2).contiguous()

    def _source_stft(self, source: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            real, imag = self.mel2wav._stft(source.squeeze(1))
            return torch.cat([real, imag], dim=1).contiguous()

    def run_core(self, speech_feat: torch.Tensor, source_stft: torch.Tensor) -> torch.Tensor:
        frames = int(speech_feat.shape[-1])
        module = self.modules.get(frames)
        if module is None:
            raise ValueError(
                f"Unsupported mel frame count {frames}; supported={self.supported_frame_sizes}"
            )

        speech_np = speech_feat.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        stft_np = source_stft.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        device_output = module["forward"](speech_np, stft_np)
        host_output = device_output.to_host() if hasattr(device_output, "to_host") else np.asarray(device_output)
        return torch.from_numpy(np.asarray(host_output)).to(dtype=speech_feat.dtype)

    def decode_from_source(self, speech_feat: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        speech_feat, source, requested_source_len = self._pad_for_compiled_shape(speech_feat, source)
        source_stft = self._source_stft(source)
        core_output = self.run_core(speech_feat, source_stft)
        n_mag = self.mel2wav.istft_params["n_fft"] // 2 + 1
        with torch.inference_mode():
            wav = self.mel2wav._istft(core_output[:, :n_mag, :], core_output[:, n_mag:, :])
            wav = torch.clamp(wav, -self.mel2wav.audio_limit, self.mel2wav.audio_limit)
            return wav[:, :requested_source_len]

    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        source = self._source_from_speech_feat(speech_feat)
        if cache_source is not None and cache_source.shape[2] != 0:
            source[:, :, : cache_source.shape[2]] = cache_source
        wav = self.decode_from_source(speech_feat, source)
        return wav, source


def diff_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual_np = actual.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()
    diff = np.abs(actual_np - expected_np)
    return {
        "shape": list(actual_np.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "allclose_1e_4": bool(np.allclose(actual_np, expected_np, atol=1e-4, rtol=1e-4)),
        "allclose_5e_4": bool(np.allclose(actual_np, expected_np, atol=5e-4, rtol=5e-4)),
    }


def time_call(fn, iterations: int = 20) -> dict:
    for _ in range(3):
        fn()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "mean_ms": elapsed * 1000.0 / iterations,
    }


def _validate_case(
    split: SplitHiFTVulkan,
    mel2wav: HiFTGenerator,
    frames: int,
    benchmark: bool,
) -> dict:
    torch.manual_seed(1000 + frames)
    speech_feat = torch.rand(1, 80, frames, dtype=torch.float32)

    with torch.inference_mode():
        source = split._source_from_speech_feat(speech_feat)
        cpu_decode = mel2wav.decode(speech_feat, source)
        split_decode = split.decode_from_source(speech_feat, source)

        torch.manual_seed(9000 + frames)
        cpu_inference_wav, cpu_inference_source = mel2wav.inference(speech_feat)
        torch.manual_seed(9000 + frames)
        split_inference_wav, split_inference_source = split.inference(speech_feat)

    compiled_frames = split.select_frame_size(frames)
    case = {
        "mel_frames": frames,
        "compiled_frames": compiled_frames,
        "padded": compiled_frames != frames,
        "audio_samples": int(source.shape[-1]),
        "audio_seconds_at_24k": float(source.shape[-1] / 24000.0),
        "decode_from_same_source_diff": diff_summary(split_decode, cpu_decode),
        "inference_source_diff": diff_summary(split_inference_source, cpu_inference_source),
        "inference_wav_diff": diff_summary(split_inference_wav, cpu_inference_wav),
    }

    if benchmark:
        case["benchmark"] = {
            "cpu_decode_same_source": time_call(lambda: mel2wav.decode(speech_feat, source), iterations=10),
            "split_decode_same_source": time_call(lambda: split.decode_from_source(speech_feat, source), iterations=10),
        }
        cpu_ms = case["benchmark"]["cpu_decode_same_source"]["mean_ms"]
        split_ms = case["benchmark"]["split_decode_same_source"]["mean_ms"]
        case["benchmark"]["cpu_to_split_speedup"] = cpu_ms / split_ms

    torch_wav_path = OUT_DIR / f"cpu_decode_t{frames}.npy"
    split_wav_path = OUT_DIR / f"split_vulkan_decode_t{frames}.npy"
    np.save(torch_wav_path, cpu_decode.detach().cpu().numpy())
    np.save(split_wav_path, split_decode.detach().cpu().numpy())
    case["cpu_decode_wav"] = torch_wav_path.as_posix()
    case["split_decode_wav"] = split_wav_path.as_posix()
    return case


def validate(
    frame_sizes: Iterable[int] | None,
    benchmark: bool,
    padded_frame_sizes: Iterable[int] = (),
    include_supported: bool = True,
    output_path: Path | None = None,
    allow_padding: bool = False,
) -> dict:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    mel2wav = load_real_mel2wav()
    split = SplitHiFTVulkan(mel2wav, frame_sizes=frame_sizes, allow_padding=allow_padding)

    results = {
        "checkpoint": CKPT.as_posix(),
        "discovered_frame_sizes": list(discover_compiled_frame_sizes()),
        "supported_frame_sizes": list(split.supported_frame_sizes),
        "allow_padding": allow_padding,
        "cases": [],
        "padded_cases": [],
    }

    if include_supported:
        for frames in split.supported_frame_sizes:
            case = _validate_case(split, mel2wav, frames, benchmark)
            results["cases"].append(case)
            print(
                f"t={frames}: max_abs={case['decode_from_same_source_diff']['max_abs_error']:.3e} "
                f"mean_abs={case['decode_from_same_source_diff']['mean_abs_error']:.3e}"
            )
            if benchmark:
                print(
                    f"t={frames}: cpu={case['benchmark']['cpu_decode_same_source']['mean_ms']:.3f}ms "
                    f"split={case['benchmark']['split_decode_same_source']['mean_ms']:.3f}ms "
                    f"speedup={case['benchmark']['cpu_to_split_speedup']:.2f}x"
                )

    for frames in padded_frame_sizes:
        case = _validate_case(split, mel2wav, frames, benchmark)
        results["padded_cases"].append(case)
        print(
            f"padded t={frames}->T{case['compiled_frames']}: "
            f"max_abs={case['decode_from_same_source_diff']['max_abs_error']:.3e} "
            f"mean_abs={case['decode_from_same_source_diff']['mean_abs_error']:.3e}"
        )
        if benchmark:
            print(
                f"padded t={frames}: cpu={case['benchmark']['cpu_decode_same_source']['mean_ms']:.3f}ms "
                f"split={case['benchmark']['split_decode_same_source']['mean_ms']:.3f}ms "
                f"speedup={case['benchmark']['cpu_to_split_speedup']:.2f}x"
            )

    result_path = output_path or OUT_DIR / "split_hift_vulkan_validation.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results={result_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="auto", help="Comma-separated frame sizes, or 'auto'.")
    parser.add_argument("--padded-frames", default="", help="Comma-separated non-compiled frame sizes to test.")
    parser.add_argument("--only-padded", action="store_true", help="Skip already-supported fixed-shape cases.")
    parser.add_argument("--allow-padding", action="store_true", help="Allow experimental padding to the next compiled size.")
    parser.add_argument("--output", type=Path, help="Validation JSON path.")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.frames.lower() == "auto":
        frame_sizes = None
    else:
        frame_sizes = [int(part) for part in args.frames.split(",") if part.strip()]
    padded_frame_sizes = [int(part) for part in args.padded_frames.split(",") if part.strip()]
    validate(
        frame_sizes,
        args.benchmark,
        padded_frame_sizes,
        include_supported=not args.only_padded,
        output_path=args.output,
        allow_padding=args.allow_padding,
    )


if __name__ == "__main__":
    main()
