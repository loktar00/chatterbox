#!/usr/bin/env python3
"""Compare real HiFT core Vulkan outputs after CPU ISTFT reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import HiFTGenerator


BASE = Path(__file__).resolve().parent / "exports" / "iree_vulkan_real_hift_core"


def build_generator() -> HiFTGenerator:
    return HiFTGenerator(
        sampling_rate=S3GEN_SR,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        f0_predictor=ConvRNNF0Predictor(),
    ).cpu().eval()


def reconstruct(generator: HiFTGenerator, core_output: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(core_output)
        n_mag = generator.istft_params["n_fft"] // 2 + 1
        wav = generator._istft(tensor[:, :n_mag, :], tensor[:, n_mag:, :])
        wav = torch.clamp(wav, -generator.audio_limit, generator.audio_limit)
        return wav.numpy()


def summarize_diff(actual: np.ndarray, expected: np.ndarray) -> dict:
    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
    }


def main() -> None:
    generator = build_generator()
    results = []
    for torch_path in sorted(BASE.glob("torch_output_t*.npy")):
        suffix = torch_path.stem.removeprefix("torch_output_")
        vulkan_path = BASE / f"iree_vulkan_output_{suffix}.npy"
        if not vulkan_path.exists():
            continue
        torch_core = np.load(torch_path)
        vulkan_core = np.load(vulkan_path)

        raw_diff = summarize_diff(vulkan_core, torch_core)
        torch_wav = reconstruct(generator, torch_core)
        vulkan_wav = reconstruct(generator, vulkan_core)
        wav_diff = summarize_diff(vulkan_wav, torch_wav)

        torch_wav_path = BASE / f"torch_reconstructed_wav_{suffix}.npy"
        vulkan_wav_path = BASE / f"vulkan_reconstructed_wav_{suffix}.npy"
        np.save(torch_wav_path, torch_wav)
        np.save(vulkan_wav_path, vulkan_wav)

        results.append({
            "case": suffix,
            "torch_core": torch_path.as_posix(),
            "vulkan_core": vulkan_path.as_posix(),
            "raw_core_diff": raw_diff,
            "torch_reconstructed_wav": torch_wav_path.as_posix(),
            "vulkan_reconstructed_wav": vulkan_wav_path.as_posix(),
            "reconstructed_wav_diff": wav_diff,
        })

    result_path = BASE / "real_hift_core_reconstruction_compare.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        wav = result["reconstructed_wav_diff"]
        print(
            f"{result['case']}: wav max_abs={wav['max_abs_error']:.3e} "
            f"mean_abs={wav['mean_abs_error']:.3e} allclose={wav['allclose_1e_4']}"
        )
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
