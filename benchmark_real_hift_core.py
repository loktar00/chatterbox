#!/usr/bin/env python3
"""Benchmark real Chatterbox Turbo HiFT no-FFT core on CPU and IREE Vulkan."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

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
IREE_BENCH = ROOT / ".venv" / "bin" / "iree-benchmark-module"
CKPT = Path(
    "/root/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/"
    "snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd/s3gen_meanflow.safetensors"
)
BASE = ROOT / "exports" / "iree_vulkan_real_hift_core"


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def load_real_generator() -> HiFTGenerator:
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
        k[len("mel2wav.") :]: v
        for k, v in weights.items()
        if k.startswith("mel2wav.")
    }
    model.load_state_dict(mel2wav_state, strict=True)
    bake_parametrized_weights(model)
    return model.cpu().eval()


class RealHiFTCoreNoFft(nn.Module):
    def __init__(self, generator: HiFTGenerator) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, speech_feat: torch.Tensor, source_stft: torch.Tensor) -> torch.Tensor:
        g = self.generator
        x = g.conv_pre(speech_feat)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, g.lrelu_slope)
            x = g.ups[i](x)

            if i == g.num_upsamples - 1:
                x = g.reflection_pad(x)

            si = g.source_resblocks[i](g.source_downs[i](source_stft))
            x = x + si

            xs = None
            for j in range(g.num_kernels):
                y = g.resblocks[i * g.num_kernels + j](x)
                xs = y if xs is None else xs + y
            x = xs / g.num_kernels

        x = F.leaky_relu(x)
        x = g.conv_post(x)
        magnitude = torch.exp(x[:, : g.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, g.istft_params["n_fft"] // 2 + 1 :, :])
        return torch.cat([magnitude, phase], dim=1)


def benchmark_pytorch(model: nn.Module, speech_feat: torch.Tensor, source_stft: torch.Tensor) -> dict:
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            model(speech_feat, source_stft)
        iterations = 20
        started = time.perf_counter()
        for _ in range(iterations):
            model(speech_feat, source_stft)
        elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "mean_ms": (elapsed / iterations) * 1000.0,
    }


def run_iree_benchmark(t: int) -> dict:
    vmfb = BASE / f"real_hift_core_no_fft_t{t}_vulkan_gfx1013.vmfb"
    speech_feat = BASE / f"speech_feat_t{t}.npy"
    source_stft = BASE / f"source_stft_t{t}.npy"
    cmd = [
        IREE_BENCH.as_posix(),
        f"--module={vmfb}",
        "--device=vulkan",
        "--function=forward",
        f"--input=@{speech_feat}",
        f"--input=@{source_stft}",
        "--benchmark_min_time=1s",
        "--benchmark_repetitions=3",
        "--benchmark_format=json",
    ]
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result = {
        "returncode": completed.returncode,
        "stderr_tail": completed.stderr.splitlines()[-12:],
    }
    if completed.returncode == 0:
        parsed = json.loads(completed.stdout)
        mean = next(
            item
            for item in parsed["benchmarks"]
            if item.get("run_type") == "aggregate" and item.get("aggregate_name") == "mean"
        )
        result.update({
            "mean_ms": mean["real_time"],
            "cpu_time_ms": mean["cpu_time"],
            "items_per_second": mean["items_per_second"],
            "time_unit": mean["time_unit"],
            "raw": parsed,
        })
    else:
        result["stdout_tail"] = completed.stdout.splitlines()[-12:]
    return result


def main() -> None:
    torch.manual_seed(0)
    generator = load_real_generator()
    model = RealHiFTCoreNoFft(generator).cpu().eval()

    results = []
    for t in [2, 4, 8]:
        speech_feat = torch.from_numpy(np.load(BASE / f"speech_feat_t{t}.npy"))
        source_stft = torch.from_numpy(np.load(BASE / f"source_stft_t{t}.npy"))
        cpu = benchmark_pytorch(model, speech_feat, source_stft)
        vulkan = run_iree_benchmark(t)
        speedup = None
        if vulkan.get("mean_ms"):
            speedup = cpu["mean_ms"] / vulkan["mean_ms"]
        results.append({
            "mel_frames": t,
            "audio_samples": t * 8 * 5 * 3 * 4,
            "audio_seconds_at_24k": (t * 8 * 5 * 3 * 4) / 24000.0,
            "pytorch_cpu": cpu,
            "iree_vulkan": vulkan,
            "cpu_to_vulkan_speedup": speedup,
        })
        print(
            f"t={t}: cpu={cpu['mean_ms']:.3f}ms "
            f"vulkan={vulkan.get('mean_ms', float('nan')):.3f}ms "
            f"speedup={speedup:.2f}x"
        )

    result_path = BASE / "real_hift_core_benchmark.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
