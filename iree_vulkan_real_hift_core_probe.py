#!/usr/bin/env python3
"""IREE Vulkan probe for the real Chatterbox Turbo HiFT core without FFT."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from iree.turbine import aot
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import parametrize

from chatterbox.models.s3gen.const import S3GEN_SR
from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import HiFTGenerator


ROOT = Path(__file__).resolve().parent
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
CKPT = Path(
    "/root/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/"
    "snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd/s3gen_meanflow.safetensors"
)
OUT_DIR = ROOT / "exports" / "iree_vulkan_real_hift_core"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def build_real_mel2wav_core() -> "RealHiFTCoreNoFft":
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
    return RealHiFTCoreNoFft(model).cpu().eval()


class RealHiFTCoreNoFft(nn.Module):
    """Real HiFT decode core taking source STFT directly and returning mag/phase."""

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

            si = g.source_downs[i](source_stft)
            si = g.source_resblocks[i](si)
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


def run(cmd: list[str]) -> dict:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout.splitlines()[-8:],
        "stderr_tail": completed.stderr.splitlines()[-20:],
    }


def compare(output_path: Path, expected: np.ndarray) -> dict:
    output = np.load(output_path)
    diff = np.abs(output - expected)
    return {
        "output": output_path.as_posix(),
        "shape": list(output.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(output, expected, atol=1e-4, rtol=1e-4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mel-frames", type=int, default=1)
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()

    result: dict = {
        "mel_frames": args.mel_frames,
        "checkpoint": CKPT.as_posix(),
        "rss_start_mb": rss_mb(),
    }
    started_all = time.monotonic()
    try:
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = build_real_mel2wav_core()
        result["rss_after_model_mb"] = rss_mb()
        result["param_count"] = sum(p.numel() for p in model.parameters())

        speech_feat = torch.rand(1, 80, args.mel_frames, dtype=torch.float32)
        source_len = args.mel_frames * 8 * 5 * 3 * 4
        source = torch.rand(1, 1, source_len, dtype=torch.float32)
        with torch.no_grad():
            real, imag = model.generator._stft(source.squeeze(1))
            source_stft = torch.cat([real, imag], dim=1)
            expected = model(speech_feat, source_stft).detach().cpu().numpy()

        result["source_len"] = source_len
        result["source_stft_shape"] = list(source_stft.shape)
        result["expected_shape"] = list(expected.shape)

        speech_feat_path = OUT_DIR / f"speech_feat_t{args.mel_frames}.npy"
        source_stft_path = OUT_DIR / f"source_stft_t{args.mel_frames}.npy"
        expected_path = OUT_DIR / f"torch_output_t{args.mel_frames}.npy"
        np.save(speech_feat_path, speech_feat.numpy())
        np.save(source_stft_path, source_stft.numpy())
        np.save(expected_path, expected)
        result.update({
            "speech_feat": speech_feat_path.as_posix(),
            "source_stft": source_stft_path.as_posix(),
            "torch_output": expected_path.as_posix(),
        })

        if not args.skip_cpu:
            cpu_started = time.monotonic()
            cpu_export = aot.export(
                model,
                args=(speech_feat, source_stft),
                module_name=f"real_hift_core_no_fft_t{args.mel_frames}",
                function_name="forward",
            )
            mlir_path = OUT_DIR / f"real_hift_core_no_fft_t{args.mel_frames}.mlir"
            cpu_export.save_mlir(mlir_path)
            cpu_vmfb = OUT_DIR / f"real_hift_core_no_fft_t{args.mel_frames}_llvm_cpu.vmfb"
            cpu_export.compile(cpu_vmfb, target_backends=("llvm-cpu",))
            cpu_output = OUT_DIR / f"iree_cpu_output_t{args.mel_frames}.npy"
            cpu_run = run([
                IREE_RUN.as_posix(),
                f"--module={cpu_vmfb}",
                "--device=local-task",
                "--function=forward",
                f"--input=@{speech_feat_path}",
                f"--input=@{source_stft_path}",
                f"--output=@{cpu_output}",
            ])
            result["cpu"] = {
                "seconds": time.monotonic() - cpu_started,
                "rss_after_mb": rss_mb(),
                "vmfb": cpu_vmfb.as_posix(),
                "run": cpu_run,
            }
            if cpu_run["returncode"] == 0 and cpu_output.exists():
                result["cpu"]["compare"] = compare(cpu_output, expected)

        vulkan_started = time.monotonic()
        vulkan_export = aot.export(
            model,
            args=(speech_feat, source_stft),
            module_name=f"real_hift_core_no_fft_t{args.mel_frames}",
            function_name="forward",
        )
        vulkan_export.session.set_flags("--iree-vulkan-target=gfx1013")
        vulkan_vmfb = OUT_DIR / f"real_hift_core_no_fft_t{args.mel_frames}_vulkan_gfx1013.vmfb"
        vulkan_export.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
        vulkan_output = OUT_DIR / f"iree_vulkan_output_t{args.mel_frames}.npy"
        vulkan_run = run([
            IREE_RUN.as_posix(),
            f"--module={vulkan_vmfb}",
            "--device=vulkan",
            "--function=forward",
            f"--input=@{speech_feat_path}",
            f"--input=@{source_stft_path}",
            f"--output=@{vulkan_output}",
        ])
        result["vulkan"] = {
            "seconds": time.monotonic() - vulkan_started,
            "rss_after_mb": rss_mb(),
            "vmfb": vulkan_vmfb.as_posix(),
            "run": vulkan_run,
        }
        if vulkan_run["returncode"] == 0 and vulkan_output.exists():
            result["vulkan"]["compare"] = compare(vulkan_output, expected)
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]

    result["total_seconds"] = time.monotonic() - started_all
    result["rss_end_mb"] = rss_mb()
    result_path = OUT_DIR / f"real_hift_core_t{args.mel_frames}_probe.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
