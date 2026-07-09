#!/usr/bin/env python3
"""IREE Vulkan debug for real HiFT ConvTranspose1d layers."""

from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path
from typing import Callable

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
OUT_DIR = ROOT / "exports" / "iree_vulkan_convtranspose_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(self, name: str, build: Callable[[HiFTGenerator], nn.Module], sample: torch.Tensor) -> None:
        self.name = name
        self.build = build
        self.sample = sample


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


class RealUpLayer(nn.Module):
    def __init__(self, g: HiFTGenerator, idx: int) -> None:
        super().__init__()
        self.layer = g.ups[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(F.leaky_relu(x, 0.1))


class RandomConvTranspose(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> None:
        super().__init__()
        self.layer = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel,
            stride,
            padding=(kernel - stride) // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(F.leaky_relu(x, 0.1))


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
        "stderr_tail": completed.stderr.splitlines()[-16:],
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


def run_probe(probe: Probe, generator: HiFTGenerator) -> dict:
    probe_dir = OUT_DIR / probe.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    try:
        model = probe.build(generator).cpu().eval()
        sample = probe.sample.cpu()
        with torch.no_grad():
            expected = model(sample).detach().cpu().numpy()
        input_path = probe_dir / "input.npy"
        expected_path = probe_dir / "torch_output.npy"
        np.save(input_path, sample.numpy())
        np.save(expected_path, expected)

        exported = aot.export(model, args=(sample,), module_name=probe.name, function_name="forward")
        mlir_path = probe_dir / f"{probe.name}.mlir"
        exported.save_mlir(mlir_path)
        try:
            exported.session.set_flags("--iree-vulkan-target=gfx1013")
            vmfb = probe_dir / f"{probe.name}_vulkan_gfx1013.vmfb"
            exported.compile(vmfb, target_backends=("vulkan-spirv",))
        except Exception as exc:
            return {
                "name": probe.name,
                "status": "vulkan_compile_failed",
                "input_shape": list(sample.shape),
                "expected_shape": list(expected.shape),
                "mlir": mlir_path.as_posix(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }

        output_path = probe_dir / "iree_vulkan_output.npy"
        vulkan_run = run([
            IREE_RUN.as_posix(),
            f"--module={vmfb}",
            "--device=vulkan",
            "--function=forward",
            f"--input=@{input_path}",
            f"--output=@{output_path}",
        ])
        result = {
            "name": probe.name,
            "status": "ok",
            "input_shape": list(sample.shape),
            "expected_shape": list(expected.shape),
            "mlir": mlir_path.as_posix(),
            "vmfb": vmfb.as_posix(),
            "vulkan_run": vulkan_run,
        }
        if vulkan_run["returncode"] == 0 and output_path.exists():
            result["vulkan_compare"] = compare(output_path, expected)
        return result
    except Exception as exc:
        return {
            "name": probe.name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    g = load_real_generator()

    probes = [
        Probe("real_up0_len1", lambda g: RealUpLayer(g, 0), torch.rand(1, 512, 1)),
        Probe("real_up0_len2", lambda g: RealUpLayer(g, 0), torch.rand(1, 512, 2)),
        Probe("real_up1_len8", lambda g: RealUpLayer(g, 1), torch.rand(1, 256, 8)),
        Probe("real_up2_len40", lambda g: RealUpLayer(g, 2), torch.rand(1, 128, 40)),
        Probe("random_up0_shape_len1", lambda g: RandomConvTranspose(512, 256, 16, 8), torch.rand(1, 512, 1)),
        Probe("random_up0_shape_len2", lambda g: RandomConvTranspose(512, 256, 16, 8), torch.rand(1, 512, 2)),
        Probe("random_up1_shape_len8", lambda g: RandomConvTranspose(256, 128, 11, 5), torch.rand(1, 256, 8)),
        Probe("random_up2_shape_len40", lambda g: RandomConvTranspose(128, 64, 7, 3), torch.rand(1, 128, 40)),
    ]

    results = [run_probe(probe, g) for probe in probes]
    result_path = OUT_DIR / "convtranspose_debug.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        cmp = result.get("vulkan_compare")
        if cmp:
            print(
                f"{result['name']}: ok max_abs={cmp['max_abs_error']:.3e} "
                f"allclose={cmp['allclose_1e_4']}"
            )
        else:
            print(f"{result['name']}: {result['status']} {result.get('error', '')[:160]}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
