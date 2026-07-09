#!/usr/bin/env python3
"""Stage matrix for real Chatterbox Turbo HiFT core on IREE Vulkan."""

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
OUT_DIR = ROOT / "exports" / "iree_vulkan_real_hift_stages"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


class Probe:
    def __init__(self, name: str, build: Callable[[HiFTGenerator], nn.Module], inputs: tuple[torch.Tensor, ...]) -> None:
        self.name = name
        self.build = build
        self.inputs = inputs


class ConvPreOnly(nn.Module):
    def __init__(self, g: HiFTGenerator) -> None:
        super().__init__()
        self.g = g

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        return self.g.conv_pre(speech_feat)


class Up0Only(nn.Module):
    def __init__(self, g: HiFTGenerator) -> None:
        super().__init__()
        self.g = g

    def forward(self, speech_feat: torch.Tensor) -> torch.Tensor:
        x = self.g.conv_pre(speech_feat)
        x = F.leaky_relu(x, self.g.lrelu_slope)
        return self.g.ups[0](x)


class SourceBranch(nn.Module):
    def __init__(self, g: HiFTGenerator, idx: int) -> None:
        super().__init__()
        self.g = g
        self.idx = idx

    def forward(self, source_stft: torch.Tensor) -> torch.Tensor:
        return self.g.source_resblocks[self.idx](self.g.source_downs[self.idx](source_stft))


class Stages(nn.Module):
    def __init__(self, g: HiFTGenerator, n_stages: int, include_post: bool = False) -> None:
        super().__init__()
        self.g = g
        self.n_stages = n_stages
        self.include_post = include_post

    def forward(self, speech_feat: torch.Tensor, source_stft: torch.Tensor) -> torch.Tensor:
        g = self.g
        x = g.conv_pre(speech_feat)
        for i in range(self.n_stages):
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

        if not self.include_post:
            return x

        x = F.leaky_relu(x)
        x = g.conv_post(x)
        magnitude = torch.exp(x[:, : g.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, g.istft_params["n_fft"] // 2 + 1 :, :])
        return torch.cat([magnitude, phase], dim=1)


class PostOnly(nn.Module):
    def __init__(self, g: HiFTGenerator) -> None:
        super().__init__()
        self.g = g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x)
        x = self.g.conv_post(x)
        magnitude = torch.exp(x[:, : self.g.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.g.istft_params["n_fft"] // 2 + 1 :, :])
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
        inputs = tuple(inp.cpu() for inp in probe.inputs)
        with torch.no_grad():
            expected = model(*inputs).detach().cpu().numpy()

        input_paths = []
        for idx, inp in enumerate(inputs):
            input_path = probe_dir / f"input_{idx}.npy"
            np.save(input_path, inp.numpy())
            input_paths.append(input_path)
        expected_path = probe_dir / "torch_output.npy"
        np.save(expected_path, expected)

        exported = aot.export(model, args=inputs, module_name=probe.name, function_name="forward")
        mlir_path = probe_dir / f"{probe.name}.mlir"
        exported.save_mlir(mlir_path)

        vulkan_vmfb = probe_dir / f"{probe.name}_vulkan_gfx1013.vmfb"
        try:
            exported.session.set_flags("--iree-vulkan-target=gfx1013")
            exported.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
        except Exception as exc:
            return {
                "name": probe.name,
                "status": "vulkan_compile_failed",
                "input_shapes": [list(inp.shape) for inp in inputs],
                "expected_shape": list(expected.shape),
                "mlir": mlir_path.as_posix(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }

        output_path = probe_dir / "iree_vulkan_output.npy"
        cmd = [
            IREE_RUN.as_posix(),
            f"--module={vulkan_vmfb}",
            "--device=vulkan",
            "--function=forward",
        ]
        for input_path in input_paths:
            cmd.append(f"--input=@{input_path}")
        cmd.append(f"--output=@{output_path}")
        vulkan_run = run(cmd)

        result = {
            "name": probe.name,
            "status": "ok",
            "input_shapes": [list(inp.shape) for inp in inputs],
            "expected_shape": list(expected.shape),
            "mlir": mlir_path.as_posix(),
            "vulkan_vmfb": vulkan_vmfb.as_posix(),
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

    mel_frames = 1
    speech_feat = torch.rand(1, 80, mel_frames, dtype=torch.float32)
    source = torch.rand(1, 1, mel_frames * 8 * 5 * 3 * 4, dtype=torch.float32)
    with torch.no_grad():
        real, imag = g._stft(source.squeeze(1))
        source_stft = torch.cat([real, imag], dim=1)

        stage0 = Stages(g, 1)(speech_feat, source_stft)
        stage01 = Stages(g, 2)(speech_feat, source_stft)
        stage012 = Stages(g, 3)(speech_feat, source_stft)

    probes = [
        Probe("real_conv_pre_only", lambda g: ConvPreOnly(g), (speech_feat,)),
        Probe("real_up0_only", lambda g: Up0Only(g), (speech_feat,)),
        Probe("real_source_branch0", lambda g: SourceBranch(g, 0), (source_stft,)),
        Probe("real_source_branch1", lambda g: SourceBranch(g, 1), (source_stft,)),
        Probe("real_source_branch2", lambda g: SourceBranch(g, 2), (source_stft,)),
        Probe("real_stage0", lambda g: Stages(g, 1), (speech_feat, source_stft)),
        Probe("real_stage01", lambda g: Stages(g, 2), (speech_feat, source_stft)),
        Probe("real_stage012_no_post", lambda g: Stages(g, 3), (speech_feat, source_stft)),
        Probe("real_post_only_from_stage012", lambda g: PostOnly(g), (stage012,)),
        Probe("real_stage012_with_post", lambda g: Stages(g, 3, include_post=True), (speech_feat, source_stft)),
    ]

    results = [run_probe(probe, g) for probe in probes]
    result_path = OUT_DIR / "real_hift_stage_matrix.json"
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
