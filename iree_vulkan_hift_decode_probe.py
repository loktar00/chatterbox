#!/usr/bin/env python3
"""IREE Vulkan probe for a reduced HiFT/HiFiGAN decode path."""

from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from iree.turbine import aot
from torch import nn
from torch.nn.utils import parametrize

from chatterbox.models.s3gen.hifigan import HiFTGenerator


ROOT = Path(__file__).resolve().parent
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
OUT_DIR = ROOT / "exports" / "iree_vulkan_hift_decode"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


class TinyHiFTDecode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = HiFTGenerator(
            in_channels=80,
            base_channels=64,
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            source_resblock_kernel_sizes=[7, 7],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            istft_params={"n_fft": 8, "hop_len": 2},
            f0_predictor=None,
        )
        bake_parametrized_weights(self.generator)

    def forward(self, speech_feat: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.generator.decode(speech_feat, source)


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


def main() -> None:
    result: dict = {}
    try:
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = TinyHiFTDecode().cpu().eval()
        speech_feat = torch.rand(1, 80, 8, dtype=torch.float32)
        source = torch.rand(1, 1, 64, dtype=torch.float32)

        with torch.no_grad():
            expected = model(speech_feat, source).detach().cpu().numpy()

        speech_feat_path = OUT_DIR / "speech_feat.npy"
        source_path = OUT_DIR / "source.npy"
        expected_path = OUT_DIR / "torch_output.npy"
        np.save(speech_feat_path, speech_feat.numpy())
        np.save(source_path, source.numpy())
        np.save(expected_path, expected)

        cpu_export = aot.export(
            model,
            args=(speech_feat, source),
            module_name="tiny_hift_decode",
            function_name="forward",
        )
        mlir_path = OUT_DIR / "tiny_hift_decode.mlir"
        cpu_export.save_mlir(mlir_path)

        result.update({
            "speech_feat": speech_feat_path.as_posix(),
            "source": source_path.as_posix(),
            "torch_output": expected_path.as_posix(),
            "mlir": mlir_path.as_posix(),
        })

        cpu_vmfb = OUT_DIR / "tiny_hift_decode_llvm_cpu.vmfb"
        try:
            cpu_export.compile(cpu_vmfb, target_backends=("llvm-cpu",))
            cpu_output = OUT_DIR / "iree_cpu_output.npy"
            cpu_run = run([
                IREE_RUN.as_posix(),
                f"--module={cpu_vmfb}",
                "--device=local-task",
                "--function=forward",
                f"--input=@{speech_feat_path}",
                f"--input=@{source_path}",
                f"--output=@{cpu_output}",
            ])
            result["cpu"] = {
                "compile": "ok",
                "run": cpu_run,
            }
            if cpu_run["returncode"] == 0 and cpu_output.exists():
                result["cpu"]["compare"] = compare(cpu_output, expected)
        except Exception as exc:
            result["cpu"] = {
                "compile": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        vulkan_vmfb = OUT_DIR / "tiny_hift_decode_vulkan_gfx1013.vmfb"
        try:
            vulkan_export = aot.export(
                model,
                args=(speech_feat, source),
                module_name="tiny_hift_decode",
                function_name="forward",
            )
            vulkan_export.session.set_flags("--iree-vulkan-target=gfx1013")
            vulkan_export.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
            vulkan_output = OUT_DIR / "iree_vulkan_output.npy"
            vulkan_run = run([
                IREE_RUN.as_posix(),
                f"--module={vulkan_vmfb}",
                "--device=vulkan",
                "--function=forward",
                f"--input=@{speech_feat_path}",
                f"--input=@{source_path}",
                f"--output=@{vulkan_output}",
            ])
            result["vulkan"] = {
                "compile": "ok",
                "run": vulkan_run,
            }
            if vulkan_run["returncode"] == 0 and vulkan_output.exists():
                result["vulkan"]["compare"] = compare(vulkan_output, expected)
        except Exception as exc:
            result["vulkan"] = {
                "compile": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }

    result_path = OUT_DIR / "tiny_hift_decode_probe.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
