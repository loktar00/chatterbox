#!/usr/bin/env python3
"""IREE Vulkan compile/run matrix for small Chatterbox subgraphs."""

from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from iree.turbine import aot
from torch import nn
from torch.nn.utils import parametrize

from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import ResBlock, Snake
from chatterbox.models.s3gen.transformer.positionwise_feed_forward import PositionwiseFeedForward


ROOT = Path(__file__).resolve().parent
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
OUT_DIR = ROOT / "exports" / "iree_vulkan_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(self, name: str, build: Callable[[], nn.Module], sample: torch.Tensor) -> None:
        self.name = name
        self.build = build
        self.sample = sample


class EluOnly(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.elu(x)


class Conv1dElu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.conv(x))


class F0PredictorLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f0 = bake_parametrized_weights(ConvRNNF0Predictor())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f0.condnet(x)
        x = x.transpose(1, 2)
        return self.f0.classifier(x).squeeze(-1)


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


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
        "stderr_tail": completed.stderr.splitlines()[-12:],
    }


def compare(output_path: Path, expected: np.ndarray) -> dict:
    output = np.load(output_path)
    return {
        "output": output_path.as_posix(),
        "shape": list(output.shape),
        "max_abs_error": float(np.max(np.abs(output - expected))),
        "allclose_1e_4": bool(np.allclose(output, expected, atol=1e-4, rtol=1e-4)),
    }


def run_probe(probe: Probe) -> dict:
    probe_dir = OUT_DIR / probe.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    try:
        torch.manual_seed(0)
        model = probe.build().cpu().eval()
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

        cpu_vmfb = probe_dir / f"{probe.name}_llvm_cpu.vmfb"
        exported.compile(cpu_vmfb, target_backends=("llvm-cpu",))
        cpu_output_path = probe_dir / "iree_cpu_output.npy"
        cpu_run = run([
            IREE_RUN.as_posix(),
            f"--module={cpu_vmfb}",
            "--device=local-task",
            "--function=forward",
            f"--input=@{input_path}",
            f"--output=@{cpu_output_path}",
        ])

        result = {
            "name": probe.name,
            "status": "ok",
            "input": input_path.as_posix(),
            "torch_output": expected_path.as_posix(),
            "mlir": mlir_path.as_posix(),
            "cpu_vmfb": cpu_vmfb.as_posix(),
            "cpu_run": cpu_run,
        }
        if cpu_run["returncode"] == 0 and cpu_output_path.exists():
            result["cpu_compare"] = compare(cpu_output_path, expected)

        vulkan_vmfb = probe_dir / f"{probe.name}_vulkan_gfx1013.vmfb"
        try:
            # Use a fresh compiler session for Vulkan. Reusing the session after
            # an llvm-cpu compile can leave CPU executable variants in the VMFB.
            exported_vulkan = aot.export(model, args=(sample,), module_name=probe.name, function_name="forward")
            exported_vulkan.session.set_flags("--iree-vulkan-target=gfx1013")
            exported_vulkan.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
            result["vulkan_compile"] = {"status": "ok", "vmfb": vulkan_vmfb.as_posix()}
        except Exception as exc:
            result["vulkan_compile"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return result

        vulkan_output_path = probe_dir / "iree_vulkan_output.npy"
        vulkan_run = run([
            IREE_RUN.as_posix(),
            f"--module={vulkan_vmfb}",
            "--device=vulkan",
            "--function=forward",
            f"--input=@{input_path}",
            f"--output=@{vulkan_output_path}",
        ])
        result["vulkan_run"] = vulkan_run
        if vulkan_run["returncode"] == 0 and vulkan_output_path.exists():
            result["vulkan_compare"] = compare(vulkan_output_path, expected)
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
    probes = [
        Probe("elu_only", lambda: EluOnly(), torch.randn(1, 16, 64, dtype=torch.float32)),
        Probe("conv1d_elu", lambda: Conv1dElu(), torch.randn(1, 8, 64, dtype=torch.float32)),
        Probe("hifigan_snake", lambda: Snake(64), torch.rand(1, 64, 64, dtype=torch.float32)),
        Probe(
            "s3gen_positionwise_ffn",
            lambda: PositionwiseFeedForward(idim=128, hidden_units=512, dropout_rate=0.0, activation=nn.SiLU()),
            torch.rand(1, 64, 128, dtype=torch.float32),
        ),
        Probe(
            "f0_predictor_baked_weightnorm",
            lambda: bake_parametrized_weights(ConvRNNF0Predictor()),
            torch.rand(1, 80, 64, dtype=torch.float32),
        ),
        Probe(
            "f0_predictor_logits_baked_weightnorm",
            lambda: F0PredictorLogits(),
            torch.rand(1, 80, 64, dtype=torch.float32),
        ),
        Probe(
            "hifigan_resblock_baked_weightnorm",
            lambda: bake_parametrized_weights(ResBlock(channels=64, kernel_size=3, dilations=[1, 3, 5])),
            torch.rand(1, 64, 64, dtype=torch.float32),
        ),
    ]
    results = [run_probe(probe) for probe in probes]
    result_path = OUT_DIR / "iree_vulkan_subgraph_matrix.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")

    for result in results:
        vulkan = result.get("vulkan_compare")
        if vulkan:
            print(
                f"{result['name']}: vulkan ok max_abs={vulkan['max_abs_error']:.3e} "
                f"allclose={vulkan['allclose_1e_4']}"
            )
        else:
            print(f"{result['name']}: {result['status']} vulkan={result.get('vulkan_compile')}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
