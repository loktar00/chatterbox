#!/usr/bin/env python3
"""Stage-by-stage IREE Vulkan debug for the Chatterbox F0 predictor."""

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


ROOT = Path(__file__).resolve().parent
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"
OUT_DIR = ROOT / "exports" / "iree_vulkan_f0_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(self, name: str, build: Callable[[], nn.Module], sample: torch.Tensor) -> None:
        self.name = name
        self.build = build
        self.sample = sample


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


class LargeConvElu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(80, 512, kernel_size=3, padding=1)
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(self.conv(x))


class F0CondNetPrefix(nn.Module):
    def __init__(self, conv_count: int) -> None:
        super().__init__()
        f0 = bake_parametrized_weights(ConvRNNF0Predictor())
        self.layers = nn.Sequential(*list(f0.condnet.children())[: conv_count * 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class F0ClassifierOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        f0 = bake_parametrized_weights(ConvRNNF0Predictor())
        self.classifier = f0.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)


class Linear512To1NoSqueeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Linear512To1Squeeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class Linear512To2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class F0ClassifierAsConv1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        f0 = bake_parametrized_weights(ConvRNNF0Predictor())
        self.conv = nn.Conv1d(512, 1, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.copy_(f0.classifier.weight[:, :, None])
            self.conv.bias.copy_(f0.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.conv(x).squeeze(1)


class F0CondNetThenClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f0 = bake_parametrized_weights(ConvRNNF0Predictor())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f0.condnet(x)
        x = x.transpose(1, 2)
        return self.f0.classifier(x).squeeze(-1)


class F0CondNetThenConvClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f0 = bake_parametrized_weights(ConvRNNF0Predictor())
        self.conv = nn.Conv1d(512, 1, kernel_size=1)
        with torch.no_grad():
            self.conv.weight.copy_(self.f0.classifier.weight[:, :, None])
            self.conv.bias.copy_(self.f0.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f0.condnet(x)
        return self.conv(x).squeeze(1)


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
        "expected_min": float(expected.min()),
        "expected_max": float(expected.max()),
        "expected_mean": float(expected.mean()),
        "actual_min": float(output.min()),
        "actual_max": float(output.max()),
        "actual_mean": float(output.mean()),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(output, expected, atol=1e-4, rtol=1e-4)),
    }


def compile_and_run(exported: aot.ExportOutput, vmfb: Path, input_path: Path, output_path: Path, device: str) -> dict:
    return run([
        IREE_RUN.as_posix(),
        f"--module={vmfb}",
        f"--device={device}",
        "--function=forward",
        f"--input=@{input_path}",
        f"--output=@{output_path}",
    ])


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

        cpu_export = aot.export(model, args=(sample,), module_name=probe.name, function_name="forward")
        mlir_path = probe_dir / f"{probe.name}.mlir"
        cpu_export.save_mlir(mlir_path)

        cpu_vmfb = probe_dir / f"{probe.name}_llvm_cpu.vmfb"
        cpu_export.compile(cpu_vmfb, target_backends=("llvm-cpu",))
        cpu_output = probe_dir / "iree_cpu_output.npy"
        cpu_run = compile_and_run(cpu_export, cpu_vmfb, input_path, cpu_output, "local-task")

        result = {
            "name": probe.name,
            "status": "ok",
            "input": input_path.as_posix(),
            "torch_output": expected_path.as_posix(),
            "mlir": mlir_path.as_posix(),
            "cpu_run": cpu_run,
        }
        if cpu_run["returncode"] == 0 and cpu_output.exists():
            result["cpu_compare"] = compare(cpu_output, expected)

        vulkan_export = aot.export(model, args=(sample,), module_name=probe.name, function_name="forward")
        vulkan_export.session.set_flags("--iree-vulkan-target=gfx1013")
        vulkan_vmfb = probe_dir / f"{probe.name}_vulkan_gfx1013.vmfb"
        vulkan_export.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
        vulkan_output = probe_dir / "iree_vulkan_output.npy"
        vulkan_run = compile_and_run(vulkan_export, vulkan_vmfb, input_path, vulkan_output, "vulkan")
        result["vulkan_run"] = vulkan_run
        if vulkan_run["returncode"] == 0 and vulkan_output.exists():
            result["vulkan_compare"] = compare(vulkan_output, expected)

        return result
    except Exception as exc:
        return {
            "name": probe.name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }


def main() -> None:
    torch.manual_seed(1234)
    mel = torch.rand(1, 80, 64, dtype=torch.float32)
    hidden = torch.rand(1, 64, 512, dtype=torch.float32)

    probes = [
        Probe("large_conv1d_elu_80_512", LargeConvElu, mel),
        Probe("f0_condnet_prefix_1", lambda: F0CondNetPrefix(1), mel),
        Probe("f0_condnet_prefix_2", lambda: F0CondNetPrefix(2), mel),
        Probe("f0_condnet_prefix_3", lambda: F0CondNetPrefix(3), mel),
        Probe("f0_condnet_prefix_4", lambda: F0CondNetPrefix(4), mel),
        Probe("f0_condnet_prefix_5", lambda: F0CondNetPrefix(5), mel),
        Probe("linear_512_to_1_no_squeeze", Linear512To1NoSqueeze, hidden),
        Probe("linear_512_to_1_squeeze", Linear512To1Squeeze, hidden),
        Probe("linear_512_to_2", Linear512To2, hidden),
        Probe("f0_classifier_only", F0ClassifierOnly, hidden),
        Probe("f0_classifier_as_conv1d", F0ClassifierAsConv1d, hidden),
        Probe("f0_condnet_then_classifier", F0CondNetThenClassifier, mel),
        Probe("f0_condnet_then_conv_classifier", F0CondNetThenConvClassifier, mel),
    ]
    results = [run_probe(probe) for probe in probes]
    result_path = OUT_DIR / "iree_vulkan_f0_debug.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")

    for result in results:
        cmp = result.get("vulkan_compare")
        if cmp:
            print(
                f"{result['name']}: max_abs={cmp['max_abs_error']:.3e} "
                f"mean_abs={cmp['mean_abs_error']:.3e} allclose={cmp['allclose_1e_4']}"
            )
        else:
            print(f"{result['name']}: {result['status']} {result.get('error', '')}")
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
