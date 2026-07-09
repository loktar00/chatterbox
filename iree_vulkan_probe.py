#!/usr/bin/env python3
"""Tiny IREE Vulkan probe using a Chatterbox HiFiGAN submodule."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from iree.turbine import aot

from chatterbox.models.s3gen.hifigan import Snake


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "iree_vulkan_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    torch.manual_seed(0)
    model = Snake(64).cpu().eval()
    sample = torch.rand(1, 64, 64, dtype=torch.float32)

    with torch.no_grad():
        torch_output = model(sample).detach().cpu().numpy()

    input_path = OUT_DIR / "snake_input.npy"
    torch_output_path = OUT_DIR / "snake_torch_output.npy"
    np.save(input_path, sample.numpy())
    np.save(torch_output_path, torch_output)

    exported = aot.export(model, args=(sample,), module_name="chatterbox_snake", function_name="forward")
    mlir_path = OUT_DIR / "snake_torch_iree.mlir"
    exported.save_mlir(mlir_path)

    cpu_vmfb = OUT_DIR / "snake_llvm_cpu.vmfb"
    exported.compile(cpu_vmfb, target_backends=("llvm-cpu",))

    vulkan_vmfb = OUT_DIR / "snake_vulkan_gfx1013.vmfb"
    vulkan_compile = {"status": "not_run"}
    try:
        exported.session.set_flags("--iree-vulkan-target=gfx1013")
        exported.compile(vulkan_vmfb, target_backends=("vulkan-spirv",))
        vulkan_compile = {"status": "ok", "vmfb": vulkan_vmfb.as_posix()}
    except Exception as exc:
        vulkan_compile = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    cpu_output_path = OUT_DIR / "snake_iree_cpu_output.npy"
    cpu_run = run([
        (ROOT / ".venv" / "bin" / "iree-run-module").as_posix(),
        f"--module={cpu_vmfb}",
        "--device=local-task",
        "--function=forward",
        f"--input=@{input_path}",
        f"--output=@{cpu_output_path}",
    ])

    result = {
        "input": input_path.as_posix(),
        "torch_output": torch_output_path.as_posix(),
        "mlir": mlir_path.as_posix(),
        "cpu_vmfb": cpu_vmfb.as_posix(),
        "vulkan_compile": vulkan_compile,
        "cpu_run": {
            "returncode": cpu_run["returncode"],
            "stderr_tail": cpu_run["stderr"].splitlines()[-8:],
            "stdout_tail": cpu_run["stdout"].splitlines()[-8:],
        },
    }

    if cpu_run["returncode"] == 0 and cpu_output_path.exists():
        cpu_output = np.load(cpu_output_path)
        result["cpu_compare"] = {
            "max_abs_error": float(np.max(np.abs(cpu_output - torch_output))),
            "allclose_1e_5": bool(np.allclose(cpu_output, torch_output, atol=1e-5, rtol=1e-5)),
            "output": cpu_output_path.as_posix(),
        }

    if vulkan_compile["status"] == "ok":
        vulkan_output_path = OUT_DIR / "snake_iree_vulkan_output.npy"
        vulkan_run = run([
            (ROOT / ".venv" / "bin" / "iree-run-module").as_posix(),
            f"--module={vulkan_vmfb}",
            "--device=vulkan",
            "--function=forward",
            f"--input=@{input_path}",
            f"--output=@{vulkan_output_path}",
        ])
        result["vulkan_run"] = {
            "returncode": vulkan_run["returncode"],
            "stderr_tail": vulkan_run["stderr"].splitlines()[-12:],
            "stdout_tail": vulkan_run["stdout"].splitlines()[-12:],
        }
        if vulkan_run["returncode"] == 0 and vulkan_output_path.exists():
            vulkan_output = np.load(vulkan_output_path)
            result["vulkan_compare"] = {
                "max_abs_error": float(np.max(np.abs(vulkan_output - torch_output))),
                "allclose_1e_5": bool(np.allclose(vulkan_output, torch_output, atol=1e-5, rtol=1e-5)),
                "output": vulkan_output_path.as_posix(),
            }

    result_path = OUT_DIR / "iree_vulkan_probe_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"results={result_path}")


if __name__ == "__main__":
    main()
