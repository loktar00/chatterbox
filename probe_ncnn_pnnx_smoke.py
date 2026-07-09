#!/usr/bin/env python3
"""Tiny pnnx -> ncnn runtime smoke test.

This intentionally avoids loading Chatterbox. It verifies that the installed
pnnx converter and ncnn Python runtime can execute a small Torch subgraph on
CPU and, when available, Vulkan.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "ncnn_pnnx_smoke"
BENCH = ROOT / "exports" / "benchmarks"


class TinyConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.act = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(self.act(self.conv0(x)))


def init_weights(model: TinyConv) -> None:
    with torch.no_grad():
        for index, param in enumerate(model.parameters()):
            values = torch.linspace(-0.25, 0.25, param.numel(), dtype=torch.float32)
            param.copy_(values.reshape_as(param) + index * 0.01)


def pnnx_convert(model: torch.nn.Module, sample: torch.Tensor, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    traced_path = out_dir / "tiny_conv.pt"
    traced = torch.jit.trace(model.eval(), sample, check_trace=True)
    traced.save(traced_path.as_posix())

    pnnx = ROOT / ".venv" / "bin" / "pnnx"
    param_path = out_dir / "tiny_conv.ncnn.param"
    bin_path = out_dir / "tiny_conv.ncnn.bin"
    command = [
        pnnx.as_posix(),
        traced_path.as_posix(),
        "inputshape=[1,3,8,8]",
        "device=cpu",
        "optlevel=2",
        "fp16=0",
        f"ncnnparam={param_path.as_posix()}",
        f"ncnnbin={bin_path.as_posix()}",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True, cwd=ROOT.as_posix())
    return {
        "command": command,
        "seconds": time.perf_counter() - started,
        "returncode": result.returncode,
        "stdout_tail": (result.stdout or "").splitlines()[-80:],
        "stderr_tail": (result.stderr or "").splitlines()[-80:],
        "traced_path": traced_path.as_posix(),
        "param_path": param_path.as_posix(),
        "bin_path": bin_path.as_posix(),
        "param_exists": param_path.exists(),
        "bin_exists": bin_path.exists(),
    }


def ncnn_run(param_path: Path, bin_path: Path, sample: np.ndarray, *, use_vulkan: bool) -> dict[str, Any]:
    import ncnn

    gpu_count = 0
    gpu_error = None
    if use_vulkan:
        try:
            ncnn.create_gpu_instance()
            gpu_count = int(ncnn.get_gpu_count())
        except Exception as exc:  # noqa: BLE001
            gpu_error = repr(exc)
            use_vulkan = False

    try:
        net = ncnn.Net()
        net.opt.num_threads = 1
        net.opt.use_vulkan_compute = bool(use_vulkan and gpu_count > 0)
        load_param_rc = net.load_param(param_path.as_posix())
        load_model_rc = net.load_model(bin_path.as_posix())
        input_names = list(net.input_names())
        output_names = list(net.output_names())
        if not input_names or not output_names:
            raise RuntimeError(f"missing ncnn IO names: inputs={input_names}, outputs={output_names}")

        ncnn_sample = sample[0] if sample.ndim == 4 and sample.shape[0] == 1 else sample
        mat_in = ncnn.Mat(np.ascontiguousarray(ncnn_sample.astype(np.float32))).clone()
        ex = net.create_extractor()
        ex.input(input_names[0], mat_in)
        started = time.perf_counter()
        extract_rc, mat_out = ex.extract(output_names[0])
        elapsed = time.perf_counter() - started
        output = np.array(mat_out.numpy("f"), copy=True)
        return {
            "requested_vulkan": use_vulkan,
            "used_vulkan": bool(net.opt.use_vulkan_compute),
            "gpu_count": gpu_count,
            "gpu_error": gpu_error,
            "load_param_rc": int(load_param_rc),
            "load_model_rc": int(load_model_rc),
            "extract_rc": int(extract_rc),
            "seconds": elapsed,
            "input_names": input_names,
            "output_names": output_names,
            "output_shape": list(output.shape),
            "output": output.reshape(-1).tolist(),
        }
    finally:
        if use_vulkan:
            try:
                ncnn.destroy_gpu_instance()
            except Exception:
                pass


def summarize_against_torch(ncnn_result: dict[str, Any], torch_output: np.ndarray) -> dict[str, Any]:
    out_shape = tuple(ncnn_result["output_shape"])
    expected = torch_output
    if expected.shape != out_shape and expected.ndim == len(out_shape) + 1 and expected.shape[0] == 1:
        expected = expected[0]
    out = np.asarray(ncnn_result["output"], dtype=np.float32).reshape(out_shape)
    diff = np.abs(out - expected)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "expected_shape": list(expected.shape),
        "allclose_1e_5": bool(np.allclose(out, expected, rtol=1e-5, atol=1e-5)),
        "allclose_1e_4": bool(np.allclose(out, expected, rtol=1e-4, atol=1e-4)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BENCH / "ncnn_pnnx_smoke_2026-07-08.json")
    parser.add_argument(
        "--try-vulkan",
        action="store_true",
        help="Opt in to ncnn Vulkan. This segfaulted once on RADV/BC-250 during this audit.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.manual_seed(20260708)
    model = TinyConv().eval()
    init_weights(model)
    sample = torch.linspace(-1.0, 1.0, 1 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
    with torch.inference_mode():
        torch_output = model(sample).detach().cpu().numpy()

    conversion = pnnx_convert(model, sample, OUT_DIR)
    report: dict[str, Any] = {
        "description": "Tiny Torch -> pnnx -> ncnn runtime smoke test.",
        "conversion": conversion,
        "torch_output_shape": list(torch_output.shape),
        "runtime": {},
    }
    if conversion["returncode"] == 0 and conversion["param_exists"] and conversion["bin_exists"]:
        param_path = Path(conversion["param_path"])
        bin_path = Path(conversion["bin_path"])
        modes = [("cpu", False)]
        if args.try_vulkan:
            modes.append(("vulkan", True))
        for mode, use_vulkan in modes:
            try:
                result = ncnn_run(param_path, bin_path, sample.detach().cpu().numpy(), use_vulkan=use_vulkan)
                result["comparison"] = summarize_against_torch(result, torch_output)
            except Exception as exc:  # noqa: BLE001
                result = {"error": repr(exc), "requested_vulkan": use_vulkan}
            report["runtime"][mode] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"json={args.output}")
    for mode, result in report["runtime"].items():
        comparison = result.get("comparison")
        if comparison:
            print(
                f"{mode}: used_vulkan={result.get('used_vulkan')} "
                f"max_abs={comparison['max_abs_error']:.6g} "
                f"allclose_1e_5={comparison['allclose_1e_5']} "
                f"seconds={result.get('seconds'):.6f}"
            )
        else:
            print(f"{mode}: {result.get('error')}")


if __name__ == "__main__":
    main()
