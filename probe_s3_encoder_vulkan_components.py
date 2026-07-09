#!/usr/bin/env python3
"""IREE Vulkan probes for S3 flow encoder components."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "s3_flow_vulkan_components"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"

IREE_FLAGS = (
    "--iree-vulkan-target=gfx1013",
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


@dataclass(frozen=True)
class Probe:
    name: str
    module: nn.Module
    args: tuple[torch.Tensor, ...]


class EncoderLayerOutputWrapper(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        chunk_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _, _, _ = self.layer(hidden_states, chunk_mask, pos_emb, mask_pad)
        return hidden_states


class EncoderLayerFloatMaskOutputWrapper(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        chunk_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _, _, _ = self.layer(
            hidden_states,
            chunk_mask > 0.5,
            pos_emb,
            mask_pad > 0.5,
        )
        return hidden_states


class UpLayerOutputWrapper(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.layer(hidden_states, lengths)
        return hidden_states


class SubsamplingFloatMaskWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states, pos_emb, mask = self.module(hidden_states, mask > 0.5)
        return hidden_states, pos_emb, mask.to(hidden_states.dtype)


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_tail": completed.stdout.splitlines()[-30:],
            "stderr_tail": completed.stderr.splitlines()[-60:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-30:]
            if isinstance(exc.stdout, str)
            else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-60:]
            if isinstance(exc.stderr, str)
            else [],
        }


def diff_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def normalize_tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in value):
        return tuple(value)
    raise TypeError(f"Expected tensor output or tuple/list of tensors, got {type(value).__name__}")


def export_probe(probe: Probe) -> dict[str, Any]:
    probe_dir = OUT_DIR / probe.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    module = probe.module.cpu().eval()
    args = tuple(arg.cpu().contiguous() for arg in probe.args)

    with torch.inference_mode():
        expected_outputs = normalize_tensor_outputs(module(*args))

    input_paths = []
    for index, arg in enumerate(args):
        path = probe_dir / f"input_{index}.npy"
        np.save(path, arg.detach().cpu().numpy())
        input_paths.append(path)
    expected_paths = []
    expected_shapes = []
    for index, expected_tensor in enumerate(expected_outputs):
        expected = expected_tensor.detach().cpu().numpy()
        path_name = "torch_output.npy" if len(expected_outputs) == 1 else f"torch_output_{index}.npy"
        expected_path = probe_dir / path_name
        np.save(expected_path, expected)
        expected_paths.append(expected_path)
        expected_shapes.append(list(expected.shape))

    exported = aot.export(module, args=args, module_name=probe.name, function_name="forward")
    mlir_path = probe_dir / f"{probe.name}.mlir"
    exported.save_mlir(mlir_path)
    graph_path = probe_dir / f"{probe.name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, args).graph_module) + "\n")

    result = {
        "name": probe.name,
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "inputs": [path.as_posix() for path in input_paths],
        "expected_outputs": [path.as_posix() for path in expected_paths],
        "expected_shapes": expected_shapes,
    }
    if len(expected_paths) == 1:
        result["expected"] = expected_paths[0].as_posix()
        result["expected_shape"] = expected_shapes[0]
    return result


def compile_and_run(probe_result: dict[str, Any], compile_timeout: int, run_timeout: int) -> dict[str, Any]:
    probe_dir = Path(probe_result["mlir"]).parent
    vmfb = probe_dir / f"{probe_result['name']}_vulkan_gfx1013.vmfb"
    expected_output_values = probe_result.get("expected_outputs")
    if expected_output_values is None:
        expected_output_values = [probe_result["expected"]]
    expected_paths = [Path(path) for path in expected_output_values]
    output_paths = [
        probe_dir / ("iree_vulkan_output.npy" if len(expected_paths) == 1 else f"iree_vulkan_output_{index}.npy")
        for index in range(len(expected_paths))
    ]
    compile_cmd = [
        IREE_COMPILE.as_posix(),
        probe_result["mlir"],
        "--iree-hal-target-backends=vulkan-spirv",
        *IREE_FLAGS,
        f"-o={vmfb}",
    ]
    compile_result = run_cmd(compile_cmd, timeout=compile_timeout)
    result: dict[str, Any] = {
        "compile": compile_result,
        "vmfb": vmfb.as_posix(),
    }
    if compile_result["returncode"] != 0:
        result["status"] = "compile_failed"
        return result

    for output in output_paths:
        if output.exists():
            output.unlink()
    run_cmdline = [
        IREE_RUN.as_posix(),
        f"--module={vmfb}",
        "--device=vulkan",
        "--function=forward",
        *[f"--input=@{path}" for path in probe_result["inputs"]],
        *[f"--output=@{path}" for path in output_paths],
    ]
    run_result = run_cmd(run_cmdline, timeout=run_timeout)
    result["run"] = run_result
    if run_result["returncode"] != 0:
        result["status"] = "run_failed"
        return result
    missing_outputs = [path.as_posix() for path in output_paths if not path.exists()]
    if missing_outputs:
        result["status"] = "missing_output"
        result["missing_outputs"] = missing_outputs
        return result

    compares = [
        diff_summary(np.load(output), np.load(expected))
        for output, expected in zip(output_paths, expected_paths, strict=True)
    ]
    result["outputs"] = [path.as_posix() for path in output_paths]
    result["compare_outputs"] = compares
    result["compare"] = {
        "outputs": len(compares),
        "max_abs_error": max(compare["max_abs_error"] for compare in compares),
        "mean_abs_error": max(compare["mean_abs_error"] for compare in compares),
        "p95_abs_error": max(compare["p95_abs_error"] for compare in compares),
        "allclose_1e_4": all(compare["allclose_1e_4"] for compare in compares),
        "allclose_1e_3": all(compare["allclose_1e_3"] for compare in compares),
    }
    if len(output_paths) == 1:
        result["output"] = output_paths[0].as_posix()
    result["status"] = "ok"
    return result


def build_probes(encoder: nn.Module, token_frames: int, up_frames: int) -> dict[str, Probe]:
    torch.manual_seed(20260708 + token_frames + up_frames)
    token_hidden = torch.randn(1, token_frames, 512, dtype=torch.float32)
    token_hidden_ct = torch.randn(1, 512, token_frames, dtype=torch.float32)
    token_mask = torch.ones(1, 1, token_frames, dtype=torch.bool)
    token_mask_float = torch.ones(1, 1, token_frames, dtype=torch.float32)
    token_pos = torch.randn(1, token_frames * 2 - 1, 512, dtype=torch.float32)
    up_hidden = torch.randn(1, up_frames, 512, dtype=torch.float32)
    up_mask = torch.ones(1, 1, up_frames, dtype=torch.bool)
    up_mask_float = torch.ones(1, 1, up_frames, dtype=torch.float32)
    up_pos = torch.randn(1, up_frames * 2 - 1, 512, dtype=torch.float32)
    lengths = torch.tensor([token_frames], dtype=torch.int64)

    return {
        "embed_fmask": Probe(
            name=f"s3_encoder_embed_fmask_t{token_frames}",
            module=SubsamplingFloatMaskWrapper(encoder.embed),
            args=(token_hidden, token_mask_float),
        ),
        "pre_lookahead": Probe(
            name=f"s3_encoder_pre_lookahead_t{token_frames}",
            module=encoder.pre_lookahead_layer,
            args=(token_hidden,),
        ),
        "encoder_layer": Probe(
            name=f"s3_encoder_layer0_t{token_frames}",
            module=EncoderLayerOutputWrapper(encoder.encoders[0]),
            args=(token_hidden, token_mask, token_pos, token_mask),
        ),
        "encoder_layer_fmask": Probe(
            name=f"s3_encoder_layer0_fmask_t{token_frames}",
            module=EncoderLayerFloatMaskOutputWrapper(encoder.encoders[0]),
            args=(token_hidden, token_mask_float, token_pos, token_mask_float),
        ),
        "up_layer": Probe(
            name=f"s3_encoder_up_layer_t{token_frames}",
            module=UpLayerOutputWrapper(encoder.up_layer),
            args=(token_hidden_ct, lengths),
        ),
        "up_embed_fmask": Probe(
            name=f"s3_encoder_up_embed_fmask_t{up_frames}",
            module=SubsamplingFloatMaskWrapper(encoder.up_embed),
            args=(up_hidden, up_mask_float),
        ),
        "up_encoder_layer": Probe(
            name=f"s3_encoder_up_layer0_t{up_frames}",
            module=EncoderLayerOutputWrapper(encoder.up_encoders[0]),
            args=(up_hidden, up_mask, up_pos, up_mask),
        ),
        "up_encoder_layer_fmask": Probe(
            name=f"s3_encoder_up_layer0_fmask_t{up_frames}",
            module=EncoderLayerFloatMaskOutputWrapper(encoder.up_encoders[0]),
            args=(up_hidden, up_mask_float, up_pos, up_mask_float),
        ),
        "after_norm": Probe(
            name=f"s3_encoder_after_norm_t{up_frames}",
            module=encoder.after_norm,
            args=(up_hidden,),
        ),
    }


def attempt_probe(probe: Probe, compile_timeout: int, run_timeout: int, skip_vulkan: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"name": probe.name}
    try:
        result.update(export_probe(probe))
        if not skip_vulkan:
            result["iree_vulkan"] = compile_and_run(result, compile_timeout, run_timeout)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
    result["seconds"] = time.perf_counter() - started
    print(f"{result['name']}: {result['status']} {result['seconds']:.1f}s")
    vulkan = result.get("iree_vulkan", {})
    if vulkan:
        compare = vulkan.get("compare", {})
        if compare:
            print(
                f"  vulkan={vulkan['status']} outputs={compare.get('outputs', 1)} "
                f"max_abs={compare['max_abs_error']:.3e} "
                f"allclose_1e_4={compare['allclose_1e_4']}"
            )
        else:
            print(f"  vulkan={vulkan.get('status')}")
    elif result["status"] == "failed":
        print(f"  {result.get('error_type')}: {result.get('error')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-frames", type=int, default=605)
    parser.add_argument("--up-frames", type=int, default=1210)
    parser.add_argument(
        "--probes",
        default="pre_lookahead,encoder_layer_fmask,up_layer,up_encoder_layer_fmask,after_norm",
        help="Comma-separated probes to run.",
    )
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--skip-vulkan", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "s3_encoder_component_probe_latest.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    encoder = model.s3gen.flow.encoder.cpu().eval()
    load_seconds = time.perf_counter() - load_started

    probes = build_probes(encoder, args.token_frames, args.up_frames)
    token_hidden = probes["encoder_layer_fmask"].args[0]
    token_mask_float = probes["encoder_layer_fmask"].args[1]
    token_pos = probes["encoder_layer_fmask"].args[2]
    up_hidden = probes["up_encoder_layer_fmask"].args[0]
    up_mask_float = probes["up_encoder_layer_fmask"].args[1]
    up_pos = probes["up_encoder_layer_fmask"].args[2]
    for index, layer in enumerate(encoder.encoders):
        probes[f"encoder_layer_fmask_{index}"] = Probe(
            name=f"s3_encoder_layer{index}_fmask_t{args.token_frames}",
            module=EncoderLayerFloatMaskOutputWrapper(layer),
            args=(token_hidden, token_mask_float, token_pos, token_mask_float),
        )
    for index, layer in enumerate(encoder.up_encoders):
        probes[f"up_encoder_layer_fmask_{index}"] = Probe(
            name=f"s3_encoder_up_layer{index}_fmask_t{args.up_frames}",
            module=EncoderLayerFloatMaskOutputWrapper(layer),
            args=(up_hidden, up_mask_float, up_pos, up_mask_float),
        )
    selected = [name.strip() for name in args.probes.split(",") if name.strip()]
    unknown = [name for name in selected if name not in probes]
    if unknown:
        raise SystemExit(f"Unknown probes: {unknown}; known={sorted(probes)}")

    results = [
        attempt_probe(probes[name], args.compile_timeout, args.run_timeout, args.skip_vulkan)
        for name in selected
    ]
    report = {
        "token_frames": args.token_frames,
        "up_frames": args.up_frames,
        "selected": selected,
        "load_seconds": load_seconds,
        "iree_flags": list(IREE_FLAGS),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
