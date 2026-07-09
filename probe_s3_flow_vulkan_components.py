#!/usr/bin/env python3
"""IREE Vulkan probes for real-weight S3 flow estimator components."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor
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


class TransformerWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            timestep=timestep,
        )


class TransformerNoTimestepWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )


class TransformerManualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_hidden_states = self.block.norm1(hidden_states)
        hidden_states = self.block.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states
        norm_hidden_states = self.block.norm3(hidden_states)
        hidden_states = self.block.ff(norm_hidden_states) + hidden_states
        return hidden_states


class AttentionWrapper(nn.Module):
    def __init__(self, attn: nn.Module) -> None:
        super().__init__()
        self.attn = attn

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden_states, attention_mask=attention_mask)


class FeedForwardWrapper(nn.Module):
    def __init__(self, ff: nn.Module) -> None:
        super().__init__()
        self.ff = ff

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class LayerNormWrapper(nn.Module):
    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class ManualLayerNormWrapper(nn.Module):
    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.eps = norm.eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class AttentionResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.block.norm1(hidden_states)
        return self.block.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states


class FeedForwardResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.block.norm3(hidden_states)
        return self.block.ff(norm_hidden_states) + hidden_states


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


def export_probe(probe: Probe) -> dict[str, Any]:
    probe_dir = OUT_DIR / probe.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    module = probe.module.cpu().eval()
    args = tuple(arg.cpu().contiguous() for arg in probe.args)

    with torch.inference_mode():
        expected = module(*args).detach().cpu().numpy()

    input_paths = []
    for index, arg in enumerate(args):
        path = probe_dir / f"input_{index}.npy"
        np.save(path, arg.detach().cpu().numpy())
        input_paths.append(path)
    expected_path = probe_dir / "torch_output.npy"
    np.save(expected_path, expected)

    exported = aot.export(module, args=args, module_name=probe.name, function_name="forward")
    mlir_path = probe_dir / f"{probe.name}.mlir"
    exported.save_mlir(mlir_path)
    graph_path = probe_dir / f"{probe.name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, args).graph_module) + "\n")

    return {
        "name": probe.name,
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "inputs": [path.as_posix() for path in input_paths],
        "expected": expected_path.as_posix(),
        "expected_shape": list(expected.shape),
    }


def compile_and_run(probe_result: dict[str, Any], compile_timeout: int, run_timeout: int) -> dict[str, Any]:
    probe_dir = Path(probe_result["mlir"]).parent
    vmfb = probe_dir / f"{probe_result['name']}_vulkan_gfx1013.vmfb"
    output = probe_dir / "iree_vulkan_output.npy"
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

    if output.exists():
        output.unlink()
    run_cmdline = [
        IREE_RUN.as_posix(),
        f"--module={vmfb}",
        "--device=vulkan",
        "--function=forward",
        *[f"--input=@{path}" for path in probe_result["inputs"]],
        f"--output=@{output}",
    ]
    run_result = run_cmd(run_cmdline, timeout=run_timeout)
    result["run"] = run_result
    if run_result["returncode"] != 0:
        result["status"] = "run_failed"
        return result
    if not output.exists():
        result["status"] = "missing_output"
        return result

    actual = np.load(output)
    expected = np.load(probe_result["expected"])
    result["output"] = output.as_posix()
    result["compare"] = diff_summary(actual, expected)
    result["status"] = "ok"
    return result


def build_probes(estimator: nn.Module, frames: int) -> dict[str, Probe]:
    torch.manual_seed(20260708 + frames)
    mask = torch.ones(1, 1, frames, dtype=torch.float32)
    half_mask = torch.ones(1, 1, frames // 2, dtype=torch.float32)
    attention_mask = torch.zeros(1, frames, frames, dtype=torch.float32)
    half_attention_mask = torch.zeros(1, frames // 2, frames // 2, dtype=torch.float32)
    # Actual S3 estimator calls use mask_to_bias(add_optional_chunk_mask(...)),
    # which is shaped [batch, 1, time] for the current non-streaming path.
    attention_bias = torch.zeros(1, 1, frames, dtype=torch.float32)
    time_emb = torch.randn(1, 1024, dtype=torch.float32)

    return {
        "down_resnet": Probe(
            name=f"s3_flow_down_resnet_t{frames}",
            module=estimator.down_blocks[0][0],
            args=(torch.randn(1, 320, frames), mask, time_emb),
        ),
        "down_transformer": Probe(
            name=f"s3_flow_down_transformer_t{frames}",
            module=TransformerWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_mask, time_emb),
        ),
        "down_transformer_bias": Probe(
            name=f"s3_flow_down_transformer_bias_t{frames}",
            module=TransformerWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_bias, time_emb),
        ),
        "down_transformer_notimestep": Probe(
            name=f"s3_flow_down_transformer_notimestep_t{frames}",
            module=TransformerNoTimestepWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_mask),
        ),
        "down_transformer_manual": Probe(
            name=f"s3_flow_down_transformer_manual_t{frames}",
            module=TransformerManualWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_mask),
        ),
        "down_attention": Probe(
            name=f"s3_flow_down_attention_t{frames}",
            module=AttentionWrapper(estimator.down_blocks[0][1][0].attn1),
            args=(torch.randn(1, frames, 256), attention_mask),
        ),
        "down_attention_bias": Probe(
            name=f"s3_flow_down_attention_bias_t{frames}",
            module=AttentionWrapper(estimator.down_blocks[0][1][0].attn1),
            args=(torch.randn(1, frames, 256), attention_bias),
        ),
        "down_norm1": Probe(
            name=f"s3_flow_down_norm1_t{frames}",
            module=LayerNormWrapper(estimator.down_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames, 256),),
        ),
        "down_norm1_manual": Probe(
            name=f"s3_flow_down_norm1_manual_t{frames}",
            module=ManualLayerNormWrapper(estimator.down_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames, 256),),
        ),
        "down_attention_residual": Probe(
            name=f"s3_flow_down_attention_residual_t{frames}",
            module=AttentionResidualWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_mask),
        ),
        "down_ff": Probe(
            name=f"s3_flow_down_ff_t{frames}",
            module=FeedForwardWrapper(estimator.down_blocks[0][1][0].ff),
            args=(torch.randn(1, frames, 256),),
        ),
        "down_ff_residual": Probe(
            name=f"s3_flow_down_ff_residual_t{frames}",
            module=FeedForwardResidualWrapper(estimator.down_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256),),
        ),
        "downsample": Probe(
            name=f"s3_flow_downsample_t{frames}",
            module=estimator.down_blocks[0][2],
            args=(torch.randn(1, 256, frames),),
        ),
        "mid_transformer": Probe(
            name=f"s3_flow_mid_transformer_t{frames // 2}",
            module=TransformerWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames // 2, 256), half_attention_mask, time_emb),
        ),
        "mid_resnet_full": Probe(
            name=f"s3_flow_mid_resnet_full_t{frames}",
            module=estimator.mid_blocks[0][0],
            args=(torch.randn(1, 256, frames), mask, time_emb),
        ),
        "mid_transformer_full": Probe(
            name=f"s3_flow_mid_transformer_full_t{frames}",
            module=TransformerWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_bias, time_emb),
        ),
        "mid_transformer_notimestep": Probe(
            name=f"s3_flow_mid_transformer_notimestep_t{frames // 2}",
            module=TransformerNoTimestepWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames // 2, 256), half_attention_mask),
        ),
        "mid_transformer_manual": Probe(
            name=f"s3_flow_mid_transformer_manual_t{frames // 2}",
            module=TransformerManualWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames // 2, 256), half_attention_mask),
        ),
        "mid_attention": Probe(
            name=f"s3_flow_mid_attention_t{frames // 2}",
            module=AttentionWrapper(estimator.mid_blocks[0][1][0].attn1),
            args=(torch.randn(1, frames // 2, 256), half_attention_mask),
        ),
        "mid_attention_full": Probe(
            name=f"s3_flow_mid_attention_full_t{frames}",
            module=AttentionWrapper(estimator.mid_blocks[0][1][0].attn1),
            args=(torch.randn(1, frames, 256), attention_bias),
        ),
        "mid_norm1": Probe(
            name=f"s3_flow_mid_norm1_t{frames // 2}",
            module=LayerNormWrapper(estimator.mid_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames // 2, 256),),
        ),
        "mid_norm1_full": Probe(
            name=f"s3_flow_mid_norm1_full_t{frames}",
            module=LayerNormWrapper(estimator.mid_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames, 256),),
        ),
        "mid_norm1_manual": Probe(
            name=f"s3_flow_mid_norm1_manual_t{frames // 2}",
            module=ManualLayerNormWrapper(estimator.mid_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames // 2, 256),),
        ),
        "mid_attention_residual": Probe(
            name=f"s3_flow_mid_attention_residual_t{frames // 2}",
            module=AttentionResidualWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames // 2, 256), half_attention_mask),
        ),
        "mid_ff": Probe(
            name=f"s3_flow_mid_ff_t{frames // 2}",
            module=FeedForwardWrapper(estimator.mid_blocks[0][1][0].ff),
            args=(torch.randn(1, frames // 2, 256),),
        ),
        "mid_ff_full": Probe(
            name=f"s3_flow_mid_ff_full_t{frames}",
            module=FeedForwardWrapper(estimator.mid_blocks[0][1][0].ff),
            args=(torch.randn(1, frames, 256),),
        ),
        "mid_ff_residual": Probe(
            name=f"s3_flow_mid_ff_residual_t{frames // 2}",
            module=FeedForwardResidualWrapper(estimator.mid_blocks[0][1][0]),
            args=(torch.randn(1, frames // 2, 256),),
        ),
        "up_resnet": Probe(
            name=f"s3_flow_up_resnet_t{frames}",
            module=estimator.up_blocks[0][0],
            args=(torch.randn(1, 512, frames), mask, time_emb),
        ),
        "up_transformer": Probe(
            name=f"s3_flow_up_transformer_t{frames}",
            module=TransformerWrapper(estimator.up_blocks[0][1][0]),
            args=(torch.randn(1, frames, 256), attention_bias, time_emb),
        ),
        "upsample": Probe(
            name=f"s3_flow_upsample_t{frames}",
            module=estimator.up_blocks[0][2],
            args=(torch.randn(1, 256, frames),),
        ),
        "up_attention": Probe(
            name=f"s3_flow_up_attention_t{frames}",
            module=AttentionWrapper(estimator.up_blocks[0][1][0].attn1),
            args=(torch.randn(1, frames, 256), attention_bias),
        ),
        "up_norm1": Probe(
            name=f"s3_flow_up_norm1_t{frames}",
            module=LayerNormWrapper(estimator.up_blocks[0][1][0].norm1),
            args=(torch.randn(1, frames, 256),),
        ),
        "up_ff": Probe(
            name=f"s3_flow_up_ff_t{frames}",
            module=FeedForwardWrapper(estimator.up_blocks[0][1][0].ff),
            args=(torch.randn(1, frames, 256),),
        ),
        "final_block": Probe(
            name=f"s3_flow_final_block_t{frames}",
            module=estimator.final_block,
            args=(torch.randn(1, 256, frames), mask),
        ),
        "final_proj": Probe(
            name=f"s3_flow_final_proj_t{frames}",
            module=estimator.final_proj,
            args=(torch.randn(1, 256, frames),),
        ),
    }


def force_eager_attention(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(AttnProcessor())
            count += 1
    return count


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
                f"  vulkan={vulkan['status']} max_abs={compare['max_abs_error']:.3e} "
                f"allclose_1e_4={compare['allclose_1e_4']}"
            )
        else:
            print(f"  vulkan={vulkan.get('status')}")
    elif result["status"] == "failed":
        print(f"  {result.get('error_type')}: {result.get('error')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument(
        "--probes",
        default="downsample,final_proj,down_resnet,down_transformer",
        help="Comma-separated probes to run.",
    )
    parser.add_argument("--compile-timeout", type=int, default=240)
    parser.add_argument("--run-timeout", type=int, default=60)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--skip-vulkan", action="store_true")
    parser.add_argument("--eager-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "s3_flow_component_probe_latest.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    estimator = model.s3gen.flow.decoder.estimator.cpu().eval()
    attention_processors_changed = force_eager_attention(estimator) if args.eager_attention else 0
    load_seconds = time.perf_counter() - load_started

    probes = build_probes(estimator, args.frames)
    selected = [name.strip() for name in args.probes.split(",") if name.strip()]
    unknown = [name for name in selected if name not in probes]
    if unknown:
        raise SystemExit(f"Unknown probes: {unknown}; known={sorted(probes)}")

    results = [
        attempt_probe(probes[name], args.compile_timeout, args.run_timeout, args.skip_vulkan)
        for name in selected
    ]
    report = {
        "frames": args.frames,
        "selected": selected,
        "load_seconds": load_seconds,
        "eager_attention": args.eager_attention,
        "attention_processors_changed": attention_processors_changed,
        "iree_flags": list(IREE_FLAGS),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
