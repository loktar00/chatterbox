#!/usr/bin/env python3
"""Probe export/runtime feasibility for Chatterbox Turbo T3 subgraphs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"


class SpeechHeadWrapper(nn.Module):
    def __init__(self, speech_head: nn.Module) -> None:
        super().__init__()
        self.speech_head = speech_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.speech_head(hidden_states)


class GPT2BlockNoCacheWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.block(
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            use_cache=False,
            output_attentions=False,
        )
        return output[0]


class GPT2AttentionNoCacheWrapper(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.attention(
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            output_attentions=False,
        )
        return output[0]


class GPT2MLPWrapper(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class GPT2ProjectionWrapper(nn.Module):
    def __init__(self, projection: nn.Module) -> None:
        super().__init__()
        self.projection = projection

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


def run(cmd: list[str]) -> dict[str, Any]:
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
        "stdout_tail": completed.stdout.splitlines()[-12:],
        "stderr_tail": completed.stderr.splitlines()[-30:],
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


def attempt(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {"name": name}
    try:
        result.update(fn())
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
    result["seconds"] = time.perf_counter() - started
    print(f"{name}: {result['status']} {result['seconds']:.1f}s")
    if result["status"] == "failed":
        print(f"  {result['error_type']}: {result['error']}")
    return result


def torch_export_probe(module: nn.Module, args: tuple[torch.Tensor, ...], name: str) -> dict[str, Any]:
    exported = torch.export.export(module, args)
    graph_path = OUT_DIR / f"{name}.torch_export.txt"
    graph_path.write_text(str(exported.graph_module) + "\n")
    return {"graph": graph_path.as_posix()}


def iree_vulkan_probe(module: nn.Module, args: tuple[torch.Tensor, ...], name: str) -> dict[str, Any]:
    with torch.inference_mode():
        expected = module(*args).detach().cpu().numpy()

    input_paths = []
    for index, arg in enumerate(args):
        path = OUT_DIR / f"{name}_input_{index}.npy"
        np.save(path, arg.detach().cpu().numpy())
        input_paths.append(path)
    expected_path = OUT_DIR / f"{name}_torch_output.npy"
    np.save(expected_path, expected)

    exported = aot.export(module, args=args, module_name=name, function_name="forward")
    exported.session.set_flags("--iree-vulkan-target=gfx1013")
    mlir_path = OUT_DIR / f"{name}.mlir"
    exported.save_mlir(mlir_path)
    vmfb_path = OUT_DIR / f"{name}_vulkan_gfx1013.vmfb"
    exported.compile(vmfb_path, target_backends=("vulkan-spirv",))

    output_path = OUT_DIR / f"{name}_iree_vulkan_output.npy"
    cmd = [
        IREE_RUN.as_posix(),
        f"--module={vmfb_path}",
        "--device=vulkan",
        "--function=forward",
        *[f"--input=@{path}" for path in input_paths],
        f"--output=@{output_path}",
    ]
    run_result = run(cmd)
    result: dict[str, Any] = {
        "mlir": mlir_path.as_posix(),
        "vmfb": vmfb_path.as_posix(),
        "expected": expected_path.as_posix(),
        "run": run_result,
    }
    if run_result["returncode"] == 0 and output_path.exists():
        result["output"] = output_path.as_posix()
        result["compare"] = diff_summary(np.load(output_path), expected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--attention-impl", choices=["default", "eager"], default="default")
    parser.add_argument("--skip-iree", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "t3_exportability_latest.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(0)

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_seconds = time.perf_counter() - load_started
    t3 = model.t3.cpu().eval()
    if args.attention_impl != "default":
        t3.tfmr.config._attn_implementation = args.attention_impl
        for block in t3.tfmr.h:
            block.attn.config._attn_implementation = args.attention_impl

    hidden_1 = torch.randn(1, 1, 1024, dtype=torch.float32)
    hidden_16 = torch.randn(1, 16, 1024, dtype=torch.float32)
    hidden_4096 = torch.randn(1, 1, 4096, dtype=torch.float32)
    speech_head = SpeechHeadWrapper(t3.speech_head).eval()
    first_block = GPT2BlockNoCacheWrapper(t3.tfmr.h[0]).eval()
    first_attention = GPT2AttentionNoCacheWrapper(t3.tfmr.h[0].attn).eval()
    first_mlp = GPT2MLPWrapper(t3.tfmr.h[0].mlp).eval()
    first_mlp_fc = GPT2ProjectionWrapper(t3.tfmr.h[0].mlp.c_fc).eval()
    first_mlp_proj = GPT2ProjectionWrapper(t3.tfmr.h[0].mlp.c_proj).eval()

    probes: list[dict[str, Any]] = []
    probes.append(attempt(
        "speech_head_torch_export",
        lambda: torch_export_probe(speech_head, (hidden_1,), "speech_head_t1"),
    ))
    probes.append(attempt(
        "gpt2_block_t1_torch_export",
        lambda: torch_export_probe(first_block, (hidden_1,), "gpt2_block_t1"),
    ))
    probes.append(attempt(
        "gpt2_block_t16_torch_export",
        lambda: torch_export_probe(first_block, (hidden_16,), "gpt2_block_t16"),
    ))
    probes.append(attempt(
        "gpt2_attention_t1_torch_export",
        lambda: torch_export_probe(first_attention, (hidden_1,), "gpt2_attention_t1"),
    ))
    probes.append(attempt(
        "gpt2_mlp_t1_torch_export",
        lambda: torch_export_probe(first_mlp, (hidden_1,), "gpt2_mlp_t1"),
    ))
    probes.append(attempt(
        "gpt2_mlp_c_fc_t1_torch_export",
        lambda: torch_export_probe(first_mlp_fc, (hidden_1,), "gpt2_mlp_c_fc_t1"),
    ))
    probes.append(attempt(
        "gpt2_mlp_c_proj_t1_torch_export",
        lambda: torch_export_probe(first_mlp_proj, (hidden_4096,), "gpt2_mlp_c_proj_t1"),
    ))

    if not args.skip_iree:
        probes.append(attempt(
            "speech_head_t1_iree_vulkan",
            lambda: iree_vulkan_probe(speech_head, (hidden_1,), "speech_head_t1"),
        ))
        probes.append(attempt(
            "gpt2_block_t1_iree_vulkan",
            lambda: iree_vulkan_probe(first_block, (hidden_1,), "gpt2_block_t1"),
        ))
        probes.append(attempt(
            "gpt2_attention_t1_iree_vulkan",
            lambda: iree_vulkan_probe(first_attention, (hidden_1,), "gpt2_attention_t1"),
        ))
        probes.append(attempt(
            "gpt2_mlp_t1_iree_vulkan",
            lambda: iree_vulkan_probe(first_mlp, (hidden_1,), "gpt2_mlp_t1"),
        ))
        probes.append(attempt(
            "gpt2_mlp_c_fc_t1_iree_vulkan",
            lambda: iree_vulkan_probe(first_mlp_fc, (hidden_1,), "gpt2_mlp_c_fc_t1"),
        ))
        probes.append(attempt(
            "gpt2_mlp_c_proj_t1_iree_vulkan",
            lambda: iree_vulkan_probe(first_mlp_proj, (hidden_4096,), "gpt2_mlp_c_proj_t1"),
        ))

    results = {
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "model_load_seconds": load_seconds,
        "model": {
            "transformer_type": type(t3.tfmr).__name__,
            "attention_impl": t3.tfmr.config._attn_implementation,
            "layers": len(t3.tfmr.h),
            "hidden_size": t3.cfg.hidden_size,
            "heads": t3.cfg.n_head,
            "speech_vocab": t3.hp.speech_tokens_dict_size,
        },
        "probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
