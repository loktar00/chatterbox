#!/usr/bin/env python3
"""Export and validate cached GPT-2 T3 stack probes that return new K/V cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"
FLAG_DIR = OUT_DIR / "flag_variants"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"

DEFAULT_COMPILE_FLAGS = (
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


class CachedManualBlockKVWrapper(nn.Module):
    """Manual GPT-2 cached block that also returns the token's new K/V tensors."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.head_dim
        self.split_size = block.attn.split_size
        self.scale_attn_weights = block.attn.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = block.attn.scale_attn_by_inverse_layer_idx
        self.layer_idx = block.attn.layer_idx or 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)

        query, new_key, new_value = self.block.attn.c_attn(hidden_states).split(
            self.split_size,
            dim=2,
        )
        query_shape = (*query.shape[:-1], -1, self.head_dim)
        query = query.view(query_shape).transpose(1, 2)
        key_shape = (*new_key.shape[:-1], -1, self.head_dim)
        new_key = new_key.view(key_shape).transpose(1, 2)
        new_value = new_value.view(key_shape).transpose(1, 2)

        key = torch.cat((past_key, new_key), dim=2)
        value = torch.cat((past_value, new_value), dim=2)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        if self.scale_attn_weights:
            attn_weights = attn_weights / math.sqrt(float(self.head_dim))
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)
        attn_weights = F.softmax(attn_weights, dim=-1).type(value.dtype)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.block.attn.c_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.block.ln_2(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_key.contiguous(), new_value.contiguous()


class CachedManualStackKVWrapper(nn.Module):
    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CachedManualBlockKVWrapper(block) for block in blocks])

    def forward(self, hidden_states: torch.Tensor, *past_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        new_cache: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden_states, new_key, new_value = block(
                hidden_states,
                past_tensors[index * 2],
                past_tensors[index * 2 + 1],
            )
            new_cache.extend((new_key, new_value))
        return (hidden_states, *new_cache)


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
            "stderr_tail": completed.stderr.splitlines()[-80:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-30:]
            if isinstance(exc.stdout, str)
            else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-80:]
            if isinstance(exc.stderr, str)
            else [],
        }


def compare(actual_path: Path, expected_path: Path) -> dict[str, Any]:
    actual = np.load(actual_path)
    expected = np.load(expected_path)
    shape_match = actual.shape == expected.shape
    if not shape_match:
        return {
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "shape_match": False,
        }

    diff = np.abs(actual - expected)
    return {
        "shape": list(actual.shape),
        "shape_match": True,
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def export_probe(
    name: str,
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_labels: list[str],
) -> dict[str, Any]:
    module.eval()
    with torch.inference_mode():
        expected_outputs = module(*inputs)

    input_paths = []
    for index, tensor in enumerate(inputs):
        path = OUT_DIR / f"{name}_input_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        input_paths.append(path)

    expected_paths = []
    output_shapes = []
    for index, tensor in enumerate(expected_outputs):
        path = OUT_DIR / f"{name}_torch_output_{index}.npy"
        array = tensor.detach().cpu().numpy()
        np.save(path, array)
        expected_paths.append(path)
        output_shapes.append(list(array.shape))

    mlir_path = OUT_DIR / f"{name}.mlir"
    graph_path = OUT_DIR / f"{name}.torch_export.txt"
    exported = aot.export(module, args=inputs, module_name=name, function_name="forward")
    exported.save_mlir(mlir_path)
    graph_path.write_text(str(torch.export.export(module, inputs).graph_module) + "\n")

    return {
        "name": name,
        "inputs": [path.as_posix() for path in input_paths],
        "expected": [path.as_posix() for path in expected_paths],
        "labels": output_labels,
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "output_shapes": output_shapes,
    }


def compile_and_run(
    probe: dict[str, Any],
    compile_timeout: int,
    run_timeout: int,
    compile_flags: tuple[str, ...],
) -> dict[str, Any]:
    subgraph = probe["name"]
    variant = "split_matmul4_split_reduction"
    variant_dir = FLAG_DIR / subgraph / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    vmfb = variant_dir / f"{subgraph}_{variant}.vmfb"
    compile_cmd = [
        IREE_COMPILE.as_posix(),
        probe["mlir"],
        "--iree-hal-target-backends=vulkan-spirv",
        "--iree-vulkan-target=gfx1013",
        *compile_flags,
        f"-o={vmfb}",
    ]
    compile_result = run_cmd(compile_cmd, compile_timeout)
    result: dict[str, Any] = {
        "variant": variant,
        "compile_flags": list(compile_flags),
        "compile": compile_result,
        "vmfb": vmfb.as_posix(),
    }
    if compile_result["returncode"] != 0:
        result["status"] = "compile_failed"
        return result

    output_paths = [
        variant_dir / f"{subgraph}_{variant}_vulkan_output_{index}.npy"
        for index in range(len(probe["expected"]))
    ]
    for path in output_paths:
        if path.exists():
            path.unlink()

    run_cmdline = [
        IREE_RUN.as_posix(),
        f"--module={vmfb}",
        "--device=vulkan",
        "--function=forward",
        *[f"--input=@{input_path}" for input_path in probe["inputs"]],
        *[f"--output=@{output_path}" for output_path in output_paths],
    ]
    run_result = run_cmd(run_cmdline, run_timeout)
    result["run"] = run_result
    if run_result["returncode"] != 0:
        result["status"] = "run_failed"
        return result

    missing = [path.as_posix() for path in output_paths if not path.exists()]
    if missing:
        result["status"] = "missing_output"
        result["missing_outputs"] = missing
        return result

    comparisons = []
    for index, (actual_path, expected_path, label) in enumerate(
        zip(output_paths, probe["expected"], probe["labels"]),
    ):
        comparisons.append(
            {
                "index": index,
                "label": label,
                "actual": actual_path.as_posix(),
                "expected": expected_path,
                "compare": compare(actual_path, Path(expected_path)),
            },
        )
    result["outputs"] = [path.as_posix() for path in output_paths]
    result["comparisons"] = comparisons
    result["status"] = "ok"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--past-len", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--compile-timeout", type=int, default=900)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument("--skip-compile-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_cached_stack_kv_probe_latest.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FLAG_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    end_layer = args.start_layer + args.layers
    blocks = [t3.tfmr.h[index].eval() for index in range(args.start_layer, end_layer)]
    wrapper = CachedManualStackKVWrapper(blocks).eval()
    first = wrapper.blocks[0]

    inputs: list[torch.Tensor] = [torch.randn(1, 1, t3.dim, dtype=torch.float32)]
    output_labels = ["hidden"]
    for layer_index in range(args.start_layer, end_layer):
        inputs.append(torch.randn(1, first.num_heads, args.past_len, first.head_dim, dtype=torch.float32))
        inputs.append(torch.randn(1, first.num_heads, args.past_len, first.head_dim, dtype=torch.float32))
        output_labels.extend((f"layer{layer_index}_new_key", f"layer{layer_index}_new_value"))

    name = f"gpt2_cached_stack_kv_s{args.start_layer}_l{args.layers}_p{args.past_len}_t1"
    probe = export_probe(name, wrapper, tuple(inputs), output_labels)

    run_result = None
    if not args.skip_compile_run:
        run_result = compile_and_run(
            probe,
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
            compile_flags=DEFAULT_COMPILE_FLAGS,
        )

    report: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "start_layer": args.start_layer,
        "end_layer": end_layer,
        "layers": args.layers,
        "past_len": args.past_len,
        "probe": probe,
        "run_result": run_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"{name}: {probe['output_shapes']}")
    if run_result is not None:
        print(f"Vulkan status: {run_result['status']}")
        for comparison in run_result.get("comparisons", []):
            detail = comparison["compare"]
            max_abs = detail.get("max_abs_error")
            close = detail.get("allclose_1e_4")
            if isinstance(max_abs, float):
                print(f"  {comparison['label']}: max_abs={max_abs:.3e} allclose_1e_4={close}")


if __name__ == "__main__":
    main()
