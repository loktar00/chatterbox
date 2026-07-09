#!/usr/bin/env python3
"""Export and validate distinct S3 estimator components on IREE Vulkan."""

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
from diffusers.models.attention_processor import Attention, AttnProcessor
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "s3_flow_vulkan_components" / "distinct_estimator"
IREE_COMPILE = ROOT / ".venv" / "bin" / "iree-compile"
IREE_RUN = ROOT / ".venv" / "bin" / "iree-run-module"

IREE_FLAGS = (
    "--iree-vulkan-target=gfx1013",
    "--iree-dispatch-creation-split-matmul-reduction=4",
    "--iree-dispatch-creation-enable-split-reduction",
)


@dataclass(frozen=True)
class Probe:
    key: str
    name: str
    module: nn.Module
    args: tuple[torch.Tensor, ...]
    role: str
    index: tuple[int, ...]


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


def force_eager_attention(module: nn.Module) -> int:
    count = 0
    for child in module.modules():
        if isinstance(child, Attention):
            child.set_processor(AttnProcessor())
            count += 1
    return count


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
    diff = np.abs(actual.astype(np.float32, copy=False) - expected.astype(np.float32, copy=False))
    return {
        "shape": list(actual.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p95_abs_error": float(np.percentile(diff, 95)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "allclose_1e_4": bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(np.allclose(actual, expected, atol=1e-3, rtol=1e-3)),
    }


def export_probe(probe: Probe, force_export: bool) -> dict[str, Any]:
    probe_dir = OUT_DIR / probe.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = probe_dir / f"{probe.name}.mlir"
    expected_path = probe_dir / "torch_output.npy"
    input_paths = [probe_dir / f"input_{index}.npy" for index in range(len(probe.args))]

    if not force_export and mlir_path.exists() and expected_path.exists() and all(path.exists() for path in input_paths):
        expected = np.load(expected_path, mmap_mode="r")
        return {
            "name": probe.name,
            "key": probe.key,
            "role": probe.role,
            "index": list(probe.index),
            "status": "export_exists",
            "mlir": mlir_path.as_posix(),
            "inputs": [path.as_posix() for path in input_paths],
            "expected": expected_path.as_posix(),
            "expected_shape": list(expected.shape),
        }

    module = probe.module.cpu().eval()
    args = tuple(arg.cpu().contiguous() for arg in probe.args)
    with torch.inference_mode():
        expected = module(*args).detach().cpu().numpy()

    for index, arg in enumerate(args):
        np.save(input_paths[index], arg.detach().cpu().numpy())
    np.save(expected_path, expected)

    exported = aot.export(module, args=args, module_name=probe.name, function_name="forward")
    exported.save_mlir(mlir_path)
    graph_path = probe_dir / f"{probe.name}.torch_export.txt"
    graph_path.write_text(str(torch.export.export(module, args).graph_module) + "\n")

    return {
        "name": probe.name,
        "key": probe.key,
        "role": probe.role,
        "index": list(probe.index),
        "status": "exported",
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "inputs": [path.as_posix() for path in input_paths],
        "expected": expected_path.as_posix(),
        "expected_shape": list(expected.shape),
    }


def compile_and_run(
    probe_result: dict[str, Any],
    compile_timeout: int,
    run_timeout: int,
    force_compile: bool,
    skip_run: bool,
) -> dict[str, Any]:
    probe_dir = Path(probe_result["mlir"]).parent
    vmfb = probe_dir / f"{probe_result['name']}_vulkan_gfx1013.vmfb"
    output = probe_dir / "iree_vulkan_output.npy"

    result: dict[str, Any] = {"vmfb": vmfb.as_posix()}
    if vmfb.exists() and not force_compile:
        result["compile"] = {"status": "exists", "vmfb_size_bytes": vmfb.stat().st_size}
    else:
        compile_cmd = [
            IREE_COMPILE.as_posix(),
            probe_result["mlir"],
            "--iree-hal-target-backends=vulkan-spirv",
            *IREE_FLAGS,
            f"-o={vmfb}",
        ]
        compile_result = run_cmd(compile_cmd, timeout=compile_timeout)
        result["compile"] = compile_result
        if compile_result["returncode"] != 0:
            result["status"] = "compile_failed"
            return result

    if skip_run:
        result["status"] = "compiled"
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
    torch.manual_seed(20260708 + frames + 1000)
    mask = torch.ones(1, 1, frames, dtype=torch.float32)
    attention_bias = torch.zeros(1, 1, frames, dtype=torch.float32)
    time_emb = torch.randn(1, 1024, dtype=torch.float32)
    c256 = torch.randn(1, 256, frames, dtype=torch.float32)
    t256 = torch.randn(1, frames, 256, dtype=torch.float32)

    probes: dict[str, Probe] = {}
    probes["down_resnet_0"] = Probe(
        key="down_resnet_0",
        name=f"s3_distinct_down_resnet0_t{frames}",
        module=estimator.down_blocks[0][0],
        args=(torch.randn(1, 320, frames, dtype=torch.float32), mask, time_emb),
        role="down_resnet",
        index=(0,),
    )
    for block_index, block in enumerate(estimator.down_blocks[0][1]):
        key = f"down_transformer_{block_index}"
        probes[key] = Probe(
            key=key,
            name=f"s3_distinct_down_transformer{block_index}_t{frames}",
            module=TransformerWrapper(block),
            args=(t256, attention_bias, time_emb),
            role="down_transformer",
            index=(0, block_index),
        )
    probes["downsample_0"] = Probe(
        key="downsample_0",
        name=f"s3_distinct_downsample0_t{frames}",
        module=estimator.down_blocks[0][2],
        args=(c256,),
        role="downsample",
        index=(0,),
    )

    for mid_index, (resnet, transformer_blocks) in enumerate(
        (block[0], block[1]) for block in estimator.mid_blocks
    ):
        key = f"mid_resnet_{mid_index}"
        probes[key] = Probe(
            key=key,
            name=f"s3_distinct_mid_resnet{mid_index}_t{frames}",
            module=resnet,
            args=(c256, mask, time_emb),
            role="mid_resnet",
            index=(mid_index,),
        )
        for block_index, block in enumerate(transformer_blocks):
            key = f"mid_transformer_{mid_index}_{block_index}"
            probes[key] = Probe(
                key=key,
                name=f"s3_distinct_mid{mid_index}_transformer{block_index}_t{frames}",
                module=TransformerWrapper(block),
                args=(t256, attention_bias, time_emb),
                role="mid_transformer",
                index=(mid_index, block_index),
            )

    probes["up_resnet_0"] = Probe(
        key="up_resnet_0",
        name=f"s3_distinct_up_resnet0_t{frames}",
        module=estimator.up_blocks[0][0],
        args=(torch.randn(1, 512, frames, dtype=torch.float32), mask, time_emb),
        role="up_resnet",
        index=(0,),
    )
    for block_index, block in enumerate(estimator.up_blocks[0][1]):
        key = f"up_transformer_{block_index}"
        probes[key] = Probe(
            key=key,
            name=f"s3_distinct_up_transformer{block_index}_t{frames}",
            module=TransformerWrapper(block),
            args=(t256, attention_bias, time_emb),
            role="up_transformer",
            index=(0, block_index),
        )
    probes["upsample_0"] = Probe(
        key="upsample_0",
        name=f"s3_distinct_upsample0_t{frames}",
        module=estimator.up_blocks[0][2],
        args=(c256,),
        role="upsample",
        index=(0,),
    )
    probes["final_block"] = Probe(
        key="final_block",
        name=f"s3_distinct_final_block_t{frames}",
        module=estimator.final_block,
        args=(c256, mask),
        role="final_block",
        index=(),
    )
    probes["final_proj"] = Probe(
        key="final_proj",
        name=f"s3_distinct_final_proj_t{frames}",
        module=estimator.final_proj,
        args=(c256,),
        role="final_proj",
        index=(),
    )
    return probes


def expand_selection(selected: list[str], probes: dict[str, Probe]) -> list[str]:
    groups = {
        "down_transformers": [key for key in probes if key.startswith("down_transformer_")],
        "up_transformers": [key for key in probes if key.startswith("up_transformer_")],
        "mid_transformers": [key for key in probes if key.startswith("mid_transformer_")],
        "mid_resnets": [key for key in probes if key.startswith("mid_resnet_")],
        "edges": ["down_resnet_0", "downsample_0", "up_resnet_0", "upsample_0", "final_block", "final_proj"],
    }
    groups["nontransformers"] = groups["mid_resnets"] + groups["edges"]
    groups["all"] = list(probes)

    expanded: list[str] = []
    for item in selected:
        if item in groups:
            expanded.extend(groups[item])
        else:
            expanded.append(item)
    seen = set()
    ordered = []
    for key in expanded:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def attempt_probe(
    probe: Probe,
    compile_timeout: int,
    run_timeout: int,
    force_export: bool,
    force_compile: bool,
    skip_run: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "key": probe.key,
        "name": probe.name,
        "role": probe.role,
        "index": list(probe.index),
    }
    try:
        result.update(export_probe(probe, force_export=force_export))
        result["iree_vulkan"] = compile_and_run(
            result,
            compile_timeout=compile_timeout,
            run_timeout=run_timeout,
            force_compile=force_compile,
            skip_run=skip_run,
        )
        result["status"] = "ok" if result["iree_vulkan"]["status"] in {"ok", "compiled"} else "failed"
    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback_tail"] = traceback.format_exc().splitlines()[-20:]
    result["seconds"] = time.perf_counter() - started
    vulkan = result.get("iree_vulkan", {})
    compare = vulkan.get("compare", {}) if vulkan else {}
    if compare:
        print(
            f"{probe.key}: {vulkan.get('status')} {result['seconds']:.1f}s "
            f"max_abs={compare['max_abs_error']:.3e} "
            f"allclose_1e_4={compare['allclose_1e_4']}"
        )
    else:
        print(f"{probe.key}: {result['status']} {result['seconds']:.1f}s vulkan={vulkan.get('status')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=1210)
    parser.add_argument(
        "--probes",
        default="down_transformers",
        help="Comma-separated probe keys or groups: down_transformers, up_transformers, mid_transformers, mid_resnets, nontransformers, edges, all.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit selected probes after expansion.")
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=180)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "distinct_estimator_components_latest.json",
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
    attention_processors_changed = force_eager_attention(estimator)
    load_seconds = time.perf_counter() - load_started

    probes = build_probes(estimator, frames=args.frames)
    requested = [part.strip() for part in args.probes.split(",") if part.strip()]
    selected = expand_selection(requested, probes)
    unknown = [key for key in selected if key not in probes]
    if unknown:
        raise SystemExit(f"Unknown probes: {unknown}; known groups and probes include {sorted(probes)[:8]}...")
    if args.limit > 0:
        selected = selected[: args.limit]

    results = [
        attempt_probe(
            probes[key],
            compile_timeout=args.compile_timeout,
            run_timeout=args.run_timeout,
            force_export=args.force_export,
            force_compile=args.force_compile,
            skip_run=args.skip_run,
        )
        for key in selected
    ]
    ok = [
        result
        for result in results
        if result.get("iree_vulkan", {}).get("status") == "ok"
        and result.get("iree_vulkan", {}).get("compare", {}).get("allclose_1e_4")
    ]
    report = {
        "description": "Distinct S3 estimator component export/compile/validation on IREE Vulkan.",
        "frames": args.frames,
        "requested": requested,
        "selected": selected,
        "load_seconds": load_seconds,
        "attention_processors_changed": attention_processors_changed,
        "iree_flags": list(IREE_FLAGS),
        "out_dir": OUT_DIR.as_posix(),
        "summary": {
            "selected_count": len(selected),
            "ok_allclose_1e_4_count": len(ok),
            "failed_count": len(results) - len(ok),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")
    print(f"ok_allclose_1e_4={len(ok)}/{len(selected)}")


if __name__ == "__main__":
    main()
