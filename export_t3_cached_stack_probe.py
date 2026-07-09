#!/usr/bin/env python3
"""Export a small stack of cached GPT-2 blocks for T3 Vulkan timing."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from iree.turbine import aot
from torch import nn

from chatterbox.tts_turbo import ChatterboxTurboTTS
from export_t3_cached_block_probe import CachedManualBlockHiddenWrapper


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"


class CachedManualStackHiddenWrapper(nn.Module):
    def __init__(self, blocks: list[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CachedManualBlockHiddenWrapper(block) for block in blocks])

    def forward(self, hidden_states: torch.Tensor, *past_tensors: torch.Tensor) -> torch.Tensor:
        for index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                past_tensors[index * 2],
                past_tensors[index * 2 + 1],
            )
        return hidden_states


def export_probe(name: str, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    module.eval()
    with torch.inference_mode():
        expected = module(*inputs).detach().cpu().numpy()

    input_paths = []
    for index, tensor in enumerate(inputs):
        path = OUT_DIR / f"{name}_input_{index}.npy"
        np.save(path, tensor.detach().cpu().numpy())
        input_paths.append(path)

    expected_path = OUT_DIR / f"{name}_torch_output.npy"
    mlir_path = OUT_DIR / f"{name}.mlir"
    graph_path = OUT_DIR / f"{name}.torch_export.txt"
    np.save(expected_path, expected)

    exported = aot.export(module, args=inputs, module_name=name, function_name="forward")
    exported.save_mlir(mlir_path)
    graph_path.write_text(str(torch.export.export(module, inputs).graph_module) + "\n")

    return {
        "name": name,
        "inputs": [path.as_posix() for path in input_paths],
        "expected": expected_path.as_posix(),
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "output_shape": list(expected.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument(
        "--start-layers",
        default="",
        help="Comma-separated start layers to export in one model load. Overrides --start-layer.",
    )
    parser.add_argument("--past-len", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_cached_stack_export_latest.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    if args.start_layers:
        start_layers = [int(part) for part in args.start_layers.split(",") if part.strip()]
    else:
        start_layers = [args.start_layer]

    probes = []
    for start_layer in start_layers:
        end_layer = start_layer + args.layers
        blocks = [t3.tfmr.h[index].eval() for index in range(start_layer, end_layer)]
        wrapper = CachedManualStackHiddenWrapper(blocks).eval()
        first = wrapper.blocks[0]
        inputs: list[torch.Tensor] = [torch.randn(1, 1, t3.dim, dtype=torch.float32)]
        for _ in range(args.layers):
            inputs.append(torch.randn(1, first.num_heads, args.past_len, first.head_dim, dtype=torch.float32))
            inputs.append(torch.randn(1, first.num_heads, args.past_len, first.head_dim, dtype=torch.float32))

        name = f"gpt2_cached_stack_s{start_layer}_l{args.layers}_p{args.past_len}_t1"
        probes.append(
            {
                "start_layer": start_layer,
                "end_layer": end_layer,
                "probe": export_probe(name, wrapper, tuple(inputs)),
            }
        )

    report: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "layers": args.layers,
        "start_layers": start_layers,
        "past_len": args.past_len,
        "probes": probes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for item in probes:
        probe = item["probe"]
        print(f"{probe['name']}: {probe['output_shape']}")


if __name__ == "__main__":
    main()
