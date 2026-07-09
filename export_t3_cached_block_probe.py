#!/usr/bin/env python3
"""Export a static-tensor cached GPT-2 block probe for T3 Vulkan testing."""

from __future__ import annotations

import argparse
import json
import math
import os
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


class CachedManualBlockHiddenWrapper(nn.Module):
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
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)

        query, key, value = self.block.attn.c_attn(hidden_states).split(self.split_size, dim=2)
        query_shape = (*query.shape[:-1], -1, self.head_dim)
        query = query.view(query_shape).transpose(1, 2)
        key_shape = (*key.shape[:-1], -1, self.head_dim)
        key = key.view(key_shape).transpose(1, 2)
        value = value.view(key_shape).transpose(1, 2)

        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)

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
        return residual + hidden_states


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
    parser.add_argument("--past-lens", default="16,128,384")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_cached_block_exports_latest.json",
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
    block = t3.tfmr.h[args.block_index].eval()
    wrapper = CachedManualBlockHiddenWrapper(block).eval()

    probes = []
    for past_len_text in args.past_lens.split(","):
        past_len = int(past_len_text.strip())
        hidden = torch.randn(1, 1, t3.dim, dtype=torch.float32)
        past_key = torch.randn(1, wrapper.num_heads, past_len, wrapper.head_dim, dtype=torch.float32)
        past_value = torch.randn(1, wrapper.num_heads, past_len, wrapper.head_dim, dtype=torch.float32)
        name = f"gpt2_cached_block{args.block_index}_hidden_p{past_len}_t1"
        probes.append(export_probe(name, wrapper, (hidden, past_key, past_value)))

    report: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "block_index": args.block_index,
        "probes": probes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for probe in probes:
        print(f"{probe['name']}: {probe['output_shape']}")


if __name__ == "__main__":
    main()
