#!/usr/bin/env python3
"""Export smaller T3 block component graphs for IREE Vulkan debugging."""

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


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability"


class LayerNormWrapper(nn.Module):
    def __init__(self, layer_norm: nn.Module) -> None:
        super().__init__()
        self.layer_norm = layer_norm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(hidden_states)


class LnAttentionResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)
        attention_outputs = self.block.attn(
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            output_attentions=False,
        )
        return residual + attention_outputs[0]


class LnMlpResidualWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.block.ln_2(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        return residual + hidden_states


class ManualBlockWrapper(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.block.ln_1(hidden_states)
        attention_outputs = self.block.attn(
            hidden_states,
            past_key_values=None,
            attention_mask=None,
            output_attentions=False,
        )
        hidden_states = residual + attention_outputs[0]

        residual = hidden_states
        hidden_states = self.block.ln_2(hidden_states)
        hidden_states = self.block.mlp(hidden_states)
        return residual + hidden_states


def export_probe(name: str, module: nn.Module, sample: torch.Tensor) -> dict[str, Any]:
    module.eval()
    with torch.inference_mode():
        expected = module(sample).detach().cpu().numpy()

    input_path = OUT_DIR / f"{name}_input_0.npy"
    expected_path = OUT_DIR / f"{name}_torch_output.npy"
    mlir_path = OUT_DIR / f"{name}.mlir"
    graph_path = OUT_DIR / f"{name}.torch_export.txt"

    np.save(input_path, sample.detach().cpu().numpy())
    np.save(expected_path, expected)

    exported = aot.export(module, args=(sample,), module_name=name, function_name="forward")
    exported.save_mlir(mlir_path)
    graph_path.write_text(str(torch.export.export(module, (sample,)).graph_module) + "\n")

    return {
        "name": name,
        "input": input_path.as_posix(),
        "expected": expected_path.as_posix(),
        "mlir": mlir_path.as_posix(),
        "graph": graph_path.as_posix(),
        "output_shape": list(expected.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument(
        "--input",
        type=Path,
        default=OUT_DIR / "gpt2_block_t1_input_0.npy",
        help="Saved block input to reuse for component probes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_block_component_exports_latest.json",
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
    block = t3.tfmr.h[0].eval()

    hidden = torch.from_numpy(np.load(args.input)).to(dtype=torch.float32).clone()
    with torch.inference_mode():
        attention_residual_array = (
            LnAttentionResidualWrapper(block).eval()(hidden).detach().cpu().numpy().copy()
        )
    attention_residual = torch.from_numpy(attention_residual_array).to(dtype=torch.float32)

    probes = [
        export_probe("gpt2_ln1_t1", LayerNormWrapper(block.ln_1), hidden),
        export_probe("gpt2_ln2_t1", LayerNormWrapper(block.ln_2), attention_residual),
        export_probe("gpt2_ln1_attn_resid_t1", LnAttentionResidualWrapper(block), hidden),
        export_probe("gpt2_ln2_mlp_resid_t1", LnMlpResidualWrapper(block), attention_residual),
        export_probe("gpt2_block_manual_t1", ManualBlockWrapper(block), hidden),
    ]

    report: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "source_input": args.input.as_posix(),
        "probes": probes,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for probe in probes:
        print(f"{probe['name']}: {probe['output_shape']}")


if __name__ == "__main__":
    main()
