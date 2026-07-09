#!/usr/bin/env python3
"""CPU-only ONNX export probes for Chatterbox subgraphs."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Callable

import onnx
import torch
from torch import nn
from torch.nn.utils import parametrize

from chatterbox.models.s3gen.f0_predictor import ConvRNNF0Predictor
from chatterbox.models.s3gen.hifigan import ResBlock, Snake
from chatterbox.models.s3gen.transformer.positionwise_feed_forward import PositionwiseFeedForward
from chatterbox.models.voice_encoder.config import VoiceEncConfig
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder


OUT_DIR = Path(__file__).resolve().parent / "exports" / "subgraph_probes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class Probe:
    def __init__(
        self,
        name: str,
        build_model: Callable[[], nn.Module],
        sample_inputs: tuple[torch.Tensor, ...],
        input_names: list[str],
        output_names: list[str],
    ) -> None:
        self.name = name
        self.build_model = build_model
        self.sample_inputs = sample_inputs
        self.input_names = input_names
        self.output_names = output_names


def bake_parametrized_weights(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
    return model


def export_probe(probe: Probe) -> dict:
    output_path = OUT_DIR / f"{probe.name}.onnx"
    try:
        torch.manual_seed(0)
        model = probe.build_model().cpu().eval()
        with torch.no_grad():
            eager_output = model(*probe.sample_inputs)
            if torch.is_tensor(eager_output):
                eager_shapes = [tuple(eager_output.shape)]
            else:
                eager_shapes = [
                    tuple(item.shape) if torch.is_tensor(item) else str(type(item))
                    for item in eager_output
                ]

        torch.onnx.export(
            model,
            probe.sample_inputs,
            output_path.as_posix(),
            input_names=probe.input_names,
            output_names=probe.output_names,
            opset_version=17,
            do_constant_folding=True,
        )

        exported = onnx.load(output_path.as_posix())
        onnx.checker.check_model(exported)
        inferred = onnx.shape_inference.infer_shapes(exported)
        onnx.checker.check_model(inferred)
        op_counts = Counter(node.op_type for node in exported.graph.node)

        return {
            "name": probe.name,
            "status": "ok",
            "onnx_path": output_path.as_posix(),
            "eager_output_shapes": eager_shapes,
            "node_count": len(exported.graph.node),
            "initializer_count": len(exported.graph.initializer),
            "op_counts": dict(sorted(op_counts.items())),
        }
    except Exception as exc:
        return {
            "name": probe.name,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }


def main() -> None:
    hp = VoiceEncConfig()
    probes = [
        Probe(
            "voice_encoder_forward",
            lambda: VoiceEncoder(hp),
            (torch.rand(1, hp.ve_partial_frames, hp.num_mels, dtype=torch.float32),),
            ["mels"],
            ["speaker_embedding"],
        ),
        Probe(
            "f0_predictor_parametrized",
            lambda: ConvRNNF0Predictor(),
            (torch.rand(1, 80, 64, dtype=torch.float32),),
            ["mel"],
            ["f0"],
        ),
        Probe(
            "f0_predictor_baked_weightnorm",
            lambda: bake_parametrized_weights(ConvRNNF0Predictor()),
            (torch.rand(1, 80, 64, dtype=torch.float32),),
            ["mel"],
            ["f0"],
        ),
        Probe(
            "hifigan_snake",
            lambda: Snake(64),
            (torch.rand(1, 64, 64, dtype=torch.float32),),
            ["x"],
            ["y"],
        ),
        Probe(
            "hifigan_resblock_parametrized",
            lambda: ResBlock(channels=64, kernel_size=3, dilations=[1, 3, 5]),
            (torch.rand(1, 64, 64, dtype=torch.float32),),
            ["x"],
            ["y"],
        ),
        Probe(
            "hifigan_resblock_baked_weightnorm",
            lambda: bake_parametrized_weights(ResBlock(channels=64, kernel_size=3, dilations=[1, 3, 5])),
            (torch.rand(1, 64, 64, dtype=torch.float32),),
            ["x"],
            ["y"],
        ),
        Probe(
            "s3gen_positionwise_ffn",
            lambda: PositionwiseFeedForward(idim=128, hidden_units=512, dropout_rate=0.0, activation=nn.SiLU()),
            (torch.rand(1, 64, 128, dtype=torch.float32),),
            ["x"],
            ["y"],
        ),
    ]

    results = [export_probe(probe) for probe in probes]
    output_path = OUT_DIR / "export_probe_results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        if result["status"] == "ok":
            print(
                f"{result['name']}: ok nodes={result['node_count']} "
                f"ops={','.join(result['op_counts'])}"
            )
        else:
            print(f"{result['name']}: failed {result['error_type']}: {result['error']}")
    print(f"results={output_path}")


if __name__ == "__main__":
    main()
