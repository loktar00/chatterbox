#!/usr/bin/env python3
"""CPU-only ONNX export and runtime validation probe for the voice encoder."""

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from chatterbox.models.voice_encoder.config import VoiceEncConfig
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder


def main() -> None:
    hp = VoiceEncConfig()
    output_dir = Path(__file__).resolve().parent / "exports"
    bench_dir = output_dir / "benchmarks"
    output_dir.mkdir(exist_ok=True)
    bench_dir.mkdir(exist_ok=True)
    output_path = output_dir / "voice_encoder.forward.random_weights.onnx"
    report_path = bench_dir / "voice_encoder_onnxruntime_random_weights_2026-07-08.json"

    torch.set_num_threads(1)
    torch.manual_seed(0)

    model = VoiceEncoder(hp).cpu().eval()
    sample_mels = torch.rand(1, hp.ve_partial_frames, hp.num_mels, dtype=torch.float32)

    with torch.no_grad():
        eager_output = model(sample_mels)

    torch.onnx.export(
        model,
        sample_mels,
        output_path.as_posix(),
        input_names=["mels"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mels": {0: "batch"},
            "speaker_embedding": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    exported = onnx.load(output_path.as_posix())
    onnx.checker.check_model(exported)
    inferred = onnx.shape_inference.infer_shapes(exported)
    onnx.checker.check_model(inferred)

    session = ort.InferenceSession(output_path.as_posix(), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"mels": sample_mels.detach().cpu().numpy()})[0]
    eager_np = eager_output.detach().cpu().numpy()
    diff = np.abs(ort_output - eager_np)
    report = {
        "description": "CPU-only random-weight voice encoder ONNX export and ONNX Runtime validation.",
        "onnx_path": output_path.as_posix(),
        "input_shape": list(sample_mels.shape),
        "torch_output_shape": list(eager_output.shape),
        "ort_output_shape": list(ort_output.shape),
        "providers": session.get_providers(),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "allclose_1e_5": bool(np.allclose(ort_output, eager_np, rtol=1e-5, atol=1e-5)),
        "allclose_1e_4": bool(np.allclose(ort_output, eager_np, rtol=1e-4, atol=1e-4)),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"exported={output_path}")
    print(f"json={report_path}")
    print(f"input_shape={tuple(sample_mels.shape)}")
    print(f"output_shape={tuple(eager_output.shape)}")
    print("onnx_check=ok")
    print(f"onnxruntime_allclose_1e_5={report['allclose_1e_5']}")


if __name__ == "__main__":
    main()
