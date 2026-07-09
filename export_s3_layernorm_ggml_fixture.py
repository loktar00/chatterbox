#!/usr/bin/env python3
"""Export the S3 flow LayerNorm repro tensors for a ggml/Vulkan probe."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "exports" / "s3_flow_vulkan_components" / "s3_flow_mid_norm1_t16"
OUT_DIR = ROOT / "exports" / "ggml_s3_layernorm_mid_t16"


def write_f32(path: Path, array: np.ndarray) -> dict:
    arr = np.asarray(array, dtype=np.float32)
    arr.tofile(path)
    return {
        "path": path.as_posix(),
        "shape": list(arr.shape),
        "dtype": "float32",
        "layout": "c-order",
        "bytes": int(arr.size * 4),
    }


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_arr = np.load(SRC_DIR / "input_0.npy").astype(np.float32)
    ref_arr = np.load(SRC_DIR / "torch_output.npy").astype(np.float32)
    iree_vulkan_arr = np.load(SRC_DIR / "iree_vulkan_output.npy").astype(np.float32)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    norm = model.s3gen.flow.decoder.estimator.mid_blocks[0][1][0].norm1
    weight = norm.weight.detach().cpu().numpy().astype(np.float32)
    bias = norm.bias.detach().cpu().numpy().astype(np.float32)

    # Drop the batch dimension. ggml tensors use ne0 as the fastest dimension,
    # so shape [T, H] in C-order maps to ggml [H, T].
    tensors = {
        "input": write_f32(OUT_DIR / "input_t_h.f32", input_arr[0]),
        "reference": write_f32(OUT_DIR / "reference_t_h.f32", ref_arr[0]),
        "iree_vulkan_reference": write_f32(OUT_DIR / "iree_vulkan_t_h.f32", iree_vulkan_arr[0]),
        "weight": write_f32(OUT_DIR / "weight_h.f32", weight),
        "bias": write_f32(OUT_DIR / "bias_h.f32", bias),
    }
    manifest = {
        "description": "S3 flow mid transformer norm1 repro exported for ggml/Vulkan.",
        "source_dir": SRC_DIR.as_posix(),
        "hidden": int(input_arr.shape[-1]),
        "tokens": int(input_arr.shape[-2]),
        "eps": float(norm.eps),
        "tensors": tensors,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote={OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
