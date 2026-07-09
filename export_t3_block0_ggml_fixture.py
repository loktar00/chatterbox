#!/usr/bin/env python3
"""Export a real Chatterbox T3 GPT-2 block-0 fixture for ggml/Vulkan work."""

from __future__ import annotations

import json
import math
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent

def write_ggml_f32(path: Path, array: np.ndarray) -> dict:
    arr = np.asarray(array, dtype=np.float32)
    arr_f = np.asfortranarray(arr)
    arr_f.ravel(order="F").tofile(path)
    return {
        "path": str(path),
        "shape": list(arr.shape),
        "dtype": "float32",
        "layout": "ggml/ne0-fastest/fortran-order",
        "bytes": int(arr_f.size * 4),
    }


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def manual_cached_block0(block, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
    ln1 = F.layer_norm(x, (x.shape[-1],), block.ln_1.weight, block.ln_1.bias, block.ln_1.eps)
    qkv = torch.addmm(block.attn.c_attn.bias, ln1.view(-1, x.shape[-1]), block.attn.c_attn.weight).view(1, 1, -1)
    q, k_new, v_new = qkv.split(x.shape[-1], dim=2)

    n_head = block.attn.num_heads
    head_dim = block.attn.head_dim
    q = q.view(1, 1, n_head, head_dim).transpose(1, 2)
    k_new = k_new.view(1, 1, n_head, head_dim).transpose(1, 2)
    v_new = v_new.view(1, 1, n_head, head_dim).transpose(1, 2)

    k_all = torch.cat([past_k, k_new], dim=2)
    v_all = torch.cat([past_v, v_new], dim=2)
    attn_scores = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(float(head_dim))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_probs, v_all).transpose(1, 2).reshape(1, 1, -1)
    attn_proj = torch.addmm(
        block.attn.c_proj.bias,
        attn_out.view(-1, x.shape[-1]),
        block.attn.c_proj.weight,
    ).view(1, 1, -1)
    resid1 = x + attn_proj

    ln2 = F.layer_norm(resid1, (x.shape[-1],), block.ln_2.weight, block.ln_2.bias, block.ln_2.eps)
    fc = torch.addmm(block.mlp.c_fc.bias, ln2.view(-1, x.shape[-1]), block.mlp.c_fc.weight).view(1, 1, -1)
    act = new_gelu(fc)
    mlp = torch.addmm(block.mlp.c_proj.bias, act.view(-1, act.shape[-1]), block.mlp.c_proj.weight).view(1, 1, -1)
    out = resid1 + mlp
    return {
        "ln1": ln1,
        "q": q,
        "k_new": k_new,
        "v_new": v_new,
        "k_all": k_all,
        "v_all": v_all,
        "attn_probs": attn_probs,
        "attn_out": attn_out,
        "resid1": resid1,
        "ln2": ln2,
        "fc": fc,
        "out": out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--past-len", type=int, default=16)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to exports/ggml_t3_block0_fixture for past_len=16, otherwise exports/ggml_t3_block0_fixture_pN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260708)

    out_dir = args.out_dir
    if out_dir is None:
        suffix = "" if args.past_len == 16 else f"_p{args.past_len}"
        out_dir = ROOT / "exports" / f"ggml_t3_block0_fixture{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    block = t3.tfmr.h[0]

    hidden = torch.randn(1, 1, t3.tfmr.config.n_embd, dtype=torch.float32) * 0.05
    past_len = args.past_len
    n_head = t3.tfmr.config.n_head
    head_dim = t3.tfmr.config.n_embd // n_head
    past_k = torch.randn(1, n_head, past_len, head_dim, dtype=torch.float32) * 0.04
    past_v = torch.randn(1, n_head, past_len, head_dim, dtype=torch.float32) * 0.04

    with torch.inference_mode():
        ref = manual_cached_block0(block, hidden, past_k, past_v)

    tensors: dict[str, dict] = {}

    weights = {
        "ln1_weight": block.ln_1.weight.detach().numpy(),
        "ln1_bias": block.ln_1.bias.detach().numpy(),
        "attn_c_attn_weight": block.attn.c_attn.weight.detach().numpy(),
        "attn_c_attn_bias": block.attn.c_attn.bias.detach().numpy(),
        "attn_c_proj_weight": block.attn.c_proj.weight.detach().numpy(),
        "attn_c_proj_bias": block.attn.c_proj.bias.detach().numpy(),
        "ln2_weight": block.ln_2.weight.detach().numpy(),
        "ln2_bias": block.ln_2.bias.detach().numpy(),
        "mlp_c_fc_weight": block.mlp.c_fc.weight.detach().numpy(),
        "mlp_c_fc_bias": block.mlp.c_fc.bias.detach().numpy(),
        "mlp_c_proj_weight": block.mlp.c_proj.weight.detach().numpy(),
        "mlp_c_proj_bias": block.mlp.c_proj.bias.detach().numpy(),
    }
    for name, array in weights.items():
        tensors[name] = write_ggml_f32(out_dir / f"{name}.f32", array)

    tensors["input_hidden"] = write_ggml_f32(out_dir / "input_hidden.f32", hidden[0, 0].numpy())

    # ggml attention layout: K [head_dim, seq, heads], V [seq, head_dim, heads].
    past_k_ggml = past_k[0].permute(2, 1, 0).numpy()
    past_v_ggml = past_v[0].permute(1, 2, 0).numpy()
    tensors["past_k"] = write_ggml_f32(out_dir / "past_k.f32", past_k_ggml)
    tensors["past_v"] = write_ggml_f32(out_dir / "past_v.f32", past_v_ggml)

    references = {
        "output_hidden": ref["out"][0, 0].detach().numpy(),
        "resid1": ref["resid1"][0, 0].detach().numpy(),
        "ln1": ref["ln1"][0, 0].detach().numpy(),
        "ln2": ref["ln2"][0, 0].detach().numpy(),
        "attn_out": ref["attn_out"][0, 0].detach().numpy(),
    }
    for name, array in references.items():
        tensors[f"ref_{name}"] = write_ggml_f32(out_dir / f"ref_{name}.f32", array)

    metadata = {
        "description": "Real Chatterbox T3 GPT-2 block-0 cached one-token fixture for ggml/Vulkan correctness work.",
        "hidden_size": int(t3.tfmr.config.n_embd),
        "n_layer": int(t3.tfmr.config.n_layer),
        "n_head": int(n_head),
        "head_dim": int(head_dim),
        "past_len": int(past_len),
        "seq_len_after_update": int(past_len + 1),
        "layer_norm_eps": float(t3.tfmr.config.layer_norm_epsilon),
        "activation": str(t3.tfmr.config.activation_function),
        "attention_scale": float(1.0 / math.sqrt(float(head_dim))),
        "tensors": tensors,
    }
    meta_path = out_dir / "manifest.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(meta_path)


if __name__ == "__main__":
    main()
