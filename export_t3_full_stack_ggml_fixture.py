#!/usr/bin/env python3
"""Export a full real Chatterbox T3 GPT-2 cached-stack fixture for ggml/Vulkan."""

from __future__ import annotations

import argparse
import json
import math
import os
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


def manual_cached_block(block, x: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor):
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
    return resid1 + mlp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--past-len", type=int, default=935)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to exports/ggml_t3_full_stack_fixture_pN.",
    )
    return parser.parse_args()


def export_layer(out_dir: Path, layer_idx: int, block, past_k: torch.Tensor, past_v: torch.Tensor, tensors: dict[str, dict]) -> None:
    prefix = f"layer_{layer_idx:02d}"
    arrays = {
        f"{prefix}_ln1_weight": block.ln_1.weight.detach().numpy(),
        f"{prefix}_ln1_bias": block.ln_1.bias.detach().numpy(),
        f"{prefix}_attn_c_attn_weight": block.attn.c_attn.weight.detach().numpy(),
        f"{prefix}_attn_c_attn_bias": block.attn.c_attn.bias.detach().numpy(),
        f"{prefix}_attn_c_proj_weight": block.attn.c_proj.weight.detach().numpy(),
        f"{prefix}_attn_c_proj_bias": block.attn.c_proj.bias.detach().numpy(),
        f"{prefix}_ln2_weight": block.ln_2.weight.detach().numpy(),
        f"{prefix}_ln2_bias": block.ln_2.bias.detach().numpy(),
        f"{prefix}_mlp_c_fc_weight": block.mlp.c_fc.weight.detach().numpy(),
        f"{prefix}_mlp_c_fc_bias": block.mlp.c_fc.bias.detach().numpy(),
        f"{prefix}_mlp_c_proj_weight": block.mlp.c_proj.weight.detach().numpy(),
        f"{prefix}_mlp_c_proj_bias": block.mlp.c_proj.bias.detach().numpy(),
        # ggml attention layout: K [head_dim, seq, heads], V [seq, head_dim, heads].
        f"{prefix}_past_k": past_k[0].permute(2, 1, 0).numpy(),
        f"{prefix}_past_v": past_v[0].permute(1, 2, 0).numpy(),
    }
    for name, array in arrays.items():
        tensors[name] = write_ggml_f32(out_dir / f"{name}.f32", array)


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
        out_dir = ROOT / "exports" / f"ggml_t3_full_stack_fixture_p{args.past_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"

    hidden_size = t3.tfmr.config.n_embd
    n_layer = t3.tfmr.config.n_layer
    n_head = t3.tfmr.config.n_head
    head_dim = hidden_size // n_head
    past_len = args.past_len

    hidden = torch.randn(1, 1, hidden_size, dtype=torch.float32) * 0.05
    past_pairs = []
    for _ in range(n_layer):
        past_k = torch.randn(1, n_head, past_len, head_dim, dtype=torch.float32) * 0.04
        past_v = torch.randn(1, n_head, past_len, head_dim, dtype=torch.float32) * 0.04
        past_pairs.append((past_k, past_v))

    tensors: dict[str, dict] = {}
    tensors["input_hidden"] = write_ggml_f32(out_dir / "input_hidden.f32", hidden[0, 0].numpy())

    with torch.inference_mode():
        cur = hidden
        for layer_idx, (block, (past_k, past_v)) in enumerate(zip(t3.tfmr.h, past_pairs)):
            export_layer(out_dir, layer_idx, block, past_k, past_v, tensors)
            cur = manual_cached_block(block, cur, past_k, past_v)

        final_hidden = F.layer_norm(cur, (hidden_size,), t3.tfmr.ln_f.weight, t3.tfmr.ln_f.bias, t3.tfmr.ln_f.eps)
        logits = t3.speech_head(final_hidden)

    tensors["ln_f_weight"] = write_ggml_f32(out_dir / "ln_f_weight.f32", t3.tfmr.ln_f.weight.detach().numpy())
    tensors["ln_f_bias"] = write_ggml_f32(out_dir / "ln_f_bias.f32", t3.tfmr.ln_f.bias.detach().numpy())
    # nn.Linear stores [out, in]; ggml matmul expects [in, out].
    tensors["speech_head_weight_t"] = write_ggml_f32(out_dir / "speech_head_weight_t.f32", t3.speech_head.weight.detach().numpy().T)
    tensors["speech_head_bias"] = write_ggml_f32(out_dir / "speech_head_bias.f32", t3.speech_head.bias.detach().numpy())
    tensors["ref_hidden_pre_ln_f"] = write_ggml_f32(out_dir / "ref_hidden_pre_ln_f.f32", cur[0, 0].detach().numpy())
    tensors["ref_final_hidden"] = write_ggml_f32(out_dir / "ref_final_hidden.f32", final_hidden[0, 0].detach().numpy())
    tensors["ref_logits"] = write_ggml_f32(out_dir / "ref_logits.f32", logits[0, 0].detach().numpy())

    logits_np = logits[0, 0].detach().numpy()
    top10 = np.argsort(logits_np)[-10:][::-1].astype(int).tolist()
    metadata = {
        "description": "Full real Chatterbox T3 GPT-2 cached one-token stack fixture for ggml/Vulkan correctness work.",
        "hidden_size": int(hidden_size),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "head_dim": int(head_dim),
        "past_len": int(past_len),
        "seq_len_after_update": int(past_len + 1),
        "speech_vocab": int(t3.hp.speech_tokens_dict_size),
        "layer_norm_eps": float(t3.tfmr.config.layer_norm_epsilon),
        "activation": str(t3.tfmr.config.activation_function),
        "attention_scale": float(1.0 / math.sqrt(float(head_dim))),
        "ref_argmax": int(np.argmax(logits_np)),
        "ref_top10": top10,
        "tensors": tensors,
    }
    meta_path = out_dir / "manifest.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(meta_path)


if __name__ == "__main__":
    main()
