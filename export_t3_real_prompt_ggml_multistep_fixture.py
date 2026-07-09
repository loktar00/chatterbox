#!/usr/bin/env python3
"""Export a real-prompt multi-step Chatterbox T3 fixture for ggml/Vulkan."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


ROOT = Path(__file__).resolve().parent


def selected_text(name: str) -> str:
    cases = {
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }
    if name not in cases:
        raise ValueError(f"Unknown case {name!r}; available={sorted(cases)}")
    return cases[name]


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


def legacy_past(past_key_values: Any):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def pad_key_for_ggml(key: torch.Tensor, max_len: int) -> np.ndarray:
    # HF key: [1, heads, valid_len, head_dim].
    k = key[0].detach().cpu()
    heads, valid_len, head_dim = k.shape
    out = torch.zeros(head_dim, max_len, heads, dtype=torch.float32)
    out[:, :valid_len, :] = k.permute(2, 1, 0)
    return out.numpy()


def pad_value_for_ggml(value: torch.Tensor, max_len: int) -> np.ndarray:
    # HF value: [1, heads, valid_len, head_dim].
    v = value[0].detach().cpu()
    heads, valid_len, head_dim = v.shape
    out = torch.zeros(max_len, head_dim, heads, dtype=torch.float32)
    out[:valid_len, :, :] = v.permute(1, 2, 0)
    return out.numpy()


def mask_for_step(valid_len: int, max_len: int) -> np.ndarray:
    # k_all is padded past cache [0:max_len) plus the new token at index max_len.
    # Valid positions are real past tokens and the new token. Padded slots are masked out.
    mask = np.full((max_len + 1, 1, 1), -1.0e9, dtype=np.float32)
    mask[:valid_len, :, :] = 0.0
    mask[max_len, :, :] = 0.0
    return mask


def token_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def export_weights(out_dir: Path, t3, tensors: dict[str, dict]) -> None:
    for layer_idx, block in enumerate(t3.tfmr.h):
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
        }
        for name, array in arrays.items():
            tensors[name] = write_ggml_f32(out_dir / f"{name}.f32", array)

    tensors["ln_f_weight"] = write_ggml_f32(out_dir / "ln_f_weight.f32", t3.tfmr.ln_f.weight.detach().numpy())
    tensors["ln_f_bias"] = write_ggml_f32(out_dir / "ln_f_bias.f32", t3.tfmr.ln_f.bias.detach().numpy())
    tensors["speech_head_weight_t"] = write_ggml_f32(
        out_dir / "speech_head_weight_t.f32",
        t3.speech_head.weight.detach().numpy().T,
    )
    tensors["speech_head_bias"] = write_ggml_f32(out_dir / "speech_head_bias.f32", t3.speech_head.bias.detach().numpy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="hello", choices=("hello", "short", "chunk270"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=935)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(args.seed)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "exports" / f"ggml_t3_real_prompt_multistep_{args.case}_s{args.steps}_p{args.max_len}"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = selected_text(args.case)
    normalized = punc_norm(text)

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    hidden_size = t3.tfmr.config.n_embd
    n_head = t3.tfmr.config.n_head
    head_dim = hidden_size // n_head
    n_layer = t3.tfmr.config.n_layer
    speech_vocab = t3.hp.speech_tokens_dict_size

    text_tokens = model.tokenizer(normalized, return_tensors="pt", padding=True, truncation=True)
    text_ids = text_tokens.input_ids.to(dtype=torch.long, device=t3.device)
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_ids[:, :1])

    tensors: dict[str, dict] = {}
    export_weights(out_dir, t3, tensors)

    with torch.inference_mode():
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_ids,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )
        prefill = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        initial_logits = t3.speech_head(prefill[0][:, -1:])
        current_token = token_from_logits(initial_logits)
        past = prefill.past_key_values
        initial_context_len = int(embeds.shape[1])

        if initial_context_len + args.steps >= args.max_len:
            raise SystemExit(
                f"initial context {initial_context_len} + steps {args.steps} exceeds max_len {args.max_len}"
            )

        step_meta = []
        for step in range(args.steps):
            valid_len = initial_context_len + step
            current_embed = t3.speech_emb(current_token)
            position_ids = torch.full(
                (current_embed.shape[0], current_embed.shape[1]),
                valid_len,
                dtype=torch.long,
                device=current_embed.device,
            )
            input_hidden = current_embed + t3.tfmr.wpe(position_ids)

            step_prefix = f"step_{step:02d}"
            step_tensors: dict[str, dict] = {
                "input_hidden": write_ggml_f32(out_dir / f"{step_prefix}_input_hidden.f32", input_hidden[0, 0].detach().numpy()),
                "attn_mask": write_ggml_f32(out_dir / f"{step_prefix}_attn_mask.f32", mask_for_step(valid_len, args.max_len)),
            }

            for layer_idx, layer_cache in enumerate(legacy_past(past)):
                key, value = layer_cache[0], layer_cache[1]
                layer_prefix = f"{step_prefix}_layer_{layer_idx:02d}"
                step_tensors[f"layer_{layer_idx:02d}_past_k"] = write_ggml_f32(
                    out_dir / f"{layer_prefix}_past_k.f32",
                    pad_key_for_ggml(key, args.max_len),
                )
                step_tensors[f"layer_{layer_idx:02d}_past_v"] = write_ggml_f32(
                    out_dir / f"{layer_prefix}_past_v.f32",
                    pad_value_for_ggml(value, args.max_len),
                )

            cpu_out = t3.tfmr(
                inputs_embeds=current_embed,
                past_key_values=past,
                use_cache=True,
            )
            cpu_logits = t3.speech_head(cpu_out[0])
            final_hidden = t3.tfmr.ln_f(cpu_out[0])
            next_token = token_from_logits(cpu_logits)
            logits_np = cpu_logits[0, 0].detach().numpy()
            step_tensors["ref_hidden_pre_ln_f"] = write_ggml_f32(
                out_dir / f"{step_prefix}_ref_hidden_pre_ln_f.f32",
                cpu_out[0][0, 0].detach().numpy(),
            )
            step_tensors["ref_final_hidden"] = write_ggml_f32(
                out_dir / f"{step_prefix}_ref_final_hidden.f32",
                final_hidden[0, 0].detach().numpy(),
            )
            step_tensors["ref_logits"] = write_ggml_f32(
                out_dir / f"{step_prefix}_ref_logits.f32",
                logits_np,
            )

            top10 = np.argsort(logits_np)[-10:][::-1].astype(int).tolist()
            step_meta.append(
                {
                    "step": step,
                    "valid_len_before_step": valid_len,
                    "input_token": int(current_token[0, 0].detach().item()),
                    "next_argmax": int(next_token[0, 0].detach().item()),
                    "top10": top10,
                    "tensors": step_tensors,
                }
            )
            past = cpu_out.past_key_values
            current_token = next_token

    metadata = {
        "description": "Real prompt multi-step Chatterbox T3 cached stack fixture for ggml/Vulkan.",
        "case": args.case,
        "text_chars": len(text),
        "normalized_chars": len(normalized),
        "text_tokens": int(text_ids.shape[1]),
        "conditioning_tokens": int(len_cond),
        "initial_context_len": int(initial_context_len),
        "steps": int(args.steps),
        "max_len": int(args.max_len),
        "hidden_size": int(hidden_size),
        "n_layer": int(n_layer),
        "n_head": int(n_head),
        "head_dim": int(head_dim),
        "speech_vocab": int(speech_vocab),
        "layer_norm_eps": float(t3.tfmr.config.layer_norm_epsilon),
        "activation": str(t3.tfmr.config.activation_function),
        "attention_scale": float(1.0 / math.sqrt(float(head_dim))),
        "static_tensors": tensors,
        "steps_meta": step_meta,
    }
    meta_path = out_dir / "manifest.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(meta_path)


if __name__ == "__main__":
    main()
