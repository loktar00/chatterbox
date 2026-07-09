#!/usr/bin/env python3
"""Validate p1024 Vulkan T3 follow-on steps after a real CPU prompt prefill."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_t3_full_masked_vulkan_loop import VulkanFullMaskedT3Loop, padded_reference_cache
from benchmark_t3_masked_chunk_logits_loop import logits_summary
from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from probe_t3_masked_cache_block_vulkan import compare_arrays, to_host_array


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_exportability" / "masked_cache"


def selected_text(name: str) -> str:
    cases = {
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }
    if name not in cases:
        raise ValueError(f"Unknown case {name!r}; available={sorted(cases)}")
    return cases[name]


def topk_overlap(actual: np.ndarray, expected: np.ndarray, k: int = 10) -> int:
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    return len(set(np.argsort(actual_flat)[-k:].tolist()) & set(np.argsort(expected_flat)[-k:].tolist()))


def token_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def past_to_vulkan_cache(
    past_key_values: Any,
    max_len: int,
) -> list[np.ndarray]:
    cache: list[np.ndarray] = []
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    for item in past_key_values:
        key, value = item[0], item[1]
        cache.append(padded_reference_cache(key.detach().cpu().numpy(), max_len))
        cache.append(padded_reference_cache(value.detach().cpu().numpy(), max_len))
    return cache


def compare_past_to_loop_cache(
    past_key_values: Any,
    loop: VulkanFullMaskedT3Loop,
    max_len: int,
) -> dict[str, Any]:
    comparisons = []
    flat_cpu = []
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()
    for item in past_key_values:
        key, value = item[0], item[1]
        flat_cpu.append(key.detach().cpu().numpy())
        flat_cpu.append(value.detach().cpu().numpy())

    for index, cpu_cache in enumerate(flat_cpu):
        comparisons.append(
            {
                "index": index,
                "layer": index // 2,
                "kind": "key" if index % 2 == 0 else "value",
                "comparison": compare_arrays(
                    to_host_array(loop.cache[index]),
                    padded_reference_cache(cpu_cache, max_len),
                ),
            }
        )

    return {
        "allclose_1e_4": all(item["comparison"]["allclose_1e_4"] for item in comparisons),
        "allclose_1e_3": all(item["comparison"]["allclose_1e_3"] for item in comparisons),
        "max_abs_error": max(item["comparison"]["max_abs_error"] for item in comparisons),
        "comparisons": comparisons,
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(args.seed)

    text = selected_text(args.case)
    normalized = punc_norm(text)

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_seconds = time.perf_counter() - load_started
    t3 = model.t3.cpu().eval()
    t3.tfmr.config._attn_implementation = "eager"
    for block in t3.tfmr.h:
        block.attn.config._attn_implementation = "eager"

    text_tokens = model.tokenizer(normalized, return_tensors="pt", padding=True, truncation=True)
    text_ids = text_tokens.input_ids.to(dtype=torch.long, device=t3.device)
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_ids[:, :1])

    with torch.inference_mode():
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_ids,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )
        prefill_started = time.perf_counter()
        prefill = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        prefill_seconds = time.perf_counter() - prefill_started
        initial_logits = t3.speech_head(prefill[0][:, -1:])
        current_token = token_from_logits(initial_logits)
        initial_argmax_token = int(current_token[0, 0].detach().cpu().item())
        past = prefill.past_key_values

    initial_context_len = int(embeds.shape[1])
    if initial_context_len + args.steps >= args.max_len:
        raise SystemExit(
            f"initial context {initial_context_len} + steps {args.steps} exceeds max_len {args.max_len}"
        )

    loop = VulkanFullMaskedT3Loop(max_len=args.max_len, compiled_valid_len=args.compiled_valid_len)
    loop.cache = [loop.as_device(array.astype(np.float32, copy=False)) for array in past_to_vulkan_cache(past, args.max_len)]

    step_results = []
    cpu_step_seconds = 0.0
    vulkan_step_seconds = 0.0
    with torch.inference_mode():
        for step in range(args.steps):
            current_embed = t3.speech_emb(current_token)
            position_ids = torch.full(
                (current_embed.shape[0], current_embed.shape[1]),
                initial_context_len + step,
                dtype=torch.long,
                device=current_embed.device,
            )
            # GPT2Model adds position embeddings before block 0 even when
            # inputs_embeds are provided. The Vulkan block stack starts at
            # block 0, so the input must include the same position embedding.
            vulkan_input = current_embed + t3.tfmr.wpe(position_ids)

            cpu_started = time.perf_counter()
            cpu_out = t3.tfmr(
                inputs_embeds=current_embed,
                past_key_values=past,
                use_cache=True,
            )
            cpu_logits = t3.speech_head(cpu_out[0])
            cpu_step_seconds += time.perf_counter() - cpu_started

            vulkan_started = time.perf_counter()
            vulkan_hidden, vulkan_logits = loop.step(
                vulkan_input.detach().cpu().numpy().astype(np.float32, copy=False),
                valid_len=initial_context_len + step,
                update_cache=True,
            )
            vulkan_logits_np = to_host_array(vulkan_logits)
            vulkan_step_seconds += time.perf_counter() - vulkan_started

            cpu_hidden_np = cpu_out[0].detach().cpu().numpy()
            cpu_logits_np = cpu_logits.detach().cpu().numpy()
            vulkan_hidden_np = to_host_array(vulkan_hidden)
            vulkan_hidden_final = t3.tfmr.ln_f(torch.from_numpy(vulkan_hidden_np))
            logits_compare = logits_summary(vulkan_logits_np, cpu_logits_np)
            step_results.append(
                {
                    "step": step,
                    "valid_len_before_step": initial_context_len + step,
                    "input_token": int(current_token[0, 0].detach().cpu().item()),
                    "cpu_next_argmax": int(torch.argmax(cpu_logits[:, -1, :]).detach().cpu().item()),
                    "vulkan_next_argmax": logits_compare["actual_argmax"],
                    "hidden": compare_arrays(vulkan_hidden_final.detach().cpu().numpy(), cpu_hidden_np),
                    "logits": logits_compare,
                    "top10_overlap": topk_overlap(vulkan_logits_np, cpu_logits_np, k=10),
                }
            )

            past = cpu_out.past_key_values
            current_token = token_from_logits(cpu_logits)

    cache_compare = compare_past_to_loop_cache(past, loop, args.max_len)
    return {
        "case": args.case,
        "chars": len(text),
        "normalized_chars": len(normalized),
        "text_tokens": int(text_ids.shape[1]),
        "conditioning_tokens": int(len_cond),
        "initial_context_len": initial_context_len,
        "steps": args.steps,
        "max_len": args.max_len,
        "compiled_valid_len": args.compiled_valid_len,
        "model_load_seconds": load_seconds,
        "cpu_prefill_seconds": prefill_seconds,
        "cpu_followon_seconds": cpu_step_seconds,
        "vulkan_followon_seconds": vulkan_step_seconds,
        "cpu_followon_ms_per_step": cpu_step_seconds * 1000.0 / args.steps,
        "vulkan_followon_ms_per_step": vulkan_step_seconds * 1000.0 / args.steps,
        "initial_cpu_argmax": initial_argmax_token,
        "validation": {
            "hidden_allclose_1e_4": all(item["hidden"]["allclose_1e_4"] for item in step_results),
            "hidden_allclose_1e_3": all(item["hidden"]["allclose_1e_3"] for item in step_results),
            "logits_allclose_1e_4": all(item["logits"]["allclose_1e_4"] for item in step_results),
            "logits_allclose_1e_3": all(item["logits"]["allclose_1e_3"] for item in step_results),
            "argmax_matches": sum(
                1 for item in step_results if item["logits"]["argmax_match"]
            ),
            "argmax_total": len(step_results),
            "min_top5_overlap": min(item["logits"]["top5_overlap"] for item in step_results),
            "min_top10_overlap": min(item["top10_overlap"] for item in step_results),
            "max_hidden_abs_error": max(item["hidden"]["max_abs_error"] for item in step_results),
            "max_logits_abs_error": max(item["logits"]["max_abs_error"] for item in step_results),
            "cache": cache_compare,
            "steps": step_results,
        },
        "notes": [
            "CPU computes the real initial prompt prefill. Vulkan handles only follow-on single-token T3 body steps.",
            "Next-token inputs are chosen from CPU argmax logits to keep CPU and Vulkan trajectories aligned.",
            "This applies GPT2 position embeddings before the Vulkan block stack and uses the Vulkan final ln_f+speech_head artifact for logits.",
            "CPU final ln_f is used only for the hidden-state diagnostic comparison.",
            "This is offline validation only; the live API remains CPU-backed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="hello", choices=("hello", "short", "chunk270"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--compiled-valid-len", type=int, default=935)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_real_prefill_vulkan_followon_latest.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    report = run_case(args)
    report["seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    validation = report["validation"]
    print(f"Wrote {args.output}")
    print(f"case={report['case']} initial_context_len={report['initial_context_len']} steps={report['steps']}")
    print(f"cpu_prefill_seconds={report['cpu_prefill_seconds']:.3f}")
    print(f"cpu_followon_ms_per_step={report['cpu_followon_ms_per_step']:.3f}")
    print(f"vulkan_followon_ms_per_step={report['vulkan_followon_ms_per_step']:.3f}")
    print(f"hidden_allclose_1e_4={validation['hidden_allclose_1e_4']}")
    print(f"logits_allclose_1e_4={validation['logits_allclose_1e_4']}")
    print(f"logits_allclose_1e_3={validation['logits_allclose_1e_3']}")
    print(f"argmax_matches={validation['argmax_matches']}/{validation['argmax_total']}")
    print(f"min_top5_overlap={validation['min_top5_overlap']}")
    print(f"max_logits_abs_error={validation['max_logits_abs_error']:.3e}")
    print(f"cache_allclose_1e_4={validation['cache']['allclose_1e_4']}")
    print(f"max_cache_abs_error={validation['cache']['max_abs_error']:.3e}")


if __name__ == "__main__":
    main()
