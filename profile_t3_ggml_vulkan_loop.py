#!/usr/bin/env python3
"""Profile the experimental ggml/Vulkan T3 token loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, _sample_fast, _temporary_torch_threads


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


def logits_processors(temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    return processors


def legacy_past(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def profile_t3(
    model: ChatterboxTurboTTS,
    runtime: T3GGMLVulkanRuntime,
    text: str,
    *,
    seed: int,
    max_gen_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    fast_loop: bool = False,
    fast_sampler: bool = False,
    prefill_threads: int = 0,
) -> dict:
    torch.manual_seed(seed)
    t3 = model.t3
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)
    processors = logits_processors(temperature, top_k, top_p, repetition_penalty)
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

    timings = {
        "prepare_embeds_seconds": 0.0,
        "prefill_seconds": 0.0,
        "initial_sample_seconds": 0.0,
        "cache_upload_seconds": 0.0,
        "fast_loop_setup_seconds": 0.0,
        "step_embedding_seconds": 0.0,
        "step_ggml_wall_seconds": 0.0,
        "step_ggml_reported_seconds": 0.0,
        "step_sampling_seconds": 0.0,
    }
    total_start = time.perf_counter()

    started = time.perf_counter()
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    timings["prepare_embeds_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    with _temporary_torch_threads(prefill_threads):
        llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    timings["prefill_seconds"] = time.perf_counter() - started
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values
    initial_context_len = int(embeds.shape[1])

    started = time.perf_counter()
    speech_logits = t3.speech_head(hidden_states[:, -1:])
    if fast_sampler:
        next_speech_token = _sample_fast(
            speech_start_token,
            speech_logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    else:
        processed_logits = processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
    timings["initial_sample_seconds"] = time.perf_counter() - started
    if next_speech_token is None:
        return {
            "normalized_chars": len(text),
            "seed": seed,
            "max_gen_len": max_gen_len,
            "fast_loop": fast_loop,
            "fast_sampler": fast_sampler,
            "prefill_threads": prefill_threads,
            "initial_context_len": initial_context_len,
            "generated_tokens": 0,
            "tokens": [],
            "loop_iterations": 0,
            "total_seconds": time.perf_counter() - total_start,
            "timings": timings,
            "per_step": {
                "ggml_wall_mean_ms": None,
                "ggml_wall_min_ms": None,
                "ggml_wall_max_ms": None,
                "ggml_reported_mean_ms": None,
                "ggml_reported_min_ms": None,
                "ggml_reported_max_ms": None,
            },
            "cache_prefix_available": runtime._set_layer_cache_prefix is not None,
        }

    started = time.perf_counter()
    runtime.set_cache_from_past(past_key_values, initial_context_len)
    timings["cache_upload_seconds"] = time.perf_counter() - started

    generated_speech_tokens = torch.empty(
        next_speech_token.size(0),
        max_gen_len + 1,
        dtype=next_speech_token.dtype,
        device=next_speech_token.device,
    )
    generated_speech_tokens[:, 0:1] = next_speech_token
    generated_count = 1
    current_speech_token = next_speech_token
    reported_ms = []
    wall_times = []
    max_steps = min(max_gen_len, runtime.max_len - initial_context_len)
    precomputed_masks = None
    speech_embedding_np = None
    position_embedding_np = None
    if fast_loop:
        started = time.perf_counter()
        if not hasattr(t3.tfmr, "wpe"):
            raise ValueError("fast ggml T3 loop currently requires GPT-2 position embeddings")
        precomputed_masks = runtime.masks_for_valid_lens(initial_context_len, max_steps)
        speech_embedding_np = t3.speech_emb.weight.detach().to(device="cpu", dtype=torch.float32).numpy()
        position_embedding_np = t3.tfmr.wpe.weight.detach().to(device="cpu", dtype=torch.float32).numpy()
        timings["fast_loop_setup_seconds"] = time.perf_counter() - started

    for _ in range(max_gen_len):
        slot_index = initial_context_len + generated_count - 1
        if slot_index >= runtime.max_len:
            break

        started = time.perf_counter()
        if fast_loop:
            assert precomputed_masks is not None
            assert speech_embedding_np is not None
            assert position_embedding_np is not None
            token_id = int(current_speech_token.detach().cpu().reshape(-1)[0].item())
            hidden_np = speech_embedding_np[token_id] + position_embedding_np[slot_index]
            attn_mask = precomputed_masks[generated_count - 1]
        else:
            current_speech_embed = t3.speech_emb(current_speech_token)
            position_ids = torch.full(
                (current_speech_embed.shape[0], current_speech_embed.shape[1]),
                slot_index,
                dtype=torch.long,
                device=current_speech_embed.device,
            )
            current_hidden = current_speech_embed + t3.tfmr.wpe(position_ids)
            hidden_np = current_hidden[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
            attn_mask = None
        timings["step_embedding_seconds"] += time.perf_counter() - started

        started = time.perf_counter()
        if attn_mask is None:
            logits_np, elapsed_ms = runtime.run_step(hidden_np, slot_index)
        else:
            logits_np, elapsed_ms = runtime.run_step_with_mask(hidden_np, attn_mask, slot_index)
        wall = time.perf_counter() - started
        wall_times.append(wall)
        reported_ms.append(elapsed_ms)
        timings["step_ggml_wall_seconds"] += wall
        timings["step_ggml_reported_seconds"] += elapsed_ms / 1000.0

        started = time.perf_counter()
        speech_logits = torch.from_numpy(logits_np).to(device=generated_speech_tokens.device).unsqueeze(0)
        input_ids = generated_speech_tokens[:, :generated_count]
        if fast_sampler:
            next_speech_token = _sample_fast(
                input_ids,
                speech_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        else:
            processed_logits = processors(input_ids, speech_logits)
            if torch.all(processed_logits == -float("inf")):
                break
            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)
        if next_speech_token is None:
            break
        generated_speech_tokens[:, generated_count : generated_count + 1] = next_speech_token
        generated_count += 1
        current_speech_token = next_speech_token
        timings["step_sampling_seconds"] += time.perf_counter() - started
        if torch.all(next_speech_token == t3.hp.stop_speech_token):
            break

    all_tokens = generated_speech_tokens[:, :generated_count]
    if all_tokens.size(1) > 0 and all_tokens[0, -1] == t3.hp.stop_speech_token:
        all_tokens = all_tokens[:, :-1]

    total_seconds = time.perf_counter() - total_start
    return {
        "normalized_chars": len(text),
        "seed": seed,
        "max_gen_len": max_gen_len,
        "fast_loop": fast_loop,
        "fast_sampler": fast_sampler,
        "prefill_threads": prefill_threads,
        "initial_context_len": initial_context_len,
        "generated_tokens": int(all_tokens.numel()),
        "tokens": [int(item) for item in all_tokens.detach().cpu().reshape(-1).tolist()],
        "loop_iterations": len(wall_times),
        "total_seconds": total_seconds,
        "timings": timings,
        "per_step": {
            "ggml_wall_mean_ms": float(np.mean(wall_times) * 1000.0) if wall_times else None,
            "ggml_wall_min_ms": float(np.min(wall_times) * 1000.0) if wall_times else None,
            "ggml_wall_max_ms": float(np.max(wall_times) * 1000.0) if wall_times else None,
            "ggml_reported_mean_ms": float(np.mean(reported_ms)) if reported_ms else None,
            "ggml_reported_min_ms": float(np.min(reported_ms)) if reported_ms else None,
            "ggml_reported_max_ms": float(np.max(reported_ms)) if reported_ms else None,
        },
        "cache_prefix_available": runtime._set_layer_cache_prefix is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("hello", "short", "chunk270"), default="chunk270")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--prefill-threads", type=int, default=0)
    parser.add_argument("--fast-sampler", action="store_true")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--lib-path", type=Path)
    parser.add_argument(
        "--loop-mode",
        choices=("baseline", "fast", "compare"),
        default="baseline",
        help="Profile the current Python loop, the experimental fast loop, or both with token comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "t3_ggml_vulkan_loop_profile_2026-07-08.json",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass

    text = {
        "hello": "Hello world, this is a test.",
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[args.case]

    model = ChatterboxTurboTTS.from_pretrained("cpu")
    runtime_kwargs = {"lib_path": args.lib_path} if args.lib_path is not None else {}
    runtime = T3GGMLVulkanRuntime(**runtime_kwargs)

    def run_reports(fast_loop: bool) -> list[dict]:
        return [
            profile_t3(
                model,
                runtime,
                text,
                seed=args.seed,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                fast_loop=fast_loop,
                fast_sampler=args.fast_sampler,
                prefill_threads=args.prefill_threads,
            )
            for _ in range(args.requests)
        ]

    def summarize(reports: list[dict]) -> dict:
        return {
            "mean_total_seconds": float(np.mean([item["total_seconds"] for item in reports])) if reports else None,
            "min_total_seconds": float(np.min([item["total_seconds"] for item in reports])) if reports else None,
            "max_total_seconds": float(np.max([item["total_seconds"] for item in reports])) if reports else None,
        }

    if args.loop_mode == "compare":
        baseline_reports = run_reports(False)
        fast_reports = run_reports(True)
        comparisons = []
        for index, (baseline, fast) in enumerate(zip(baseline_reports, fast_reports, strict=True)):
            comparisons.append(
                {
                    "request": index,
                    "tokens_equal": baseline["tokens"] == fast["tokens"],
                    "baseline_generated_tokens": baseline["generated_tokens"],
                    "fast_generated_tokens": fast["generated_tokens"],
                    "baseline_total_seconds": baseline["total_seconds"],
                    "fast_total_seconds": fast["total_seconds"],
                    "speedup": (
                        baseline["total_seconds"] / fast["total_seconds"]
                        if fast["total_seconds"] > 0.0
                        else None
                    ),
                }
            )
        baseline_summary = summarize(baseline_reports)
        fast_summary = summarize(fast_reports)
        report = {
            "case": args.case,
            "requests": args.requests,
            "loop_mode": args.loop_mode,
            "fast_sampler": args.fast_sampler,
            "prefill_threads": args.prefill_threads,
            "results_by_mode": {
                "baseline": baseline_reports,
                "fast": fast_reports,
            },
            "comparisons": comparisons,
            "summary": {
                "baseline": baseline_summary,
                "fast": fast_summary,
                "mean_speedup": (
                    baseline_summary["mean_total_seconds"] / fast_summary["mean_total_seconds"]
                    if baseline_summary["mean_total_seconds"] and fast_summary["mean_total_seconds"]
                    else None
                ),
                "all_tokens_equal": all(item["tokens_equal"] for item in comparisons),
            },
        }
        reports = baseline_reports + fast_reports
    else:
        reports = run_reports(args.loop_mode == "fast")
        report = {
            "case": args.case,
            "requests": args.requests,
            "loop_mode": args.loop_mode,
            "fast_sampler": args.fast_sampler,
            "prefill_threads": args.prefill_threads,
            "results": reports,
            "summary": summarize(reports),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}")
    if args.loop_mode == "compare":
        for item in report["comparisons"]:
            print(
                f"compare_request={item['request']} tokens_equal={item['tokens_equal']} "
                f"baseline_seconds={item['baseline_total_seconds']:.3f} "
                f"fast_seconds={item['fast_total_seconds']:.3f} "
                f"speedup={item['speedup']:.3f}"
            )
        print(f"all_tokens_equal={report['summary']['all_tokens_equal']}")
        print(f"mean_speedup={report['summary']['mean_speedup']:.3f}")
    for index, item in enumerate(reports):
        mode = "fast" if item["fast_loop"] else "baseline"
        print(f"request={index} mode={mode} total_seconds={item['total_seconds']:.3f}")
        print(f"generated_tokens={item['generated_tokens']} iterations={item['loop_iterations']}")
        for key, value in item["timings"].items():
            print(f"{key}={value:.3f}")
        print(f"ggml_wall_mean_ms={item['per_step']['ggml_wall_mean_ms']:.3f}")
        print(f"ggml_reported_mean_ms={item['per_step']['ggml_reported_mean_ms']:.3f}")


if __name__ == "__main__":
    main()
