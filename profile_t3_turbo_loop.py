#!/usr/bin/env python3
"""Micro-profile Chatterbox Turbo T3 token generation on CPU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "t3_profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def selected_text(case: str) -> str:
    if case == "short":
        return SHORT_TEXT
    if case == "chunk270":
        return CHUNK_270
    if case == "hello":
        return "Hello world, this is a test."
    raise ValueError(case)


def build_processors(
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> LogitsProcessorList:
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


def tokenize(model: ChatterboxTurboTTS, text: str) -> torch.Tensor:
    text = punc_norm(text)
    text_tokens = model.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return text_tokens.input_ids.to(model.device)


def timed_add(timers: dict[str, float], key: str, started_ms: float) -> None:
    timers[key] = timers.get(key, 0.0) + (now_ms() - started_ms)


@torch.inference_mode()
def run_instrumented(
    model: ChatterboxTurboTTS,
    text_tokens: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_gen_len: int,
    use_preallocated_ids: bool,
) -> dict[str, Any]:
    t3 = model.t3
    timers: dict[str, float] = {}
    step_times: list[float] = []

    total_started = now_ms()
    started = now_ms()
    processors = build_processors(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    timed_add(timers, "build_processors_ms", started)

    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    started = now_ms()
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    timed_add(timers, "prepare_input_embeds_ms", started)

    started = now_ms()
    llm_outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
    timed_add(timers, "initial_tfmr_ms", started)

    started = now_ms()
    hidden_states = llm_outputs[0]
    past_key_values = llm_outputs.past_key_values
    speech_hidden = hidden_states[:, -1:]
    speech_logits = t3.speech_head(speech_hidden)
    timed_add(timers, "initial_speech_head_ms", started)

    started = now_ms()
    processed_logits = processors(speech_start_token, speech_logits[:, -1, :])
    probs = F.softmax(processed_logits, dim=-1)
    next_speech_token = torch.multinomial(probs, num_samples=1)
    timed_add(timers, "initial_sampling_ms", started)

    generated_speech_tokens = [next_speech_token]
    if use_preallocated_ids:
        generated_ids = torch.empty(
            text_tokens.shape[0],
            max_gen_len + 1,
            dtype=next_speech_token.dtype,
            device=next_speech_token.device,
        )
        generated_ids[:, 0:1] = next_speech_token
    else:
        generated_ids = None

    current_speech_token = next_speech_token
    stop_reason = "max_gen_len"

    for step in range(max_gen_len):
        step_started = now_ms()

        started = now_ms()
        current_speech_embed = t3.speech_emb(current_speech_token)
        timed_add(timers, "loop_speech_emb_ms", started)

        started = now_ms()
        llm_outputs = t3.tfmr(
            inputs_embeds=current_speech_embed,
            past_key_values=past_key_values,
            use_cache=True,
        )
        timed_add(timers, "loop_tfmr_ms", started)

        started = now_ms()
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values
        speech_logits = t3.speech_head(hidden_states)
        timed_add(timers, "loop_speech_head_ms", started)

        started = now_ms()
        if use_preallocated_ids:
            input_ids = generated_ids[:, : step + 1]
        else:
            input_ids = torch.cat(generated_speech_tokens, dim=1)
        timed_add(timers, "loop_input_ids_ms", started)

        started = now_ms()
        processed_logits = processors(input_ids, speech_logits[:, -1, :])
        timed_add(timers, "loop_logits_processors_ms", started)

        if torch.all(processed_logits == -float("inf")):
            stop_reason = "all_logits_inf"
            break

        started = now_ms()
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)
        timed_add(timers, "loop_sampling_ms", started)

        generated_speech_tokens.append(next_speech_token)
        if use_preallocated_ids:
            generated_ids[:, step + 1 : step + 2] = next_speech_token
        current_speech_token = next_speech_token

        step_times.append(now_ms() - step_started)
        if torch.all(next_speech_token == t3.hp.stop_speech_token):
            stop_reason = "eos"
            break

    started = now_ms()
    all_tokens = torch.cat(generated_speech_tokens, dim=1)
    if all_tokens.size(1) > 0 and all_tokens[0, -1] == t3.hp.stop_speech_token:
        all_tokens = all_tokens[:, :-1]
    timed_add(timers, "final_cat_trim_ms", started)

    total_ms = now_ms() - total_started
    loop_total_ms = sum(step_times)
    return {
        "total_ms": total_ms,
        "token_count": int(all_tokens.numel()),
        "raw_generated_count": int(len(generated_speech_tokens)),
        "stop_reason": stop_reason,
        "timers_ms": timers,
        "loop_total_ms": loop_total_ms,
        "loop_step_count": len(step_times),
        "loop_mean_ms": loop_total_ms / len(step_times) if step_times else 0.0,
        "loop_p50_ms": float(torch.tensor(step_times).median().item()) if step_times else 0.0,
        "loop_p95_ms": float(torch.quantile(torch.tensor(step_times), 0.95).item()) if step_times else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["hello", "short", "chunk270"], default="short")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--max-gen-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "t3_profile_latest.json")
    parser.add_argument("--skip-no-repetition", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)

    load_started = now_ms()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_ms = now_ms() - load_started
    text = selected_text(args.case)
    text_tokens = tokenize(model, text)

    variants = [
        {
            "name": "default_list_cat",
            "temperature": 0.8,
            "top_k": 1000,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "use_preallocated_ids": False,
        },
        {
            "name": "default_preallocated_ids",
            "temperature": 0.8,
            "top_k": 1000,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "use_preallocated_ids": True,
        },
    ]
    if not args.skip_no_repetition:
        variants.append(
            {
                "name": "no_repetition_penalty",
                "temperature": 0.8,
                "top_k": 1000,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "use_preallocated_ids": True,
            }
        )

    results = {
        "case": args.case,
        "chars": len(text),
        "text_token_shape": list(text_tokens.shape),
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "model_load_ms": load_ms,
        "max_gen_len": args.max_gen_len,
        "variants": [],
    }

    for index, variant in enumerate(variants):
        torch.manual_seed(args.seed)
        print(f"running {variant['name']}")
        result = run_instrumented(
            model,
            text_tokens,
            temperature=variant["temperature"],
            top_k=variant["top_k"],
            top_p=variant["top_p"],
            repetition_penalty=variant["repetition_penalty"],
            max_gen_len=args.max_gen_len,
            use_preallocated_ids=variant["use_preallocated_ids"],
        )
        result.update(variant)
        results["variants"].append(result)
        print(
            f"{variant['name']}: total={result['total_ms']:.1f}ms "
            f"tokens={result['token_count']} loop_mean={result['loop_mean_ms']:.1f}ms "
            f"stop={result['stop_reason']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
