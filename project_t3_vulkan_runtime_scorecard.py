#!/usr/bin/env python3
"""Build a measured scorecard for the BC-250 T3 Vulkan runtime outlook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exports"
T3_DIR = EXPORT_DIR / "t3_exportability"
OUT_DIR = EXPORT_DIR / "benchmarks"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_benchmark_mean_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    data = load_json(path)
    for item in data.get("benchmarks", []):
        if item.get("run_name", "").endswith("_mean") or item.get("name", "").endswith("_mean"):
            return float(item["real_time"])
    for item in data.get("benchmarks", []):
        if "real_time" in item:
            return float(item["real_time"])
    return None


def full_loop_step_ms(path: Path) -> float | None:
    if not path.exists():
        return None
    data = load_json(path)
    timing = data["benchmark"]["timings"]["full_t3_logits_no_fetch"]
    return float(timing["mean_ms_per_step"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_json(OUT_DIR / "performance_baseline_scorecard_2026-07-08.json")
    api_chunk270 = baseline["full_api_cpu_vs_5090"]["results"][2]
    cpu_api_270_s = float(api_chunk270["local_wall_seconds"])
    rtx_api_270_s = float(api_chunk270["remote_wall_seconds"])

    hift_chunk = load_json(T3_DIR.parent / "split_hift_vulkan" / "vulkan_hift_stage_chunk270_2026-07-08.json")
    t3_cpu_270_s = float(hift_chunk["stage_ms"]["t3"]) / 1000.0
    s3_flow_270_s = float(hift_chunk["stage_ms"]["s3_flow"]) / 1000.0
    hift_cpu_270_s = float(hift_chunk["totals_ms"]["cpu_hift_stage"]) / 1000.0
    hift_vulkan_270_s = float(hift_chunk["totals_ms"]["vulkan_hift_stage"]) / 1000.0

    masked_stack_bench = first_benchmark_mean_ms(
        T3_DIR / "masked_cache" / "gpt2_masked_cache_stack_s0_l4_p128_valid42_t1_benchmark.json"
    )
    speech_head_bench = first_benchmark_mean_ms(T3_DIR / "speech_head_t1_benchmark.json")

    cache_update = load_json(T3_DIR / "cache_update" / "t3_kv_cache_slot_update_vulkan_2026-07-08.json")
    four_slot_updates_ms = float(
        cache_update["runtime"]["timings"]["four_slot_updates_from_cached_t3_outputs"]["mean_ms"]
    )

    logits_loop_32 = load_json(T3_DIR / "masked_cache" / "t3_masked_chunk_logits_loop_32step_2026-07-08.json")
    one_chunk_loop_ms = float(
        logits_loop_32["benchmark"]["timings"]["loop_logits_no_fetch"]["mean_ms_per_step"]
    )
    full_loop_32_path = T3_DIR / "masked_cache" / "t3_full_masked_vulkan_loop_32step_2026-07-08.json"
    full_loop_128_path = T3_DIR / "masked_cache" / "t3_full_masked_vulkan_loop_128step_2026-07-08.json"
    full_loop_p1024_32_path = T3_DIR / "masked_cache" / "t3_full_masked_vulkan_loop_p1024_32step_2026-07-08.json"
    full_loop_p1024_finalhead_8_path = (
        T3_DIR / "masked_cache" / "t3_full_masked_vulkan_loop_p1024_finalhead_8step_2026-07-08.json"
    )
    real_prefill_chunk270_path = (
        T3_DIR
        / "masked_cache"
        / "t3_real_prefill_vulkan_followon_chunk270_finalhead_32step_fetchlogits_2026-07-08.json"
    )
    final_norm_head_path = T3_DIR / "t3_final_norm_speech_head_vulkan_2026-07-08.json"
    prompt_budget_path = T3_DIR / "t3_real_prompt_length_budget_2026-07-08.json"
    full_loop_32 = load_json(full_loop_32_path) if full_loop_32_path.exists() else None
    full_loop_128 = load_json(full_loop_128_path) if full_loop_128_path.exists() else None
    full_loop_p1024_32 = load_json(full_loop_p1024_32_path) if full_loop_p1024_32_path.exists() else None
    full_loop_p1024_finalhead_8 = (
        load_json(full_loop_p1024_finalhead_8_path)
        if full_loop_p1024_finalhead_8_path.exists()
        else None
    )
    real_prefill_chunk270 = (
        load_json(real_prefill_chunk270_path) if real_prefill_chunk270_path.exists() else None
    )
    final_norm_head = load_json(final_norm_head_path) if final_norm_head_path.exists() else None
    prompt_budget = load_json(prompt_budget_path) if prompt_budget_path.exists() else None
    actual_full_loop_32_token_ms = full_loop_step_ms(full_loop_32_path)
    actual_full_loop_128_token_ms = full_loop_step_ms(full_loop_128_path)
    actual_full_loop_p1024_32_token_ms = full_loop_step_ms(full_loop_p1024_32_path)
    actual_full_loop_p1024_finalhead_8_token_ms = full_loop_step_ms(full_loop_p1024_finalhead_8_path)
    actual_full_loop_token_ms = actual_full_loop_128_token_ms or actual_full_loop_32_token_ms
    real_prefill_vulkan_ms = (
        float(real_prefill_chunk270["vulkan_followon_ms_per_step"])
        if real_prefill_chunk270
        else None
    )
    real_prefill_cpu_ms = (
        float(real_prefill_chunk270["cpu_followon_ms_per_step"])
        if real_prefill_chunk270
        else None
    )
    real_prefill_cpu_prefill_s = (
        float(real_prefill_chunk270["cpu_prefill_seconds"])
        if real_prefill_chunk270
        else None
    )

    # Existing CPU/Vulkan chunk timings from the T3 findings.
    cpu_four_layer_ms = 14.25
    vulkan_four_layer_ms = float(masked_stack_bench or 11.6)
    speech_head_ms = float(speech_head_bench or 0.808)

    estimated_tokens = t3_cpu_270_s / ((cpu_four_layer_ms * 6.0) / 1000.0)
    optimistic_token_ms = vulkan_four_layer_ms * 6.0 + four_slot_updates_ms * 6.0 + speech_head_ms
    python_loop_token_ms = one_chunk_loop_ms * 6.0

    optimistic_t3_s = estimated_tokens * optimistic_token_ms / 1000.0
    python_loop_t3_s = estimated_tokens * python_loop_token_ms / 1000.0
    actual_full_loop_t3_s = (
        estimated_tokens * actual_full_loop_token_ms / 1000.0
        if actual_full_loop_token_ms is not None
        else None
    )
    p1024_real_bucket_t3_s = (
        estimated_tokens * actual_full_loop_p1024_32_token_ms / 1000.0
        if actual_full_loop_p1024_32_token_ms is not None
        else None
    )
    p1024_finalhead_t3_s = (
        estimated_tokens * actual_full_loop_p1024_finalhead_8_token_ms / 1000.0
        if actual_full_loop_p1024_finalhead_8_token_ms is not None
        else None
    )
    real_prefill_vulkan_t3_s = (
        real_prefill_cpu_prefill_s + estimated_tokens * real_prefill_vulkan_ms / 1000.0
        if real_prefill_cpu_prefill_s is not None and real_prefill_vulkan_ms is not None
        else None
    )
    real_prefill_cpu_t3_s = (
        real_prefill_cpu_prefill_s + estimated_tokens * real_prefill_cpu_ms / 1000.0
        if real_prefill_cpu_prefill_s is not None and real_prefill_cpu_ms is not None
        else None
    )

    cpu_hift_to_vulkan_saved_s = hift_cpu_270_s - hift_vulkan_270_s
    optimistic_full_s = cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + optimistic_t3_s
    python_loop_full_s = cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + python_loop_t3_s
    actual_full_loop_full_s = (
        cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + actual_full_loop_t3_s
        if actual_full_loop_t3_s is not None
        else None
    )
    p1024_real_bucket_full_s = (
        cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + p1024_real_bucket_t3_s
        if p1024_real_bucket_t3_s is not None
        else None
    )
    p1024_finalhead_full_s = (
        cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + p1024_finalhead_t3_s
        if p1024_finalhead_t3_s is not None
        else None
    )
    real_prefill_vulkan_full_s = (
        cpu_api_270_s - t3_cpu_270_s - cpu_hift_to_vulkan_saved_s + real_prefill_vulkan_t3_s
        if real_prefill_vulkan_t3_s is not None
        else None
    )

    report = {
        "sources": {
            "baseline": (OUT_DIR / "performance_baseline_scorecard_2026-07-08.json").as_posix(),
            "stage_chunk270": (T3_DIR.parent / "split_hift_vulkan" / "vulkan_hift_stage_chunk270_2026-07-08.json").as_posix(),
            "masked_stack_benchmark": (T3_DIR / "masked_cache" / "gpt2_masked_cache_stack_s0_l4_p128_valid42_t1_benchmark.json").as_posix(),
            "speech_head_benchmark": (T3_DIR / "speech_head_t1_benchmark.json").as_posix(),
            "cache_update": (T3_DIR / "cache_update" / "t3_kv_cache_slot_update_vulkan_2026-07-08.json").as_posix(),
            "logits_loop_32": (T3_DIR / "masked_cache" / "t3_masked_chunk_logits_loop_32step_2026-07-08.json").as_posix(),
            "full_loop_32": full_loop_32_path.as_posix(),
            "full_loop_128": full_loop_128_path.as_posix(),
            "full_loop_p1024_32": full_loop_p1024_32_path.as_posix(),
            "full_loop_p1024_finalhead_8": full_loop_p1024_finalhead_8_path.as_posix(),
            "real_prefill_chunk270": real_prefill_chunk270_path.as_posix(),
            "final_norm_speech_head": final_norm_head_path.as_posix(),
            "prompt_budget": prompt_budget_path.as_posix(),
        },
        "baseline": {
            "cpu_api_270_s": cpu_api_270_s,
            "rtx_5090_api_270_s": rtx_api_270_s,
            "cpu_t3_270_s": t3_cpu_270_s,
            "cpu_s3_flow_270_s": s3_flow_270_s,
            "cpu_hift_270_s": hift_cpu_270_s,
            "vulkan_hift_270_s": hift_vulkan_270_s,
        },
        "measured_vulkan_t3_components_ms": {
            "masked_four_layer_chunk_steady": vulkan_four_layer_ms,
            "four_layer_cache_slot_updates": four_slot_updates_ms,
            "speech_head_steady": speech_head_ms,
            "one_chunk_plus_cache_plus_speech_head_python_loop": one_chunk_loop_ms,
            "actual_full_24_layer_plus_cache_plus_speech_head_32step": actual_full_loop_32_token_ms,
            "actual_full_24_layer_plus_cache_plus_speech_head_128step": actual_full_loop_128_token_ms,
            "actual_full_24_layer_plus_cache_plus_speech_head_p1024_32step": actual_full_loop_p1024_32_token_ms,
            "actual_full_24_layer_plus_cache_plus_finalhead_p1024_8step": actual_full_loop_p1024_finalhead_8_token_ms,
            "final_norm_speech_head_ms": (
                final_norm_head["runtime"]["timings"]["final_norm_speech_head_no_fetch"]["mean_ms"]
                if final_norm_head and final_norm_head.get("runtime")
                else None
            ),
        },
        "actual_full_loop_validation": {
            "step32": full_loop_32["validation"] if full_loop_32 else None,
            "step128": full_loop_128["validation"] if full_loop_128 else None,
            "p1024_step32": full_loop_p1024_32["validation"] if full_loop_p1024_32 else None,
            "p1024_finalhead_step8": (
                full_loop_p1024_finalhead_8["validation"] if full_loop_p1024_finalhead_8 else None
            ),
            "real_prefill_chunk270_step32": (
                real_prefill_chunk270["validation"] if real_prefill_chunk270 else None
            ),
        },
        "real_prefill_bridge": {
            "path": real_prefill_chunk270_path.as_posix(),
            "cpu_prefill_seconds": real_prefill_cpu_prefill_s,
            "cpu_followon_ms_per_step": real_prefill_cpu_ms,
            "vulkan_followon_ms_per_step": real_prefill_vulkan_ms,
            "estimated_cpu_bridge_t3_270_s": real_prefill_cpu_t3_s,
            "estimated_vulkan_bridge_t3_270_s": real_prefill_vulkan_t3_s,
        },
        "real_prompt_length_budget": {
            "path": prompt_budget_path.as_posix(),
            "cases": prompt_budget["cases"] if prompt_budget else None,
        },
        "projection": {
            "estimated_speech_tokens_for_270_case": estimated_tokens,
            "optimistic_full_24_layer_token_ms": optimistic_token_ms,
            "python_loop_full_24_layer_token_ms": python_loop_token_ms,
            "actual_full_loop_24_layer_token_ms": actual_full_loop_token_ms,
            "p1024_real_bucket_24_layer_token_ms": actual_full_loop_p1024_32_token_ms,
            "p1024_finalhead_24_layer_token_ms": actual_full_loop_p1024_finalhead_8_token_ms,
            "real_prefill_vulkan_followon_token_ms": real_prefill_vulkan_ms,
            "real_prefill_cpu_followon_token_ms": real_prefill_cpu_ms,
            "optimistic_t3_270_s": optimistic_t3_s,
            "python_loop_t3_270_s": python_loop_t3_s,
            "actual_full_loop_t3_270_s": actual_full_loop_t3_s,
            "p1024_real_bucket_t3_270_s": p1024_real_bucket_t3_s,
            "p1024_finalhead_t3_270_s": p1024_finalhead_t3_s,
            "real_prefill_vulkan_bridge_t3_270_s": real_prefill_vulkan_t3_s,
            "real_prefill_cpu_bridge_t3_270_s": real_prefill_cpu_t3_s,
            "optimistic_full_api_270_s_with_vulkan_t3_and_hift": optimistic_full_s,
            "python_loop_full_api_270_s_with_vulkan_t3_and_hift": python_loop_full_s,
            "actual_full_loop_api_270_s_with_vulkan_t3_and_hift": actual_full_loop_full_s,
            "p1024_real_bucket_api_270_s_with_vulkan_t3_and_hift": p1024_real_bucket_full_s,
            "p1024_finalhead_api_270_s_with_vulkan_t3_and_hift": p1024_finalhead_full_s,
            "real_prefill_vulkan_bridge_api_270_s_with_vulkan_hift": real_prefill_vulkan_full_s,
            "optimistic_slowdown_vs_5090": optimistic_full_s / rtx_api_270_s,
            "python_loop_slowdown_vs_5090": python_loop_full_s / rtx_api_270_s,
            "actual_full_loop_slowdown_vs_5090": (
                actual_full_loop_full_s / rtx_api_270_s
                if actual_full_loop_full_s is not None
                else None
            ),
            "p1024_real_bucket_slowdown_vs_5090": (
                p1024_real_bucket_full_s / rtx_api_270_s
                if p1024_real_bucket_full_s is not None
                else None
            ),
            "p1024_finalhead_slowdown_vs_5090": (
                p1024_finalhead_full_s / rtx_api_270_s
                if p1024_finalhead_full_s is not None
                else None
            ),
            "real_prefill_vulkan_bridge_slowdown_vs_5090": (
                real_prefill_vulkan_full_s / rtx_api_270_s
                if real_prefill_vulkan_full_s is not None
                else None
            ),
        },
        "interpretation": [
            "The optimistic model assumes six masked chunks run like the measured steady IREE benchmark and adds measured cache-update and speech-head costs.",
            "The Python-loop model multiplies the measured one-chunk Python runtime loop by six, which likely overstates overhead but reflects current integration style.",
            "The actual full-loop model uses the assembled six-chunk Vulkan runtime and measured fixed-slot cache updates through the speech head.",
            "The p128 full loop is too small for real Turbo prompts; the default conditioning alone makes initial context larger than 128 tokens.",
            "The p1024 full loop is the first real-bucket-size measurement for the 270-character case plus roughly 512 generated tokens.",
            "The real-prefill bridge validates CPU prompt prefill plus Vulkan follow-on T3 steps against real Chatterbox prompt state.",
            "The corrected real-prefill bridge is accurate but slower than CPU follow-on steps in the measured 32-step sample.",
            "The full loops still use synthetic hidden states; embeddings, sampler control flow, real prompt conditioning, and S3 flow acceleration are outside this measurement.",
            "The actual full-loop speed remains far from the user's target of about 2x slower than the RTX 5090 API for the 270-character case.",
        ],
    }

    json_path = OUT_DIR / "t3_vulkan_runtime_projection_2026-07-08.json"
    md_path = OUT_DIR / "t3_vulkan_runtime_projection_2026-07-08.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    md = f"""# T3 Vulkan Runtime Projection

Saved: 2026-07-08 UTC

This is a measured projection, not a completed full T3 Vulkan runtime.

## Inputs

- CPU full API 270-char case: `{cpu_api_270_s:.3f}s`
- RTX 5090 full API 270-char case: `{rtx_api_270_s:.3f}s`
- CPU T3 stage, same case: `{t3_cpu_270_s:.3f}s`
- CPU S3 flow stage, same case: `{s3_flow_270_s:.3f}s`
- CPU HiFT stage: `{hift_cpu_270_s:.3f}s`
- Vulkan HiFT stage: `{hift_vulkan_270_s:.3f}s`

## Measured Vulkan T3 Pieces

- Masked 4-layer chunk steady benchmark: `{vulkan_four_layer_ms:.3f} ms`
- Four layer fixed-slot cache updates: `{four_slot_updates_ms:.3f} ms`
- Speech head steady benchmark: `{speech_head_ms:.3f} ms`
- One chunk plus cache updates plus speech head in Python loop: `{one_chunk_loop_ms:.3f} ms/step`
- Actual full 24-layer loop, 32 steps: `{actual_full_loop_32_token_ms if actual_full_loop_32_token_ms is not None else float("nan"):.3f} ms/step`
- Actual full 24-layer loop, 128 steps: `{actual_full_loop_128_token_ms if actual_full_loop_128_token_ms is not None else float("nan"):.3f} ms/step`
- Actual full 24-layer loop, p1024 32 steps: `{actual_full_loop_p1024_32_token_ms if actual_full_loop_p1024_32_token_ms is not None else float("nan"):.3f} ms/step`
- Corrected full 24-layer loop with Vulkan final head, p1024 8 steps: `{actual_full_loop_p1024_finalhead_8_token_ms if actual_full_loop_p1024_finalhead_8_token_ms is not None else float("nan"):.3f} ms/step`
- Real-prefill bridge, CPU follow-on: `{real_prefill_cpu_ms if real_prefill_cpu_ms is not None else float("nan"):.3f} ms/step`
- Real-prefill bridge, Vulkan follow-on with logits fetch: `{real_prefill_vulkan_ms if real_prefill_vulkan_ms is not None else float("nan"):.3f} ms/step`

## Real Prompt Lengths

Turbo's default conditioning is larger than the p128 cache bucket:

- Hello initial context: `{prompt_budget["cases"][0]["initial_context_len"] if prompt_budget else "unknown"}` tokens
- Short benchmark initial context: `{prompt_budget["cases"][1]["initial_context_len"] if prompt_budget else "unknown"}` tokens
- 270-char benchmark initial context: `{prompt_budget["cases"][2]["initial_context_len"] if prompt_budget else "unknown"}` tokens
- 270-char plus 512 generated tokens needs about `{prompt_budget["cases"][2]["required_cache_for_512_generated"] if prompt_budget else "unknown"}` cache slots

## Projection

- Estimated speech-token steps for 270-char case: `{estimated_tokens:.1f}`
- Optimistic full 24-layer T3 token cost: `{optimistic_token_ms:.3f} ms`
- Python-loop full 24-layer T3 token cost: `{python_loop_token_ms:.3f} ms`
- Actual measured p128 full-loop T3 token cost: `{actual_full_loop_token_ms if actual_full_loop_token_ms is not None else float("nan"):.3f} ms`
- Actual measured p1024 real-bucket T3 token cost: `{actual_full_loop_p1024_32_token_ms if actual_full_loop_p1024_32_token_ms is not None else float("nan"):.3f} ms`
- Corrected p1024 final-head T3 token cost: `{actual_full_loop_p1024_finalhead_8_token_ms if actual_full_loop_p1024_finalhead_8_token_ms is not None else float("nan"):.3f} ms`
- Real-prefill Vulkan follow-on token cost: `{real_prefill_vulkan_ms if real_prefill_vulkan_ms is not None else float("nan"):.3f} ms`
- Optimistic T3 stage estimate: `{optimistic_t3_s:.3f}s`
- Python-loop T3 stage estimate: `{python_loop_t3_s:.3f}s`
- Actual measured p128-loop T3 stage estimate: `{actual_full_loop_t3_s if actual_full_loop_t3_s is not None else float("nan"):.3f}s`
- Actual measured p1024 real-bucket T3 stage estimate: `{p1024_real_bucket_t3_s if p1024_real_bucket_t3_s is not None else float("nan"):.3f}s`
- Corrected p1024 final-head T3 stage estimate: `{p1024_finalhead_t3_s if p1024_finalhead_t3_s is not None else float("nan"):.3f}s`
- Real-prefill Vulkan bridge T3 stage estimate: `{real_prefill_vulkan_t3_s if real_prefill_vulkan_t3_s is not None else float("nan"):.3f}s`
- Optimistic full API estimate with Vulkan T3 and Vulkan HiFT: `{optimistic_full_s:.3f}s`
- Python-loop full API estimate with Vulkan T3 and Vulkan HiFT: `{python_loop_full_s:.3f}s`
- Actual measured p128-loop full API estimate with Vulkan T3 and Vulkan HiFT: `{actual_full_loop_full_s if actual_full_loop_full_s is not None else float("nan"):.3f}s`
- Actual measured p1024 real-bucket full API estimate with Vulkan T3 and Vulkan HiFT: `{p1024_real_bucket_full_s if p1024_real_bucket_full_s is not None else float("nan"):.3f}s`
- Corrected p1024 final-head full API estimate with Vulkan T3 and Vulkan HiFT: `{p1024_finalhead_full_s if p1024_finalhead_full_s is not None else float("nan"):.3f}s`
- Real-prefill Vulkan bridge full API estimate with Vulkan HiFT: `{real_prefill_vulkan_full_s if real_prefill_vulkan_full_s is not None else float("nan"):.3f}s`
- Optimistic slowdown vs RTX 5090 API: `{optimistic_full_s / rtx_api_270_s:.2f}x`
- Python-loop slowdown vs RTX 5090 API: `{python_loop_full_s / rtx_api_270_s:.2f}x`
- Actual measured p128-loop slowdown vs RTX 5090 API: `{actual_full_loop_full_s / rtx_api_270_s if actual_full_loop_full_s is not None else float("nan"):.2f}x`
- Actual measured p1024 real-bucket slowdown vs RTX 5090 API: `{p1024_real_bucket_full_s / rtx_api_270_s if p1024_real_bucket_full_s is not None else float("nan"):.2f}x`
- Corrected p1024 final-head slowdown vs RTX 5090 API: `{p1024_finalhead_full_s / rtx_api_270_s if p1024_finalhead_full_s is not None else float("nan"):.2f}x`
- Real-prefill Vulkan bridge slowdown vs RTX 5090 API: `{real_prefill_vulkan_full_s / rtx_api_270_s if real_prefill_vulkan_full_s is not None else float("nan"):.2f}x`

## Readout

The BC-250 Vulkan path is real and now reaches full six-chunk, 24-layer
cache/logits loops. The p128 loop is useful for mechanics but too small for
real Turbo prompts. The p1024 loop matches the cache scale needed by the
270-character case plus roughly 512 generated tokens, but it is slower:
`{actual_full_loop_p1024_finalhead_8_token_ms if actual_full_loop_p1024_finalhead_8_token_ms is not None else float("nan"):.3f} ms/token` in the corrected final-head run.
The real-prefill bridge confirms correctness against actual Chatterbox prompt
state, but Vulkan follow-on steps are slower than CPU follow-on steps in that
sample.
These numbers do not support a claim that this path will reach only `2x`
slower than the RTX 5090 without a faster T3 backend or deeper implementation
work.
"""
    md_path.write_text(md)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"optimistic_full_api_270_s={optimistic_full_s:.3f}")
    print(f"python_loop_full_api_270_s={python_loop_full_s:.3f}")
    if actual_full_loop_full_s is not None:
        print(f"actual_full_loop_api_270_s={actual_full_loop_full_s:.3f}")
    if p1024_real_bucket_full_s is not None:
        print(f"p1024_real_bucket_api_270_s={p1024_real_bucket_full_s:.3f}")
    if p1024_finalhead_full_s is not None:
        print(f"p1024_finalhead_api_270_s={p1024_finalhead_full_s:.3f}")
    if real_prefill_vulkan_full_s is not None:
        print(f"real_prefill_vulkan_bridge_api_270_s={real_prefill_vulkan_full_s:.3f}")
    print(f"optimistic_slowdown_vs_5090={optimistic_full_s / rtx_api_270_s:.2f}x")
    print(f"python_loop_slowdown_vs_5090={python_loop_full_s / rtx_api_270_s:.2f}x")
    if actual_full_loop_full_s is not None:
        print(f"actual_full_loop_slowdown_vs_5090={actual_full_loop_full_s / rtx_api_270_s:.2f}x")
    if p1024_real_bucket_full_s is not None:
        print(f"p1024_real_bucket_slowdown_vs_5090={p1024_real_bucket_full_s / rtx_api_270_s:.2f}x")
    if p1024_finalhead_full_s is not None:
        print(f"p1024_finalhead_slowdown_vs_5090={p1024_finalhead_full_s / rtx_api_270_s:.2f}x")
    if real_prefill_vulkan_full_s is not None:
        print(f"real_prefill_vulkan_bridge_slowdown_vs_5090={real_prefill_vulkan_full_s / rtx_api_270_s:.2f}x")


if __name__ == "__main__":
    main()
