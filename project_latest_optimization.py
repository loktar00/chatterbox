#!/usr/bin/env python3
"""Project remaining Chatterbox optimization leverage from measured artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports/benchmarks"
OUT_JSON = BENCH / "latest_optimization_projection_2026-07-08.json"
OUT_MD = BENCH / "latest_optimization_projection_2026-07-08.md"
TARGET = 3.495 * 2.0

DEFAULT_QUALITY = BENCH / "vulkan_hybrid_api_t3_loop_info_no_watermark_default_quality_guarded_2026-07-08.json"
DEFAULT_BASELINE = BENCH / "vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json"
FAST_S3_STEP1 = BENCH / "vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08.json"
DEFAULT_T3_SUMMARY = BENCH / "vulkan_hybrid_api_t3_loop_info_no_watermark_default_quality_guarded_2026-07-08_t3_loop_summary.json"
FAST_T3_SUMMARY = BENCH / "vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08_t3_loop_summary.json"
NATIVE_SAMPLER = BENCH / "t3_native_sampler_microbench_2026-07-08.json"
NATIVE_SAMPLER_BRIDGE = BENCH / "t3_native_sampler_bridge_ctypes_2026-07-08.json"
NATIVE_SAMPLER_REAL_LOGITS = BENCH / "t3_native_sampler_real_logits_validation_2026-07-08.json"
NATIVE_SAMPLER_LIVE = BENCH / "vulkan_hybrid_api_native_sampler_no_watermark_default_quality_guarded_2026-07-08.json"
NATIVE_SAMPLER_PADDED_CAPPED = BENCH / "vulkan_hybrid_api_native_sampler_padded_s3_capped_guarded_2026-07-08.json"
NATIVE_SAMPLER_REQUEST_SEEDED = (
    BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_padded_s3_capped_guarded_2026-07-08.json"
)
NATIVE_SAMPLER_S3_STEP1 = (
    BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_no_watermark_guarded_2026-07-08.json"
)
NATIVE_SAMPLER_S3_STEP1_FUSED = (
    BENCH
    / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json"
)
NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_padded_vs_default_no_watermark_audio_sanity_2026-07-08.json"
)
NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_vs_default_no_watermark_audio_sanity_2026-07-08.json"
)
NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_vs_s3_step1_audio_sanity_2026-07-08.json"
)
NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_basic_audio_sanity_2026-07-09.json"
)
NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_prior_fast_audio_sanity_2026-07-09.json"
)
NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_default_no_watermark_audio_sanity_2026-07-09.json"
)
NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_req1_vs_req0_audio_sanity_2026-07-09.json"
)
S3_BOTTLENECK_AUDIT = BENCH / "s3_bottleneck_audit_2026-07-08.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def last_request(path: Path) -> dict[str, Any]:
    data = load(path)
    requests = data.get("requests") or []
    if not requests:
        return {}
    return (((requests[-1].get("debug") or {}).get("body") or {}).get("last_request") or {})


def last_wall(path: Path) -> float:
    data = load(path)
    requests = data.get("requests") or []
    return float(requests[-1]["wall_seconds"])


def warm_summary(path: Path) -> dict[str, Any]:
    data = load(path)
    requests = data.get("requests") or []
    return requests[-1] if requests else {}


def request_last_timing(row: dict[str, Any]) -> dict[str, Any]:
    return (((row.get("debug") or {}).get("body") or {}).get("last_request") or {})


def scenario(name: str, base: float, savings: float, quality: str, note: str) -> dict[str, Any]:
    projected = base - savings
    return {
        "name": name,
        "quality": quality,
        "base_seconds": base,
        "savings_seconds": savings,
        "projected_seconds": projected,
        "gap_to_target_seconds": projected - TARGET,
        "ratio_to_5090": projected / 3.495,
        "note": note,
    }


def fmt(value: Any) -> str:
    return f"{value:.3f}s" if isinstance(value, (int, float)) else "n/a"


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Latest Optimization Projection - 2026-07-08",
        "",
        f"- RTX 5090 reference: `{fmt(report['rtx_5090_reference_seconds'])}`.",
        f"- 2x target: `{fmt(report['target_seconds'])}`.",
        f"- Default-quality current wall: `{fmt(report['current']['default_quality_wall_seconds'])}`.",
        f"- Fast listen-before-default current wall: `{fmt(report['current']['fast_s3_step1_wall_seconds'])}`.",
        f"- Fast fused listen-before-default current wall: `{fmt(report['current'].get('fast_s3_step1_fused_wall_seconds'))}`.",
        "",
        "## Current Warm Stage Times",
        "",
        "| Path | Wall | T3 | S3 | HiFT decode | Watermark |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["stage_rows"]:
        lines.append(
            "| "
            f"{row['path']} | `{fmt(row['wall_seconds'])}` | `{fmt(row['t3_seconds'])}` | "
            f"`{fmt(row['s3_flow_seconds'])}` | `{fmt(row['hift_decode_seconds'])}` | "
            f"`{fmt(row['watermark_seconds'])}` |"
        )
    lines.extend(
        [
            "",
            "## T3 Loop Leverage",
            "",
            "| Path | T3 | Loop | ggml wall | sampling | logits->Torch |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["t3_rows"]:
        lines.append(
            "| "
            f"{row['path']} | `{fmt(row['t3_seconds'])}` | `{fmt(row['loop_total_seconds'])}` | "
            f"`{fmt(row['ggml_wall_seconds'])}` | `{fmt(row['sampling_seconds'])}` | "
            f"`{fmt(row['logits_to_torch_seconds'])}` |"
        )
    native = report.get("native_sampler") or {}
    if native:
        lines.extend(
            [
                "",
                "## Native Sampler Probe",
                "",
                f"- Realistic seen-length speedup: `{native.get('real_seen_len_speedup'):.2f}x`.",
                f"- Native per-call sampler time: `{native.get('native_per_trial_ms'):.4f}ms`.",
                f"- Python per-trial sampler time: `{native.get('python_current_per_trial_ms'):.4f}ms`.",
                f"- Real-logits validation pass: `{native.get('real_logits_validation_pass')}`.",
                f"- Real-logits max abs probability diff: `{native.get('real_logits_max_abs_prob_diff')}`.",
                f"- Evidence: `{native.get('source')}`.",
            ]
        )
        live = native.get("live_default_quality") or {}
        if live:
            lines.extend(
                [
                    f"- Live default-quality native sampler status: `{live.get('status')}`.",
                    f"- Live default-quality native sampler wall: `{fmt(live.get('wall_seconds'))}`.",
                    f"- Live default-quality native sampler S3: `{fmt(live.get('s3_flow_seconds'))}`.",
                    f"- Live default-quality native sampler selected S3 bucket: `{live.get('selected_s3_bucket')}`.",
                ]
            )
        padded = native.get("live_padded_capped") or {}
        if padded:
            lines.extend(
                [
                    f"- Padded/capped native sampler status: `{padded.get('status')}`.",
                    f"- Padded/capped first successful wall: `{fmt(padded.get('first_success_wall_seconds'))}`.",
                    f"- Padded/capped selected S3 bucket: `{padded.get('selected_s3_bucket')}`.",
                ]
            )
        seeded = native.get("live_request_seeded_padded_capped") or {}
        if seeded:
            lines.extend(
                [
                    f"- Request-seeded padded/capped native sampler status: `{seeded.get('status')}`.",
                    f"- Request-seeded padded/capped warm wall: `{fmt(seeded.get('wall_seconds'))}`.",
                    f"- Request-seeded padded/capped audio sanity pass: `{seeded.get('audio_sanity_pass')}`.",
                    f"- Request-seeded padded/capped correlation vs baseline: `{seeded.get('audio_correlation_vs_baseline')}`.",
                ]
            )
        fast_seeded = native.get("live_request_seeded_s3_step1") or {}
        if fast_seeded:
            lines.extend(
                [
                    f"- Request-seeded S3-step1 native sampler status: `{fast_seeded.get('status')}`.",
                    f"- Request-seeded S3-step1 warm wall: `{fmt(fast_seeded.get('wall_seconds'))}`.",
                    f"- Request-seeded S3-step1 gap to 2x target: `{fmt(fast_seeded.get('gap_to_target_seconds'))}`.",
                    f"- Request-seeded S3-step1 audio sanity pass: `{fast_seeded.get('audio_sanity_pass')}`.",
                    f"- Request-seeded S3-step1 correlation vs default: `{fast_seeded.get('audio_correlation_vs_default')}`.",
                ]
            )
        fast_fused = native.get("live_request_seeded_s3_step1_fused") or {}
        if fast_fused:
            lines.extend(
                [
                    f"- Request-seeded S3-step1 fused status: `{fast_fused.get('status')}`.",
                    f"- Request-seeded S3-step1 fused warm wall: `{fmt(fast_fused.get('wall_seconds'))}`.",
                    f"- Request-seeded S3-step1 fused gap to 2x target: `{fmt(fast_fused.get('gap_to_target_seconds'))}`.",
                    f"- Request-seeded S3-step1 fused S3: `{fmt(fast_fused.get('s3_flow_seconds'))}`.",
                    f"- Request-seeded S3-step1 fused estimator chain type: `{fast_fused.get('estimator_chain_types')}`.",
                    f"- Request-seeded S3-step1 fused audio sanity pass: `{fast_fused.get('audio_sanity_pass')}`.",
                    f"- Request-seeded S3-step1 fused correlation vs prior fast: `{fast_fused.get('audio_correlation_vs_prior_fast')}`.",
                    f"- Request-seeded S3-step1 fused correlation vs default: `{fast_fused.get('audio_correlation_vs_default')}`.",
                ]
            )
    s3 = report.get("s3_bottleneck") or {}
    if s3:
        lines.extend(
            [
                "",
                "## S3 Bottleneck Audit",
                "",
                f"- S3 flow: `{fmt(s3.get('s3_flow_seconds'))}`.",
                f"- Encoder Vulkan chain: `{fmt(s3.get('encoder_vulkan_seconds'))}`.",
                f"- Estimator Vulkan chains: `{fmt(s3.get('estimator_vulkan_seconds'))}`.",
                f"- Tensor conversion share: `{s3.get('to_numpy_share_percent'):.4f}%`.",
                f"- Observed padding delta: `{fmt(s3.get('padding_delta_seconds'))}`.",
                f"- Legacy t1210 fused mid-block projected savings: `{fmt(s3.get('legacy_fused_midblock_fetch_savings_seconds'))}`.",
                f"- Active t1222 fused mid-block recommended savings: `{fmt(s3.get('active_fused_midblock_recommended_savings_seconds'))}`.",
                f"- Active t1222 split8 compile-flag candidate savings: `{fmt(s3.get('active_fused_midblock_split8_candidate_savings_seconds'))}`.",
                f"- Active t1222 split8 tested midblocks: `{s3.get('active_fused_midblock_split8_tested_midblocks')}` of `{s3.get('active_fused_midblock_split8_total_midblocks')}`.",
                f"- Active t1222 fused estimator-chain confirmed candidate savings: `{fmt(s3.get('active_fused_estimator_chain_candidate_savings_seconds'))}`.",
                f"- Active t1222 fused API best total: `{fmt(s3.get('active_fused_api_best_total_seconds'))}`.",
                f"- Evidence: `{s3.get('source')}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scenarios",
            "",
            "| Scenario | Quality Gate | Savings | Projected | Gap To Target | Note |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in report["scenarios"]:
        lines.append(
            "| "
            f"{item['name']} | {item['quality']} | `{fmt(item['savings_seconds'])}` | "
            f"`{fmt(item['projected_seconds'])}` | `{fmt(item['gap_to_target_seconds'])}` | "
            f"{item['note']} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Default-quality work still needs about `1.15s`; the measured low-risk software levers do not add up to that.",
            "- The one-step S3 no-watermark candidate is only about `0.123s` over the 2x target, but it remains listen-before-default because waveform correlation against two-step was `0.572`.",
            "- Combining request-seeded native sampling with one-step fused S3 crosses the target at `6.903s`, but it is still listen-before-default because the fast recipe is waveform-different from default and the fused variant still needs listening validation.",
            "- Request-seeded native sampling plus padded S3 is now stable for the guarded two-request trial and saves about `0.128s`, but audio is waveform-different from the baseline, so it remains opt-in/listen-before-default.",
            "- S3 telemetry shows estimator execution dominates; tensor conversion and exact-bucket padding are too small to matter much. The active t1222 split8 fused-midblock path is validated across all midblocks, confirmed in a fixed-shape fused estimator-chain benchmark, and integrated behind an opt-in API flag, but production credit stays at zero until listening/default validation.",
            "- A stable higher-CU board remains the cleanest path to hit the target without changing model quality.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    default_lr = last_request(DEFAULT_QUALITY)
    fast_lr = last_request(FAST_S3_STEP1)
    default_wall = last_wall(DEFAULT_BASELINE)
    default_instrumented_wall = last_wall(DEFAULT_QUALITY)
    fast_wall = last_wall(FAST_S3_STEP1)
    default_t3 = warm_summary(DEFAULT_T3_SUMMARY)
    fast_t3 = warm_summary(FAST_T3_SUMMARY)

    default_timings = default_t3.get("timings") or {}
    fast_timings = fast_t3.get("timings") or {}
    default_sampling = float(default_timings.get("step_sampling_seconds") or 0.0)
    default_host_loop = sum(
        float(default_timings.get(key) or 0.0)
        for key in ("step_hidden_seconds", "step_logits_to_torch_seconds", "step_sampling_seconds")
    )
    fast_sampling = float(fast_timings.get("step_sampling_seconds") or 0.0)
    fast_host_loop = sum(
        float(fast_timings.get(key) or 0.0)
        for key in ("step_hidden_seconds", "step_logits_to_torch_seconds", "step_sampling_seconds")
    )
    native_sampler_source = NATIVE_SAMPLER_BRIDGE if NATIVE_SAMPLER_BRIDGE.exists() else NATIVE_SAMPLER
    native_sampler = load(native_sampler_source) if native_sampler_source.exists() else {}
    native_sampler_real_logits = load(NATIVE_SAMPLER_REAL_LOGITS) if NATIVE_SAMPLER_REAL_LOGITS.exists() else {}
    native_real_logits_comparisons = [
        step.get("comparison") or {}
        for step in native_sampler_real_logits.get("steps", [])
    ]
    native_real_logits_max_abs = max(
        (item.get("max_abs_prob_diff") or 0.0 for item in native_real_logits_comparisons),
        default=None,
    )
    native_real_case = next((case for case in native_sampler.get("cases", []) if case.get("seen_len") == 358), {})
    native_speedup = float(
        native_real_case.get("native_ctypes_speedup_vs_python_current")
        or native_real_case.get("native_speedup_vs_python_current")
        or 0.0
    )
    native_default_sampling_savings = (
        default_sampling * (1.0 - 1.0 / native_speedup) if native_speedup > 0.0 else 0.0
    )
    native_fast_sampling_savings = fast_sampling * (1.0 - 1.0 / native_speedup) if native_speedup > 0.0 else 0.0
    native_live = load(NATIVE_SAMPLER_LIVE) if NATIVE_SAMPLER_LIVE.exists() else {}
    native_live_requests = native_live.get("requests") or []
    native_live_warm = native_live_requests[-1] if native_live.get("status") == "ok" and native_live_requests else {}
    native_live_lr = request_last_timing(native_live_warm) if native_live_warm else {}
    native_live_loop = native_live_lr.get("t3_loop_info") or {}
    native_live_s3 = native_live_lr.get("s3_bucket_inference") or {}
    native_live_wall = native_live_warm.get("wall_seconds")
    padded_live = load(NATIVE_SAMPLER_PADDED_CAPPED) if NATIVE_SAMPLER_PADDED_CAPPED.exists() else {}
    padded_successes = [row for row in padded_live.get("requests", []) if row.get("status") == 200]
    padded_first = padded_successes[0] if padded_successes else {}
    padded_first_lr = request_last_timing(padded_first) if padded_first else {}
    padded_first_s3 = padded_first_lr.get("s3_bucket_inference") or {}
    padded_first_wall = padded_first.get("wall_seconds")
    request_seeded_live = load(NATIVE_SAMPLER_REQUEST_SEEDED) if NATIVE_SAMPLER_REQUEST_SEEDED.exists() else {}
    request_seeded_requests = request_seeded_live.get("requests") or []
    request_seeded_warm = request_seeded_requests[-1] if request_seeded_live.get("status") == "ok" and request_seeded_requests else {}
    request_seeded_lr = request_last_timing(request_seeded_warm) if request_seeded_warm else {}
    request_seeded_loop = request_seeded_lr.get("t3_loop_info") or {}
    request_seeded_s3 = request_seeded_lr.get("s3_bucket_inference") or {}
    request_seeded_audio = (
        load(NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO) if NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO.exists() else {}
    )
    request_seeded_audio_comparison = request_seeded_audio.get("comparison_to_reference") or {}
    request_seeded_audio_summary = request_seeded_audio.get("audio") or {}
    request_seeded_wall = request_seeded_warm.get("wall_seconds")
    fast_seeded_live = load(NATIVE_SAMPLER_S3_STEP1) if NATIVE_SAMPLER_S3_STEP1.exists() else {}
    fast_seeded_requests = fast_seeded_live.get("requests") or []
    fast_seeded_warm = fast_seeded_requests[-1] if fast_seeded_live.get("status") == "ok" and fast_seeded_requests else {}
    fast_seeded_lr = request_last_timing(fast_seeded_warm) if fast_seeded_warm else {}
    fast_seeded_loop = fast_seeded_lr.get("t3_loop_info") or {}
    fast_seeded_s3 = fast_seeded_lr.get("s3_bucket_inference") or {}
    fast_seeded_wall = fast_seeded_warm.get("wall_seconds")
    fast_seeded_default_audio = (
        load(NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT) if NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT.exists() else {}
    )
    fast_seeded_fast_audio = (
        load(NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST) if NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST.exists() else {}
    )
    fast_seeded_default_comparison = fast_seeded_default_audio.get("comparison_to_reference") or {}
    fast_seeded_fast_comparison = fast_seeded_fast_audio.get("comparison_to_reference") or {}
    fast_seeded_audio_summary = fast_seeded_default_audio.get("audio") or {}
    fast_fused_live = load(NATIVE_SAMPLER_S3_STEP1_FUSED) if NATIVE_SAMPLER_S3_STEP1_FUSED.exists() else {}
    fast_fused_requests = fast_fused_live.get("requests") or []
    fast_fused_warm = fast_fused_requests[-1] if fast_fused_live.get("status") == "ok" and fast_fused_requests else {}
    fast_fused_lr = request_last_timing(fast_fused_warm) if fast_fused_warm else {}
    fast_fused_loop = fast_fused_lr.get("t3_loop_info") or {}
    fast_fused_s3 = fast_fused_lr.get("s3_bucket_inference") or {}
    fast_fused_estimator = fast_fused_lr.get("s3_estimator_calls") or {}
    fast_fused_estimator_records = fast_fused_estimator.get("records") or []
    fast_fused_wall = fast_fused_warm.get("wall_seconds")
    fast_fused_audio_basic = (
        load(NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC)
        if NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC.exists()
        else {}
    )
    fast_fused_audio_prior_fast = (
        load(NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST)
        if NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST.exists()
        else {}
    )
    fast_fused_audio_default = (
        load(NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT)
        if NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT.exists()
        else {}
    )
    fast_fused_audio_repeat = (
        load(NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT)
        if NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT.exists()
        else {}
    )
    fast_fused_audio_summary = fast_fused_audio_basic.get("audio") or {}
    fast_fused_prior_comparison = fast_fused_audio_prior_fast.get("comparison_to_reference") or {}
    fast_fused_default_comparison = fast_fused_audio_default.get("comparison_to_reference") or {}
    fast_fused_repeat_comparison = fast_fused_audio_repeat.get("comparison_to_reference") or {}
    s3_bottleneck = load(S3_BOTTLENECK_AUDIT) if S3_BOTTLENECK_AUDIT.exists() else {}
    s3_api = s3_bottleneck.get("api_debug") or {}
    s3_fused = s3_bottleneck.get("fused_midblock") or {}
    s3_active_fused = s3_bottleneck.get("active_shape_fused_midblock") or {}
    s3_active_flags = s3_bottleneck.get("active_shape_fused_midblock_flag_matrix") or {}
    s3_fused_chain = s3_bottleneck.get("active_shape_fused_estimator_chain") or {}
    s3_fused_api = s3_bottleneck.get("active_shape_fused_api_benchmark") or {}
    s3_padding = s3_bottleneck.get("bucket_padding") or {}

    report = {
        "rtx_5090_reference_seconds": 3.495,
        "target_seconds": TARGET,
        "current": {
            "default_quality_wall_seconds": default_wall,
            "default_quality_instrumented_wall_seconds": default_instrumented_wall,
            "fast_s3_step1_wall_seconds": fast_wall,
            "fast_s3_step1_fused_wall_seconds": fast_fused_wall,
        },
        "stage_rows": [
            {
                "path": "default quality, no watermark",
                "wall_seconds": default_wall,
                "t3_seconds": default_lr.get("t3_seconds"),
                "s3_flow_seconds": default_lr.get("s3_flow_seconds"),
                "hift_decode_seconds": default_lr.get("hift_decode_seconds"),
                "watermark_seconds": default_lr.get("watermark_seconds"),
            },
            {
                "path": "fast S3 step1, no watermark",
                "wall_seconds": fast_wall,
                "t3_seconds": fast_lr.get("t3_seconds"),
                "s3_flow_seconds": fast_lr.get("s3_flow_seconds"),
                "hift_decode_seconds": fast_lr.get("hift_decode_seconds"),
                "watermark_seconds": fast_lr.get("watermark_seconds"),
            },
            {
                "path": "fast S3 step1 fused midblocks, no watermark",
                "wall_seconds": fast_fused_wall,
                "t3_seconds": fast_fused_lr.get("t3_seconds"),
                "s3_flow_seconds": fast_fused_lr.get("s3_flow_seconds"),
                "hift_decode_seconds": fast_fused_lr.get("hift_decode_seconds"),
                "watermark_seconds": fast_fused_lr.get("watermark_seconds"),
            },
        ],
        "t3_rows": [
            {
                "path": "default quality, no watermark",
                "t3_seconds": default_t3.get("t3_seconds"),
                "loop_total_seconds": default_t3.get("loop_total_seconds"),
                "ggml_wall_seconds": default_timings.get("step_ggml_wall_seconds"),
                "sampling_seconds": default_timings.get("step_sampling_seconds"),
                "logits_to_torch_seconds": default_timings.get("step_logits_to_torch_seconds"),
            },
            {
                "path": "fast S3 step1, no watermark",
                "t3_seconds": fast_t3.get("t3_seconds"),
                "loop_total_seconds": fast_t3.get("loop_total_seconds"),
                "ggml_wall_seconds": fast_timings.get("step_ggml_wall_seconds"),
                "sampling_seconds": fast_timings.get("step_sampling_seconds"),
                "logits_to_torch_seconds": fast_timings.get("step_logits_to_torch_seconds"),
            },
        ],
        "scenarios": [
            scenario(
                "Default quality: native sampler bridge projected",
                default_wall,
                native_default_sampling_savings,
                "default-quality",
                "Uses measured ctypes native sampler bridge speedup; still does not close the default-quality gap.",
            ),
            scenario(
                "Default quality: native sampler live benchmark",
                default_wall,
                default_wall - native_live_wall if isinstance(native_live_wall, (int, float)) else 0.0,
                "rejected",
                "Actual opt-in run missed exact S3 buckets and fell back to slow CPU S3.",
            ),
            scenario(
                "Default quality: native sampler padded/capped first request",
                default_wall,
                default_wall - padded_first_wall if isinstance(padded_first_wall, (int, float)) else 0.0,
                "rejected",
                "First padded request stayed on Vulkan S3 but did not beat baseline; second request failed.",
            ),
            scenario(
                "Request-seeded native sampler padded/capped live benchmark",
                default_wall,
                default_wall - request_seeded_wall if isinstance(request_seeded_wall, (int, float)) else 0.0,
                "listen-before-default",
                "Stable two-request guarded run after request-scoped native sampler seed; waveform-different from baseline.",
            ),
            scenario(
                "Request-seeded native sampler + S3 step1 live benchmark",
                fast_wall,
                fast_wall - fast_seeded_wall if isinstance(fast_seeded_wall, (int, float)) else 0.0,
                "listen-before-default",
                "Crosses 2x target in guarded run; waveform-different and opt-in only.",
            ),
            scenario(
                "Request-seeded native sampler + S3 step1 + fused midblocks live benchmark",
                fast_seeded_wall if isinstance(fast_seeded_wall, (int, float)) else fast_wall,
                (
                    fast_seeded_wall - fast_fused_wall
                    if isinstance(fast_seeded_wall, (int, float)) and isinstance(fast_fused_wall, (int, float))
                    else 0.0
                ),
                "listen-before-default",
                "Fastest guarded live API run so far; fused split8 S3 estimator path is opt-in and needs listening validation.",
            ),
            scenario(
                "Default quality: remove Python sampling only",
                default_wall,
                default_sampling,
                "default-quality",
                "Still far from target; sampling is not the default-quality bottleneck.",
            ),
            scenario(
                "Default quality: remove host loop prep/logits/sampling",
                default_wall,
                default_host_loop,
                "default-quality",
                "Still far from target; ggml per-token execution dominates.",
            ),
            scenario(
                "Default quality: active-shape S3 fused-midblock projection",
                default_wall,
                float(s3_active_fused.get("recommended_full_s3_savings_seconds") or 0.0),
                "default-quality",
                "Active t1222 one-off fused-midblock probe validated but production savings are not credited yet.",
            ),
            scenario(
                "Default quality: active-shape split8 fused-midblock candidate",
                default_wall,
                float(s3_active_flags.get("candidate_full_s3_fetch_output_savings_seconds") or 0.0),
                "experimental-candidate",
                "Validated isolated split8 compile-flag result across the active-shape midblocks tested so far; needs full-midblock integration before production credit.",
            ),
            scenario(
                "Default quality: fused estimator-chain candidate",
                default_wall,
                float(s3_fused_chain.get("fetch_output_seconds_for_two_s3_estimator_calls") or 0.0),
                "experimental-candidate",
                "Confirmed in a fixed-shape full estimator-chain benchmark; needs opt-in API runtime integration before production credit.",
            ),
            scenario(
                "Fast S3 step1: current measured",
                fast_wall,
                0.0,
                "listen-before-default",
                "Only 0.123s over target, but waveform-different.",
            ),
            scenario(
                "Fast S3 step1: native sampler bridge projected",
                fast_wall,
                native_fast_sampling_savings,
                "listen-before-default",
                "Uses measured ctypes native sampler bridge speedup; projects under target if one-step S3 quality is accepted.",
            ),
            scenario(
                "Fast S3 step1: remove Python sampling only",
                fast_wall,
                fast_sampling,
                "listen-before-default",
                "Would clear target if one-step S3 is accepted and native sampling is implemented.",
            ),
            scenario(
                "Fast S3 step1: remove host loop prep/logits/sampling",
                fast_wall,
                fast_host_loop,
                "listen-before-default",
                "Also clears target, but most of this win is sampling.",
            ),
        ],
        "native_sampler": {
            "source": native_sampler_source.as_posix(),
            "real_seen_len_speedup": native_speedup,
            "native_per_trial_ms": native_real_case.get("native_ctypes_per_call_ms")
            or native_real_case.get("native_per_trial_ms"),
            "python_current_per_trial_ms": native_real_case.get("python_current_per_trial_ms"),
            "default_sampling_savings_seconds": native_default_sampling_savings,
            "fast_sampling_savings_seconds": native_fast_sampling_savings,
            "real_logits_validation_source": NATIVE_SAMPLER_REAL_LOGITS.as_posix(),
            "real_logits_validation_pass": native_sampler_real_logits.get("overall_pass"),
            "real_logits_max_abs_prob_diff": native_real_logits_max_abs,
            "live_default_quality": {
                "source": NATIVE_SAMPLER_LIVE.as_posix(),
                "status": native_live.get("status"),
                "wall_seconds": native_live_wall,
                "t3_seconds": native_live_lr.get("t3_seconds"),
                "s3_flow_seconds": native_live_lr.get("s3_flow_seconds"),
                "raw_t3_tokens": native_live_lr.get("raw_t3_tokens"),
                "loop_iterations": native_live_loop.get("loop_iterations"),
                "step_sampling_seconds": (native_live_loop.get("timings") or {}).get("step_sampling_seconds"),
                "step_native_sampler_seconds": (
                    native_live_loop.get("timings") or {}
                ).get("step_native_sampler_seconds"),
                "selected_s3_bucket": native_live_s3.get("selected_bucket"),
            },
            "live_padded_capped": {
                "source": NATIVE_SAMPLER_PADDED_CAPPED.as_posix(),
                "status": padded_live.get("status"),
                "first_success_wall_seconds": padded_first_wall,
                "first_success_t3_seconds": padded_first_lr.get("t3_seconds"),
                "first_success_s3_seconds": padded_first_lr.get("s3_flow_seconds"),
                "selected_s3_bucket": padded_first_s3.get("selected_bucket"),
            },
            "live_request_seeded_padded_capped": {
                "source": NATIVE_SAMPLER_REQUEST_SEEDED.as_posix(),
                "status": request_seeded_live.get("status"),
                "wall_seconds": request_seeded_wall,
                "t3_seconds": request_seeded_lr.get("t3_seconds"),
                "s3_flow_seconds": request_seeded_lr.get("s3_flow_seconds"),
                "raw_t3_tokens": request_seeded_lr.get("raw_t3_tokens"),
                "loop_iterations": request_seeded_loop.get("loop_iterations"),
                "native_sampler_seed": request_seeded_loop.get("native_sampler_seed"),
                "step_sampling_seconds": (request_seeded_loop.get("timings") or {}).get("step_sampling_seconds"),
                "step_native_sampler_seconds": (
                    request_seeded_loop.get("timings") or {}
                ).get("step_native_sampler_seconds"),
                "selected_s3_bucket": request_seeded_s3.get("selected_bucket"),
                "audio_sanity_source": NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO.as_posix(),
                "audio_sanity_pass": request_seeded_audio_summary.get("passed_basic_sanity"),
                "audio_correlation_vs_baseline": request_seeded_audio_comparison.get("correlation"),
                "audio_rms_diff_vs_baseline": request_seeded_audio_comparison.get("rms_diff"),
            },
            "live_request_seeded_s3_step1": {
                "source": NATIVE_SAMPLER_S3_STEP1.as_posix(),
                "status": fast_seeded_live.get("status"),
                "wall_seconds": fast_seeded_wall,
                "target_seconds": TARGET,
                "gap_to_target_seconds": (
                    fast_seeded_wall - TARGET if isinstance(fast_seeded_wall, (int, float)) else None
                ),
                "ratio_to_5090": (
                    fast_seeded_wall / 3.495 if isinstance(fast_seeded_wall, (int, float)) else None
                ),
                "t3_seconds": fast_seeded_lr.get("t3_seconds"),
                "s3_flow_seconds": fast_seeded_lr.get("s3_flow_seconds"),
                "s3_timesteps": fast_seeded_lr.get("s3_timesteps"),
                "raw_t3_tokens": fast_seeded_lr.get("raw_t3_tokens"),
                "loop_iterations": fast_seeded_loop.get("loop_iterations"),
                "native_sampler_seed": fast_seeded_loop.get("native_sampler_seed"),
                "step_sampling_seconds": (fast_seeded_loop.get("timings") or {}).get("step_sampling_seconds"),
                "step_native_sampler_seconds": (
                    fast_seeded_loop.get("timings") or {}
                ).get("step_native_sampler_seconds"),
                "selected_s3_bucket": fast_seeded_s3.get("selected_bucket"),
                "audio_sanity_source_vs_default": NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT.as_posix(),
                "audio_sanity_source_vs_fast": NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST.as_posix(),
                "audio_sanity_pass": fast_seeded_audio_summary.get("passed_basic_sanity"),
                "audio_correlation_vs_default": fast_seeded_default_comparison.get("correlation"),
                "audio_rms_diff_vs_default": fast_seeded_default_comparison.get("rms_diff"),
                "audio_correlation_vs_fast_s3_step1": fast_seeded_fast_comparison.get("correlation"),
                "audio_rms_diff_vs_fast_s3_step1": fast_seeded_fast_comparison.get("rms_diff"),
            },
            "live_request_seeded_s3_step1_fused": {
                "source": NATIVE_SAMPLER_S3_STEP1_FUSED.as_posix(),
                "status": fast_fused_live.get("status"),
                "wall_seconds": fast_fused_wall,
                "target_seconds": TARGET,
                "gap_to_target_seconds": (
                    fast_fused_wall - TARGET if isinstance(fast_fused_wall, (int, float)) else None
                ),
                "ratio_to_5090": (
                    fast_fused_wall / 3.495 if isinstance(fast_fused_wall, (int, float)) else None
                ),
                "t3_seconds": fast_fused_lr.get("t3_seconds"),
                "s3_flow_seconds": fast_fused_lr.get("s3_flow_seconds"),
                "s3_timesteps": fast_fused_lr.get("s3_timesteps"),
                "raw_t3_tokens": fast_fused_lr.get("raw_t3_tokens"),
                "loop_iterations": fast_fused_loop.get("loop_iterations"),
                "native_sampler_seed": fast_fused_loop.get("native_sampler_seed"),
                "selected_s3_bucket": fast_fused_s3.get("selected_bucket"),
                "estimator_vulkan_calls": fast_fused_estimator.get("vulkan"),
                "estimator_chain_fetch_seconds": fast_fused_estimator.get("vulkan_chain_fetch_seconds"),
                "estimator_chain_types": [record.get("chain_type") for record in fast_fused_estimator_records],
                "audio_sanity_source_basic": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC.as_posix(),
                "audio_sanity_source_vs_prior_fast": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST.as_posix(),
                "audio_sanity_source_vs_default": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT.as_posix(),
                "audio_sanity_source_repeat": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT.as_posix(),
                "audio_sanity_pass": fast_fused_audio_summary.get("passed_basic_sanity"),
                "audio_peak_abs": fast_fused_audio_summary.get("peak_abs"),
                "audio_rms_dbfs": fast_fused_audio_summary.get("rms_dbfs"),
                "audio_clipped_fraction_abs_ge_0_999": fast_fused_audio_summary.get("clipped_fraction_abs_ge_0_999"),
                "audio_correlation_vs_prior_fast": fast_fused_prior_comparison.get("correlation"),
                "audio_rms_diff_vs_prior_fast": fast_fused_prior_comparison.get("rms_diff"),
                "audio_correlation_vs_default": fast_fused_default_comparison.get("correlation"),
                "audio_rms_diff_vs_default": fast_fused_default_comparison.get("rms_diff"),
                "audio_correlation_req1_vs_req0": fast_fused_repeat_comparison.get("correlation"),
                "audio_rms_diff_req1_vs_req0": fast_fused_repeat_comparison.get("rms_diff"),
            },
        },
        "s3_bottleneck": {
            "source": S3_BOTTLENECK_AUDIT.as_posix(),
            "s3_flow_seconds": s3_api.get("s3_flow_seconds"),
            "encoder_vulkan_seconds": (s3_api.get("encoder") or {}).get("vulkan_chain_fetch_seconds"),
            "estimator_vulkan_seconds": (s3_api.get("estimator") or {}).get("vulkan_chain_fetch_seconds"),
            "to_numpy_share_percent": (
                (s3_api.get("to_numpy_share_of_s3") or 0.0) * 100.0
                if s3_api
                else None
            ),
            "padding_delta_seconds": s3_padding.get("padding_delta_seconds"),
            "legacy_fused_midblock_no_fetch_savings_seconds": s3_fused.get("projected_full_s3_no_fetch_savings_seconds"),
            "legacy_fused_midblock_fetch_savings_seconds": s3_fused.get("projected_full_s3_fetch_savings_seconds"),
            "active_fused_midblock_no_fetch_savings_seconds": s3_active_fused.get("projected_full_s3_no_fetch_savings_seconds"),
            "active_fused_midblock_fetch_savings_seconds": s3_active_fused.get("projected_full_s3_fetch_savings_seconds"),
            "active_fused_midblock_recommended_savings_seconds": s3_active_fused.get("recommended_full_s3_savings_seconds"),
            "active_fused_midblock_recommendation": s3_active_fused.get("recommendation"),
            "active_fused_midblock_split8_candidate_savings_seconds": s3_active_flags.get("candidate_full_s3_fetch_output_savings_seconds"),
            "active_fused_midblock_split8_candidate_no_fetch_savings_seconds": s3_active_flags.get("candidate_full_s3_no_fetch_savings_seconds"),
            "active_fused_midblock_split8_production_credit_seconds": s3_active_flags.get("production_credit_seconds"),
            "active_fused_midblock_split8_tested_midblocks": s3_active_flags.get("tested_midblock_count"),
            "active_fused_midblock_split8_total_midblocks": s3_active_flags.get("total_midblocks"),
            "active_fused_midblock_split8_recommendation": s3_active_flags.get("recommendation"),
            "active_fused_estimator_chain_candidate_savings_seconds": s3_fused_chain.get("fetch_output_seconds_for_two_s3_estimator_calls"),
            "active_fused_estimator_chain_candidate_per_estimator_seconds": s3_fused_chain.get("fetch_output_seconds_per_estimator_call"),
            "active_fused_estimator_chain_validation_allclose_1e_4": s3_fused_chain.get("validation_allclose_1e_4"),
            "active_fused_estimator_chain_validation_max_abs_error": s3_fused_chain.get("validation_max_abs_error"),
            "active_fused_estimator_chain_production_credit_seconds": s3_fused_chain.get("production_credit_seconds"),
            "active_fused_estimator_chain_recommendation": s3_fused_chain.get("recommendation"),
            "active_fused_api_best_total_seconds": s3_fused_api.get("best_total_seconds"),
            "active_fused_api_best_wall_seconds": s3_fused_api.get("best_wall_seconds"),
            "active_fused_api_best_s3_flow_seconds": s3_fused_api.get("best_s3_flow_seconds"),
            "active_fused_api_estimator_chain_types": s3_fused_api.get("best_estimator_chain_types"),
            "active_fused_api_production_credit_seconds": s3_fused_api.get("production_credit_seconds"),
            "active_fused_api_recommendation": s3_fused_api.get("recommendation"),
            "decision": s3_bottleneck.get("decision"),
        },
        "sources": {
            "default_quality": DEFAULT_QUALITY.as_posix(),
            "default_baseline": DEFAULT_BASELINE.as_posix(),
            "fast_s3_step1": FAST_S3_STEP1.as_posix(),
            "default_t3_summary": DEFAULT_T3_SUMMARY.as_posix(),
            "fast_t3_summary": FAST_T3_SUMMARY.as_posix(),
            "native_sampler": NATIVE_SAMPLER.as_posix(),
            "native_sampler_bridge": NATIVE_SAMPLER_BRIDGE.as_posix(),
            "native_sampler_real_logits": NATIVE_SAMPLER_REAL_LOGITS.as_posix(),
            "native_sampler_live": NATIVE_SAMPLER_LIVE.as_posix(),
            "native_sampler_padded_capped": NATIVE_SAMPLER_PADDED_CAPPED.as_posix(),
            "native_sampler_request_seeded": NATIVE_SAMPLER_REQUEST_SEEDED.as_posix(),
            "native_sampler_request_seeded_audio": NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO.as_posix(),
            "native_sampler_s3_step1": NATIVE_SAMPLER_S3_STEP1.as_posix(),
            "native_sampler_s3_step1_fused": NATIVE_SAMPLER_S3_STEP1_FUSED.as_posix(),
            "native_sampler_s3_step1_fused_audio_basic": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC.as_posix(),
            "native_sampler_s3_step1_fused_audio_prior_fast": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST.as_posix(),
            "native_sampler_s3_step1_fused_audio_default": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT.as_posix(),
            "native_sampler_s3_step1_fused_audio_repeat": NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT.as_posix(),
            "native_sampler_s3_step1_audio_default": NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT.as_posix(),
            "native_sampler_s3_step1_audio_fast": NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST.as_posix(),
            "s3_bottleneck_audit": S3_BOTTLENECK_AUDIT.as_posix(),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
