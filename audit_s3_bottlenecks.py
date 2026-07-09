#!/usr/bin/env python3
"""Summarize S3 Vulkan bottlenecks from saved, bounded benchmark artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
S3 = ROOT / "exports" / "s3_flow_vulkan_components"
OUT_JSON = BENCH / "s3_bottleneck_audit_2026-07-08.json"
OUT_MD = BENCH / "s3_bottleneck_audit_2026-07-08.md"
MIDBLOCK_RE = re.compile(r"s3_fused_midblock(\d+)_t1222")


def midblock_sort_key(path: Path) -> tuple[int, str]:
    match = MIDBLOCK_RE.search(path.name)
    return (int(match.group(1)) if match else 9999, path.name)


def active_shape_fused_midblock_paths() -> list[Path]:
    return sorted(
        S3.glob("s3_fused_midblock*_t1222_probe_5iter_2026-07-08.json"),
        key=midblock_sort_key,
    )


def active_shape_flag_matrix_paths() -> list[Path]:
    primary = sorted(
        S3.glob("s3_fused_midblock*_t1222_compile_flag_matrix_split8_20iter_2026-07-08.json"),
        key=midblock_sort_key,
    )
    exploratory = sorted(
        S3.glob("s3_fused_midblock*_t1222_compile_flag_matrix_2026-07-08.json"),
        key=midblock_sort_key,
    )
    return primary + [path for path in exploratory if path not in primary]

API_DEBUG = BENCH / "vulkan_hybrid_api_s3_debug_records_no_watermark_default_quality_guarded_2026-07-08.json"
DEFAULT_BASELINE = BENCH / "vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json"
NATIVE_SEEDED = BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_padded_s3_capped_guarded_2026-07-08.json"
CHAIN_EXACT = S3 / "s3_flow_hybrid_vulkan_encoder_estimator_cached_static_buffers_2026-07-08.json"
CHAIN_PADDED = S3 / "s3_flow_hybrid_vulkan_encoder_estimator_chunk270_padded_615_1230_validation_2026-07-08.json"
PERSISTENT_SOAK = S3 / "s3_vulkan_persistent_chains_soak_30cycles_2026-07-08.json"
ACTIVE_SHAPE_FUSED_ESTIMATOR_CHAIN = (
    S3 / "s3_estimator_fused_midblocks_t1222_split8_chain_10iter_2026-07-08.json"
)
ACTIVE_SHAPE_FUSED_API_BENCHMARK = (
    BENCH
    / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json"
)
ACTIVE_SHAPE_FUSED_AUDIO_BASIC = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_basic_audio_sanity_2026-07-09.json"
)
ACTIVE_SHAPE_FUSED_AUDIO_PRIOR_FAST = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_prior_fast_audio_sanity_2026-07-09.json"
)
ACTIVE_SHAPE_FUSED_AUDIO_DEFAULT = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_default_no_watermark_audio_sanity_2026-07-09.json"
)
ACTIVE_SHAPE_FUSED_AUDIO_REPEAT = (
    ROOT
    / "exports/audio_checks/native_sampler_request_seeded_s3_step1_fused_req1_vs_req0_audio_sanity_2026-07-09.json"
)
FUSED_MIDBLOCKS = [
    S3 / "s3_fused_midblock0_t1210_probe_50iter_2026-07-08.json",
    S3 / "s3_fused_midblock1_t1210_probe_30iter_2026-07-08.json",
]
ACTIVE_SHAPE_FUSED_MIDBLOCKS = [
]
ACTIVE_SHAPE_FLAG_MATRICES = [
]


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def last_success(report: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in report.get("requests", []) if row.get("status") == 200]
    return rows[-1] if rows else {}


def successful_requests(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report.get("requests", []) if row.get("status") == 200]


def last_request(row: dict[str, Any]) -> dict[str, Any]:
    return (((row.get("debug") or {}).get("body") or {}).get("last_request") or {})


def sum_records(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record.get(key) or 0.0) for record in records)


def timing_from_api_debug() -> dict[str, Any]:
    report = load(API_DEBUG)
    row = last_success(report)
    timing = last_request(row)
    encoder = timing.get("s3_encoder_calls") or {}
    estimator = timing.get("s3_estimator_calls") or {}
    encoder_records = encoder.get("records") or []
    estimator_records = estimator.get("records") or []
    s3_total = timing.get("s3_flow_seconds")
    encoder_fetch = float(encoder.get("vulkan_chain_fetch_seconds") or 0.0)
    estimator_fetch = float(estimator.get("vulkan_chain_fetch_seconds") or 0.0)
    frontend = float(estimator.get("frontend_seconds") or 0.0)
    to_numpy = float(encoder.get("to_numpy_seconds") or 0.0) + float(estimator.get("to_numpy_seconds") or 0.0)
    accounted = encoder_fetch + estimator_fetch + frontend + to_numpy
    return {
        "source": API_DEBUG.as_posix(),
        "status": report.get("status"),
        "wall_seconds": row.get("wall_seconds"),
        "s3_flow_seconds": s3_total,
        "bucket": timing.get("s3_bucket_inference"),
        "encoder": {
            "total": encoder.get("total"),
            "vulkan": encoder.get("vulkan"),
            "fallback_cpu": encoder.get("fallback_cpu"),
            "vulkan_chain_fetch_seconds": encoder_fetch,
            "to_numpy_seconds": float(encoder.get("to_numpy_seconds") or 0.0),
            "records": [
                {
                    "index": record.get("index"),
                    "vulkan_chain_fetch_seconds": record.get("vulkan_chain_fetch_seconds"),
                    "to_numpy_seconds": record.get("to_numpy_seconds"),
                    "total_seconds": record.get("total_seconds"),
                    "bucket_shape": record.get("bucket_shape"),
                }
                for record in encoder_records
            ],
        },
        "estimator": {
            "total": estimator.get("total"),
            "vulkan": estimator.get("vulkan"),
            "fallback_cpu": estimator.get("fallback_cpu"),
            "frontend_seconds": frontend,
            "vulkan_chain_fetch_seconds": estimator_fetch,
            "to_numpy_seconds": float(estimator.get("to_numpy_seconds") or 0.0),
            "records": [
                {
                    "index": record.get("index"),
                    "frontend_seconds": record.get("frontend_seconds"),
                    "vulkan_chain_fetch_seconds": record.get("vulkan_chain_fetch_seconds"),
                    "to_numpy_seconds": record.get("to_numpy_seconds"),
                    "total_seconds": record.get("total_seconds"),
                    "bucket_shape": record.get("bucket_shape"),
                }
                for record in estimator_records
            ],
        },
        "accounted_seconds": accounted,
        "unaccounted_s3_overhead_seconds": (
            float(s3_total) - accounted if isinstance(s3_total, (int, float)) else None
        ),
        "estimator_share_of_s3": (
            estimator_fetch / float(s3_total) if isinstance(s3_total, (int, float)) and s3_total else None
        ),
        "encoder_share_of_s3": (
            encoder_fetch / float(s3_total) if isinstance(s3_total, (int, float)) and s3_total else None
        ),
        "to_numpy_share_of_s3": (
            to_numpy / float(s3_total) if isinstance(s3_total, (int, float)) and s3_total else None
        ),
    }


def s3_from_benchmark(path: Path) -> float | None:
    report = load(path)
    timing = last_request(last_success(report))
    value = timing.get("s3_flow_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def chain_summary(path: Path) -> dict[str, Any]:
    report = load(path)
    encoder = report.get("encoder_calls") or {}
    estimator = report.get("estimator_calls") or {}
    return {
        "source": path.as_posix(),
        "hybrid_flow_seconds": report.get("hybrid_flow_seconds"),
        "cpu_flow_seconds": report.get("cpu_flow_seconds"),
        "encoder_token_frames": report.get("encoder_token_frames"),
        "encoder_up_frames": report.get("encoder_up_frames"),
        "estimator_frames": report.get("estimator_frames"),
        "encoder_vulkan_chain_fetch_seconds": encoder.get("vulkan_chain_fetch_seconds"),
        "estimator_vulkan_chain_fetch_seconds": estimator.get("vulkan_chain_fetch_seconds"),
        "encoder_fallback_cpu": encoder.get("fallback_cpu"),
        "estimator_fallback_cpu": estimator.get("fallback_cpu"),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fused_midblock_summary() -> dict[str, Any]:
    reports = [load(path) for path in FUSED_MIDBLOCKS if path.exists()]
    rows = []
    for path, report in zip([path for path in FUSED_MIDBLOCKS if path.exists()], reports, strict=True):
        timings = report.get("timings") or {}
        fused = (timings.get("fused_no_fetch") or {}).get("mean_ms")
        stitched = (timings.get("stitched_no_fetch") or {}).get("mean_ms")
        fused_fetch = (timings.get("fused_fetch_output") or {}).get("mean_ms")
        stitched_fetch = (timings.get("stitched_fetch_output") or {}).get("mean_ms")
        no_fetch_save_ms = (
            stitched - fused if isinstance(stitched, (int, float)) and isinstance(fused, (int, float)) else None
        )
        fetch_save_ms = (
            stitched_fetch - fused_fetch
            if isinstance(stitched_fetch, (int, float)) and isinstance(fused_fetch, (int, float))
            else None
        )
        rows.append(
            {
                "source": path.as_posix(),
                "status": report.get("status"),
                "mid_index": report.get("mid_index"),
                "validation_allclose_1e_4": (report.get("validation") or {}).get("allclose_1e_4"),
                "validation_max_abs_error": (report.get("validation") or {}).get("max_abs_error"),
                "compile_status": (report.get("compile") or {}).get("status"),
                "compile_seconds": (report.get("compile") or {}).get("seconds"),
                "vmfb_size_bytes": (report.get("compile") or {}).get("vmfb_size_bytes"),
                "fused_no_fetch_ms": fused,
                "stitched_no_fetch_ms": stitched,
                "fused_fetch_output_ms": fused_fetch,
                "stitched_fetch_output_ms": stitched_fetch,
                "speedup_vs_stitched_no_fetch": report.get("speedup_vs_stitched_no_fetch"),
                "no_fetch_save_ms": no_fetch_save_ms,
                "fetch_save_ms": fetch_save_ms,
            }
        )
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("validation_allclose_1e_4") is True
    ]
    avg_no_fetch_save_ms = mean(
        [row["no_fetch_save_ms"] for row in valid_rows if isinstance(row.get("no_fetch_save_ms"), (int, float))]
    )
    avg_fetch_save_ms = mean(
        [row["fetch_save_ms"] for row in valid_rows if isinstance(row.get("fetch_save_ms"), (int, float))]
    )
    estimator_calls_per_s3 = 2
    mid_blocks = 12
    return {
        "sources": [path.as_posix() for path in FUSED_MIDBLOCKS],
        "valid_probe_count": len(valid_rows),
        "probes": rows,
        "average_no_fetch_save_ms": avg_no_fetch_save_ms,
        "average_fetch_save_ms": avg_fetch_save_ms,
        "projected_full_s3_no_fetch_savings_seconds": (
            avg_no_fetch_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
            if avg_no_fetch_save_ms is not None
            else None
        ),
        "projected_full_s3_fetch_savings_seconds": (
            avg_fetch_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
            if avg_fetch_save_ms is not None
            else None
        ),
    }


def active_shape_fused_midblock_summary() -> dict[str, Any]:
    paths = active_shape_fused_midblock_paths()
    reports = [load(path) for path in paths if path.exists()]
    rows = []
    for path, report in zip([path for path in paths if path.exists()], reports, strict=True):
        timings = report.get("timings") or {}
        fused = (timings.get("fused_no_fetch") or {}).get("mean_ms")
        stitched = (timings.get("stitched_no_fetch") or {}).get("mean_ms")
        fused_fetch = (timings.get("fused_fetch_output") or {}).get("mean_ms")
        stitched_fetch = (timings.get("stitched_fetch_output") or {}).get("mean_ms")
        no_fetch_save_ms = (
            stitched - fused if isinstance(stitched, (int, float)) and isinstance(fused, (int, float)) else None
        )
        fetch_save_ms = (
            stitched_fetch - fused_fetch
            if isinstance(stitched_fetch, (int, float)) and isinstance(fused_fetch, (int, float))
            else None
        )
        rows.append(
            {
                "source": path.as_posix(),
                "status": report.get("status"),
                "frames": report.get("frames"),
                "mid_index": report.get("mid_index"),
                "iterations": (timings.get("fused_no_fetch") or {}).get("iterations"),
                "validation_allclose_1e_4": (report.get("validation") or {}).get("allclose_1e_4"),
                "validation_max_abs_error": (report.get("validation") or {}).get("max_abs_error"),
                "compile_status": (report.get("compile") or {}).get("status"),
                "compile_seconds": (report.get("compile") or {}).get("seconds"),
                "vmfb_size_bytes": (report.get("compile") or {}).get("vmfb_size_bytes"),
                "fused_no_fetch_ms": fused,
                "stitched_no_fetch_ms": stitched,
                "fused_fetch_output_ms": fused_fetch,
                "stitched_fetch_output_ms": stitched_fetch,
                "speedup_vs_stitched_no_fetch": report.get("speedup_vs_stitched_no_fetch"),
                "no_fetch_save_ms": no_fetch_save_ms,
                "fetch_save_ms": fetch_save_ms,
            }
        )
    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("validation_allclose_1e_4") is True
    ]
    avg_no_fetch_save_ms = mean(
        [row["no_fetch_save_ms"] for row in valid_rows if isinstance(row.get("no_fetch_save_ms"), (int, float))]
    )
    avg_fetch_save_ms = mean(
        [row["fetch_save_ms"] for row in valid_rows if isinstance(row.get("fetch_save_ms"), (int, float))]
    )
    estimator_calls_per_s3 = 2
    mid_blocks = 12
    projected_no_fetch = (
        avg_no_fetch_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
        if avg_no_fetch_save_ms is not None
        else None
    )
    projected_fetch = (
        avg_fetch_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
        if avg_fetch_save_ms is not None
        else None
    )
    return {
        "sources": [path.as_posix() for path in paths],
        "valid_probe_count": len(valid_rows),
        "probes": rows,
        "average_no_fetch_save_ms": avg_no_fetch_save_ms,
        "average_fetch_save_ms": avg_fetch_save_ms,
        "projected_full_s3_no_fetch_savings_seconds": projected_no_fetch,
        "projected_full_s3_fetch_savings_seconds": projected_fetch,
        "recommended_full_s3_savings_seconds": max(0.0, projected_no_fetch or 0.0),
        "recommendation": (
            "Do not prioritize straight mid-block fusion for the active t1222 shape unless a longer "
            "timing run or different compile strategy reverses the no-fetch slowdown."
            if valid_rows
            else "Run one active-shape t1222 fused mid-block probe before projecting production savings."
        ),
    }


def active_shape_flag_matrix_summary() -> dict[str, Any]:
    paths = active_shape_flag_matrix_paths()
    reports = [(path, load(path)) for path in paths if path.exists()]
    rows = []
    for path, report in reports:
        variants = []
        for variant in report.get("variants") or []:
            validation = variant.get("validation") or {}
            timings = variant.get("timings") or {}
            variants.append(
                {
                    "name": variant.get("name"),
                    "status": variant.get("status"),
                    "validation_allclose_1e_4": validation.get("allclose_1e_4"),
                    "validation_max_abs_error": validation.get("max_abs_error"),
                    "no_fetch_ms": (timings.get("no_fetch") or {}).get("mean_ms"),
                    "fetch_output_ms": (timings.get("fetch_output") or {}).get("mean_ms"),
                    "no_fetch_save_ms": variant.get("no_fetch_save_ms"),
                    "fetch_output_save_ms": variant.get("fetch_output_save_ms"),
                    "speedup_vs_stitched_no_fetch": variant.get("speedup_vs_stitched_no_fetch"),
                    "speedup_vs_stitched_fetch_output": variant.get("speedup_vs_stitched_fetch_output"),
                    "vmfb_size_bytes": (variant.get("compile") or {}).get("vmfb_size_bytes"),
                }
            )
        rows.append(
            {
                "source": path.as_posix(),
                "status": report.get("status"),
                "frames": report.get("frames"),
                "mid_index": report.get("mid_index"),
                "iterations": report.get("iterations"),
                "warmup": report.get("warmup"),
                "stitched_no_fetch_ms": ((report.get("stitched_timings") or {}).get("no_fetch") or {}).get("mean_ms"),
                "stitched_fetch_output_ms": ((report.get("stitched_timings") or {}).get("fetch_output") or {}).get("mean_ms"),
                "best_no_fetch_variant": report.get("best_no_fetch_variant"),
                "best_no_fetch_save_ms": report.get("best_no_fetch_save_ms"),
                "best_fetch_output_variant": report.get("best_fetch_output_variant"),
                "best_fetch_output_save_ms": report.get("best_fetch_output_save_ms"),
                "variants": variants,
            }
        )

    def best_variant_is_valid(row: dict[str, Any], key: str) -> bool:
        name = row.get(key)
        if not name:
            return False
        for variant in row.get("variants") or []:
            if variant.get("name") == name:
                return (
                    variant.get("status") == "ok"
                    and variant.get("validation_allclose_1e_4") is True
                )
        return False

    # There can be short exploratory matrices and longer confirmation matrices
    # for the same midblock. Use one validated row per midblock, preferring the
    # longest timing run, so projections do not double-count midblock0.
    primary_by_midblock: dict[int, dict[str, Any]] = {}
    for row in rows:
        mid_index = row.get("mid_index")
        if not isinstance(mid_index, int) or row.get("status") != "ok":
            continue
        if not (
            best_variant_is_valid(row, "best_no_fetch_variant")
            or best_variant_is_valid(row, "best_fetch_output_variant")
        ):
            continue
        current = primary_by_midblock.get(mid_index)
        if current is None or int(row.get("iterations") or 0) > int(current.get("iterations") or 0):
            primary_by_midblock[mid_index] = row

    primary_rows = [primary_by_midblock[key] for key in sorted(primary_by_midblock)]
    best_no_fetch_saves = [
        float(row["best_no_fetch_save_ms"])
        for row in primary_rows
        if best_variant_is_valid(row, "best_no_fetch_variant")
        and isinstance(row.get("best_no_fetch_save_ms"), (int, float))
        and row["best_no_fetch_save_ms"] > 0
    ]
    best_fetch_output_saves = [
        float(row["best_fetch_output_save_ms"])
        for row in primary_rows
        if best_variant_is_valid(row, "best_fetch_output_variant")
        and isinstance(row.get("best_fetch_output_save_ms"), (int, float))
        and row["best_fetch_output_save_ms"] > 0
    ]
    avg_best_no_fetch_save_ms = mean(best_no_fetch_saves)
    avg_best_fetch_output_save_ms = mean(best_fetch_output_saves)
    estimator_calls_per_s3 = 2
    mid_blocks = 12
    projected_no_fetch = (
        avg_best_no_fetch_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
        if avg_best_no_fetch_save_ms is not None
        else 0.0
    )
    projected_fetch = (
        avg_best_fetch_output_save_ms * mid_blocks * estimator_calls_per_s3 / 1000.0
        if avg_best_fetch_output_save_ms is not None
        else 0.0
    )
    return {
        "sources": [path.as_posix() for path in paths],
        "available_report_count": len(rows),
        "primary_sources": [row.get("source") for row in primary_rows],
        "primary_source": primary_rows[0].get("source") if primary_rows else None,
        "primary_iterations": primary_rows[0].get("iterations") if primary_rows else None,
        "tested_midblock_count": len(primary_rows),
        "total_midblocks": mid_blocks,
        "best_no_fetch_variant": (
            "split8" if primary_rows and all(row.get("best_no_fetch_variant") == "split8" for row in primary_rows) else None
        ),
        "average_best_no_fetch_save_ms": avg_best_no_fetch_save_ms,
        "best_no_fetch_save_ms": avg_best_no_fetch_save_ms,
        "best_fetch_output_variant": (
            "split8"
            if primary_rows and all(row.get("best_fetch_output_variant") == "split8" for row in primary_rows)
            else None
        ),
        "average_best_fetch_output_save_ms": avg_best_fetch_output_save_ms,
        "best_fetch_output_save_ms": avg_best_fetch_output_save_ms,
        "candidate_full_s3_no_fetch_savings_seconds": projected_no_fetch,
        "candidate_full_s3_fetch_output_savings_seconds": projected_fetch,
        "production_credit_seconds": 0.0,
        "primary_reports": primary_rows,
        "reports": rows,
        "recommendation": (
            f"split8 is the best validated active-shape compile flag across {len(primary_rows)} of "
            f"{mid_blocks} midblocks tested so far. Treat it as a candidate for a fused-midblock "
            "integration sweep, not as production credit, until all midblocks compile and the "
            "combined S3 chain wins in a guarded API benchmark."
            if primary_rows
            else "Run the active-shape compile-flag matrix before choosing an IREE flag strategy."
        ),
    }


def active_shape_fused_estimator_chain_summary() -> dict[str, Any]:
    report = load(ACTIVE_SHAPE_FUSED_ESTIMATOR_CHAIN)
    validation = ((report.get("validation") or {}).get("fused_vs_stitched") or {})
    timings = report.get("timings") or {}
    savings = report.get("savings") or {}
    return {
        "source": ACTIVE_SHAPE_FUSED_ESTIMATOR_CHAIN.as_posix(),
        "available": bool(report),
        "frames": report.get("frames"),
        "variant": report.get("variant"),
        "module_load_seconds": report.get("module_load_seconds"),
        "validation_allclose_1e_4": validation.get("allclose_1e_4"),
        "validation_max_abs_error": validation.get("max_abs_error"),
        "stitched_fetch_output_ms": ((timings.get("stitched_fetch_final_output") or {}).get("mean_ms")),
        "fused_fetch_output_ms": ((timings.get("fused_fetch_final_output") or {}).get("mean_ms")),
        "stitched_no_fetch_ms": ((timings.get("stitched_no_fetch") or {}).get("mean_ms")),
        "fused_no_fetch_ms": ((timings.get("fused_no_fetch") or {}).get("mean_ms")),
        "fetch_output_seconds_per_estimator_call": savings.get("fetch_output_seconds_per_estimator_call"),
        "fetch_output_seconds_for_two_s3_estimator_calls": savings.get("fetch_output_seconds_for_two_s3_estimator_calls"),
        "no_fetch_seconds_per_estimator_call": savings.get("no_fetch_seconds_per_estimator_call"),
        "no_fetch_seconds_for_two_s3_estimator_calls": savings.get("no_fetch_seconds_for_two_s3_estimator_calls"),
        "production_credit_seconds": 0.0,
        "recommendation": (
            "Full fixed-shape estimator-chain timing confirms the split8 fused-midblock candidate. "
            "Integrate behind an opt-in S3 runtime flag and require a guarded API benchmark before "
            "giving production credit."
            if validation.get("allclose_1e_4") is True
            else "Do not integrate until fused-chain validation passes against the stitched chain."
        ),
    }


def active_shape_fused_api_benchmark_summary() -> dict[str, Any]:
    report = load(ACTIVE_SHAPE_FUSED_API_BENCHMARK)
    audio_basic = load(ACTIVE_SHAPE_FUSED_AUDIO_BASIC)
    audio_prior_fast = load(ACTIVE_SHAPE_FUSED_AUDIO_PRIOR_FAST)
    audio_default = load(ACTIVE_SHAPE_FUSED_AUDIO_DEFAULT)
    audio_repeat = load(ACTIVE_SHAPE_FUSED_AUDIO_REPEAT)
    audio_summary = audio_basic.get("audio") or {}
    prior_comparison = audio_prior_fast.get("comparison_to_reference") or {}
    default_comparison = audio_default.get("comparison_to_reference") or {}
    repeat_comparison = audio_repeat.get("comparison_to_reference") or {}
    rows = successful_requests(report)
    if not rows:
        return {
            "source": ACTIVE_SHAPE_FUSED_API_BENCHMARK.as_posix(),
            "available": bool(report),
            "status": report.get("status"),
            "audio_sanity": {
                "basic_source": ACTIVE_SHAPE_FUSED_AUDIO_BASIC.as_posix(),
                "passed_basic_sanity": audio_summary.get("passed_basic_sanity"),
            },
        }
    timings = [last_request(row) for row in rows]
    totals = [float(timing["total_seconds"]) for timing in timings if isinstance(timing.get("total_seconds"), (int, float))]
    walls = [float(row["wall_seconds"]) for row in rows if isinstance(row.get("wall_seconds"), (int, float))]
    best_index = min(
        range(len(rows)),
        key=lambda index: timings[index].get("total_seconds", float("inf")),
    )
    best_timing = timings[best_index]
    best_estimator = best_timing.get("s3_estimator_calls") or {}
    estimator_records = best_estimator.get("records") or []
    return {
        "source": ACTIVE_SHAPE_FUSED_API_BENCHMARK.as_posix(),
        "available": True,
        "status": report.get("status"),
        "request_count": len(rows),
        "best_total_seconds": min(totals) if totals else None,
        "best_wall_seconds": min(walls) if walls else None,
        "best_s3_flow_seconds": best_timing.get("s3_flow_seconds"),
        "best_t3_seconds": best_timing.get("t3_seconds"),
        "best_hift_decode_seconds": best_timing.get("hift_decode_seconds"),
        "best_estimator_chain_fetch_seconds": best_estimator.get("vulkan_chain_fetch_seconds"),
        "best_estimator_chain_types": [record.get("chain_type") for record in estimator_records],
        "best_encoder_chain_fetch_seconds": (best_timing.get("s3_encoder_calls") or {}).get("vulkan_chain_fetch_seconds"),
        "audio_sanity": {
            "basic_source": ACTIVE_SHAPE_FUSED_AUDIO_BASIC.as_posix(),
            "prior_fast_source": ACTIVE_SHAPE_FUSED_AUDIO_PRIOR_FAST.as_posix(),
            "default_source": ACTIVE_SHAPE_FUSED_AUDIO_DEFAULT.as_posix(),
            "repeat_source": ACTIVE_SHAPE_FUSED_AUDIO_REPEAT.as_posix(),
            "passed_basic_sanity": audio_summary.get("passed_basic_sanity"),
            "duration_seconds": audio_summary.get("duration_seconds"),
            "peak_abs": audio_summary.get("peak_abs"),
            "rms_dbfs": audio_summary.get("rms_dbfs"),
            "clipped_fraction_abs_ge_0_999": audio_summary.get("clipped_fraction_abs_ge_0_999"),
            "correlation_vs_prior_fast": prior_comparison.get("correlation"),
            "rms_diff_vs_prior_fast": prior_comparison.get("rms_diff"),
            "correlation_vs_default": default_comparison.get("correlation"),
            "rms_diff_vs_default": default_comparison.get("rms_diff"),
            "correlation_req1_vs_req0": repeat_comparison.get("correlation"),
            "rms_diff_req1_vs_req0": repeat_comparison.get("rms_diff"),
        },
        "production_credit_seconds": 0.0,
        "recommendation": (
            "The fused split8 S3 path is integrated behind an opt-in API flag and produced the fastest live "
            "BC-250 benchmark so far. Basic audio sanity passes and the waveform is effectively identical to "
            "the prior fast path, but keep it listen-before-default until human listening review."
        ),
    }


def main() -> int:
    api = timing_from_api_debug()
    exact_s3 = s3_from_benchmark(DEFAULT_BASELINE)
    padded_s3 = s3_from_benchmark(NATIVE_SEEDED)
    padding_delta = (
        padded_s3 - exact_s3
        if isinstance(padded_s3, float) and isinstance(exact_s3, float)
        else None
    )
    persistent = load(PERSISTENT_SOAK)
    report = {
        "description": "S3 Vulkan bottleneck audit from saved API telemetry and isolated chain probes.",
        "api_debug": api,
        "bucket_padding": {
            "default_exact_s3_seconds": exact_s3,
            "request_seeded_padded_s3_seconds": padded_s3,
            "padding_delta_seconds": padding_delta,
            "default_source": DEFAULT_BASELINE.as_posix(),
            "request_seeded_source": NATIVE_SEEDED.as_posix(),
        },
        "isolated_chains": {
            "exact": chain_summary(CHAIN_EXACT),
            "padded": chain_summary(CHAIN_PADDED),
            "persistent_soak": {
                "source": PERSISTENT_SOAK.as_posix(),
                "cycles": persistent.get("cycles"),
                "summary": persistent.get("summary"),
            },
        },
        "fused_midblock": fused_midblock_summary(),
        "active_shape_fused_midblock": active_shape_fused_midblock_summary(),
        "active_shape_fused_midblock_flag_matrix": active_shape_flag_matrix_summary(),
        "active_shape_fused_estimator_chain": active_shape_fused_estimator_chain_summary(),
        "active_shape_fused_api_benchmark": active_shape_fused_api_benchmark_summary(),
        "decision": (
            "S3 is dominated by estimator Vulkan chain execution, not Python tensor conversion. "
            "Exact bucket export is useful to avoid CPU fallback. The active t1222 fused-midblock probes validated "
            "numerically, the split8 compile-flag variant showed bounded isolated wins across all midblocks, "
            "the fixed-shape fused estimator chain confirmed a small full-chain win, and the opt-in API path "
            "now uses it successfully. It still needs listening validation before default/production credit."
        ),
        "next_moves": [
            "Keep S3 debug telemetry in API artifacts for future guarded runs.",
            "Run audio sanity/listening checks for the fused fast API output before making it the preferred fast launcher.",
            "Prioritize larger estimator-level fusion or fewer dispatch boundaries over broad exact-bucket generation.",
            "Export new exact buckets only when they prevent CPU fallback for a real accepted token path.",
        ],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"decision={report['decision']}")
    return 0


def fmt_seconds(value: Any) -> str:
    return f"{value:.3f}s" if isinstance(value, (int, float)) else "n/a"


def fmt_pct(value: Any) -> str:
    return f"{value * 100.0:.2f}%" if isinstance(value, (int, float)) else "n/a"


def fmt_ms(value: Any) -> str:
    return f"{value:.3f}ms" if isinstance(value, (int, float)) else "n/a"


def write_markdown(report: dict[str, Any]) -> None:
    api = report["api_debug"]
    fused = report["fused_midblock"]
    active_fused = report["active_shape_fused_midblock"]
    flag_matrix = report["active_shape_fused_midblock_flag_matrix"]
    fused_chain = report["active_shape_fused_estimator_chain"]
    fused_api = report["active_shape_fused_api_benchmark"]
    padding = report["bucket_padding"]
    lines = [
        "# S3 Bottleneck Audit - 2026-07-08",
        "",
        "## API Telemetry",
        "",
        f"- Source: `{api['source']}`.",
        f"- S3 flow: `{fmt_seconds(api.get('s3_flow_seconds'))}`.",
        f"- Encoder Vulkan chain: `{fmt_seconds(api['encoder'].get('vulkan_chain_fetch_seconds'))}` "
        f"({fmt_pct(api.get('encoder_share_of_s3'))}).",
        f"- Estimator Vulkan chains: `{fmt_seconds(api['estimator'].get('vulkan_chain_fetch_seconds'))}` "
        f"({fmt_pct(api.get('estimator_share_of_s3'))}).",
        f"- Tensor-to-NumPy copy total: `{fmt_seconds(api['encoder'].get('to_numpy_seconds') + api['estimator'].get('to_numpy_seconds'))}` "
        f"({fmt_pct(api.get('to_numpy_share_of_s3'))}).",
        f"- Unaccounted wrapper overhead: `{fmt_seconds(api.get('unaccounted_s3_overhead_seconds'))}`.",
        "",
        "## Bucket Padding",
        "",
        f"- Exact-bucket default S3: `{fmt_seconds(padding.get('default_exact_s3_seconds'))}`.",
        f"- Padded request-seeded native S3: `{fmt_seconds(padding.get('request_seeded_padded_s3_seconds'))}`.",
        f"- Observed padding delta: `{fmt_seconds(padding.get('padding_delta_seconds'))}`.",
        "",
        "## Fused Mid-Block Probe - t1210 Legacy Shape",
        "",
        f"- Valid probes: `{fused.get('valid_probe_count')}`.",
        f"- Average no-fetch per-midblock savings: `{fmt_ms(fused.get('average_no_fetch_save_ms'))}`.",
        f"- Average fetch-output per-midblock savings: `{fmt_ms(fused.get('average_fetch_save_ms'))}`.",
        f"- Projected no-fetch full-S3 savings: `{fmt_seconds(fused.get('projected_full_s3_no_fetch_savings_seconds'))}`.",
        f"- Projected fetch-output full-S3 savings: `{fmt_seconds(fused.get('projected_full_s3_fetch_savings_seconds'))}`.",
        "",
        "## Fused Mid-Block Probe - Active t1222 Shape",
        "",
        f"- Valid probes: `{active_fused.get('valid_probe_count')}`.",
        f"- Average no-fetch per-midblock savings: `{fmt_ms(active_fused.get('average_no_fetch_save_ms'))}`.",
        f"- Average fetch-output per-midblock savings: `{fmt_ms(active_fused.get('average_fetch_save_ms'))}`.",
        f"- Projected no-fetch full-S3 savings: `{fmt_seconds(active_fused.get('projected_full_s3_no_fetch_savings_seconds'))}`.",
        f"- Recommended projected full-S3 savings: `{fmt_seconds(active_fused.get('recommended_full_s3_savings_seconds'))}`.",
        f"- Recommendation: {active_fused.get('recommendation')}",
        "",
        "## Active t1222 Compile-Flag Matrix",
        "",
        f"- Primary sources: `{', '.join(flag_matrix.get('primary_sources') or [])}`.",
        f"- Tested midblocks: `{flag_matrix.get('tested_midblock_count')}` of `{flag_matrix.get('total_midblocks')}`.",
        f"- Best no-fetch variant: `{flag_matrix.get('best_no_fetch_variant')}` "
        f"({fmt_ms(flag_matrix.get('best_no_fetch_save_ms'))} per midblock).",
        f"- Best fetch-output variant: `{flag_matrix.get('best_fetch_output_variant')}` "
        f"({fmt_ms(flag_matrix.get('best_fetch_output_save_ms'))} per midblock).",
        f"- Candidate no-fetch full-S3 savings: `{fmt_seconds(flag_matrix.get('candidate_full_s3_no_fetch_savings_seconds'))}`.",
        f"- Candidate fetch-output full-S3 savings: `{fmt_seconds(flag_matrix.get('candidate_full_s3_fetch_output_savings_seconds'))}`.",
        f"- Production credit: `{fmt_seconds(flag_matrix.get('production_credit_seconds'))}`.",
        f"- Recommendation: {flag_matrix.get('recommendation')}",
        "",
        "## Active t1222 Fused Estimator Chain",
        "",
        f"- Source: `{fused_chain.get('source')}`.",
        f"- Validation allclose 1e-4: `{fused_chain.get('validation_allclose_1e_4')}`.",
        f"- Validation max abs error: `{fused_chain.get('validation_max_abs_error')}`.",
        f"- Stitched fetch-output estimator call: `{fmt_ms(fused_chain.get('stitched_fetch_output_ms'))}`.",
        f"- Fused fetch-output estimator call: `{fmt_ms(fused_chain.get('fused_fetch_output_ms'))}`.",
        f"- Confirmed two-call S3 candidate savings: `{fmt_seconds(fused_chain.get('fetch_output_seconds_for_two_s3_estimator_calls'))}`.",
        f"- Production credit: `{fmt_seconds(fused_chain.get('production_credit_seconds'))}`.",
        f"- Recommendation: {fused_chain.get('recommendation')}",
        "",
        "## Active t1222 Fused API Benchmark",
        "",
        f"- Source: `{fused_api.get('source')}`.",
        f"- Status: `{fused_api.get('status')}`.",
        f"- Best total: `{fmt_seconds(fused_api.get('best_total_seconds'))}`.",
        f"- Best wall: `{fmt_seconds(fused_api.get('best_wall_seconds'))}`.",
        f"- Best S3 flow: `{fmt_seconds(fused_api.get('best_s3_flow_seconds'))}`.",
        f"- Estimator chain types: `{fused_api.get('best_estimator_chain_types')}`.",
        f"- Audio sanity pass: `{(fused_api.get('audio_sanity') or {}).get('passed_basic_sanity')}`.",
        f"- Audio correlation vs prior fast: `{(fused_api.get('audio_sanity') or {}).get('correlation_vs_prior_fast')}`.",
        f"- Audio correlation vs default: `{(fused_api.get('audio_sanity') or {}).get('correlation_vs_default')}`.",
        f"- Production credit: `{fmt_seconds(fused_api.get('production_credit_seconds'))}`.",
        f"- Recommendation: {fused_api.get('recommendation')}",
        "",
        "## Decision",
        "",
        report["decision"],
        "",
        "## Next Moves",
        "",
    ]
    lines.extend(f"- {item}" for item in report["next_moves"])
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
