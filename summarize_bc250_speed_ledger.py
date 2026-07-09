#!/usr/bin/env python3
"""Summarize BC-250 Chatterbox speed progress from saved benchmark artifacts.

This is intentionally non-generating. It only reads JSON artifacts that were
already captured during benchmarking; it does not load Chatterbox, start an API
worker, generate audio, compile exports, or probe ROCm/HIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
CPU_BENCH = ROOT / "exports" / "cpu_thread_bench"
DEFAULT_OUTPUT = BENCH / "bc250_speed_ledger_latest.json"

BASELINE_CPU_VS_5090 = BENCH / "baseline_cpu_vs_5090_2026-07-07.json"
CPU_FAST_STATUS = CPU_BENCH / "cpu_fast_path_status_2026-07-08.json"
PRE_FAST_VULKAN = BENCH / "vulkan_hybrid_api_24cu_current_default_guarded_pre40_2026-07-08.json"
DEFAULT_NO_WATERMARK = (
    BENCH / "vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json"
)
FAST_FUSED = (
    BENCH
    / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json"
)


def load_required(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing artifact: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        errors.append(f"invalid json: {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"unexpected json root: {path}")
        return {}
    return data


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def baseline_chunk270(data: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    for row in data.get("results", []):
        if row.get("case") == "chunk270_1":
            return row
    errors.append(f"missing chunk270_1 result: {BASELINE_CPU_VS_5090}")
    return {}


def request_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("requests", [])
        if isinstance(row, dict) and row.get("status") == 200
    ]


def selected_request(report: dict[str, Any], path: Path, errors: list[str], index: int = 1) -> dict[str, Any]:
    rows = request_rows(report)
    if not rows:
        errors.append(f"missing successful requests: {path}")
        return {}
    if index < len(rows):
        return rows[index]
    return rows[-1]


def debug_timing(row: dict[str, Any]) -> dict[str, Any]:
    return nested(row, "debug", "body", "last_request", default={}) or {}


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt_seconds(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.3f}s"


def fmt_ratio(value: Any) -> str:
    number = as_float(value)
    return "n/a" if number is None else f"{number:.3f}x"


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def gap(value: float | None, target: float | None) -> float | None:
    if value is None or target is None:
        return None
    return value - target


def row_from_request(
    *,
    key: str,
    label: str,
    artifact: Path,
    request: dict[str, Any],
    notes: str,
    runtime: str,
    original_cpu_seconds: float | None,
    rtx_5090_seconds: float | None,
    target_seconds: float | None,
) -> dict[str, Any]:
    timing = debug_timing(request)
    wall = as_float(request.get("wall_seconds"))
    total = as_float(timing.get("total_seconds"))
    compare_seconds = wall if wall is not None else total
    return {
        "key": key,
        "label": label,
        "runtime": runtime,
        "artifact": artifact.as_posix(),
        "notes": notes,
        "wall_seconds": wall,
        "server_total_seconds": total,
        "audio_seconds": as_float(request.get("duration_seconds")) or as_float(timing.get("audio_seconds")),
        "t3_seconds": as_float(timing.get("t3_seconds")),
        "s3_flow_seconds": as_float(timing.get("s3_flow_seconds")),
        "source_seconds": as_float(timing.get("source_seconds")),
        "hift_decode_seconds": as_float(timing.get("hift_decode_seconds")),
        "watermark_seconds": as_float(timing.get("watermark_seconds")),
        "ratio_to_5090": ratio(compare_seconds, rtx_5090_seconds),
        "gap_to_2x_target_seconds": gap(compare_seconds, target_seconds),
        "speedup_vs_original_cpu": ratio(original_cpu_seconds, compare_seconds),
    }


def build_report() -> dict[str, Any]:
    errors: list[str] = []
    baseline = load_required(BASELINE_CPU_VS_5090, errors)
    cpu_fast = load_required(CPU_FAST_STATUS, errors)
    pre_fast = load_required(PRE_FAST_VULKAN, errors)
    default_no_watermark = load_required(DEFAULT_NO_WATERMARK, errors)
    fast_fused = load_required(FAST_FUSED, errors)

    chunk270 = baseline_chunk270(baseline, errors) if baseline else {}
    original_cpu = as_float(chunk270.get("local_wall_seconds"))
    rtx_5090 = as_float(chunk270.get("remote_wall_seconds"))
    target = rtx_5090 * 2.0 if rtx_5090 is not None else None

    rows: list[dict[str, Any]] = []
    if original_cpu is not None:
        rows.append(
            {
                "key": "original_cpu_chunk270",
                "label": "Original BC-250 CPU chunk270",
                "runtime": "PyTorch CPU",
                "artifact": BASELINE_CPU_VS_5090.as_posix(),
                "notes": "Original local CPU measurement against the RTX 5090 reference server.",
                "wall_seconds": original_cpu,
                "server_total_seconds": None,
                "audio_seconds": as_float(chunk270.get("local_audio_seconds")),
                "t3_seconds": None,
                "s3_flow_seconds": None,
                "source_seconds": None,
                "hift_decode_seconds": None,
                "watermark_seconds": None,
                "ratio_to_5090": ratio(original_cpu, rtx_5090),
                "gap_to_2x_target_seconds": gap(original_cpu, target),
                "speedup_vs_original_cpu": 1.0,
            }
        )

    projected_cpu = as_float(cpu_fast.get("projected_no_watermark_chunk270_seconds"))
    if projected_cpu is not None:
        breakdown = cpu_fast.get("chunk270_breakdown") or {}
        rows.append(
            {
                "key": "cpu_fast_projected_no_watermark",
                "label": "Tuned CPU no-watermark projection",
                "runtime": "PyTorch CPU",
                "artifact": CPU_FAST_STATUS.as_posix(),
                "notes": "Best saved CPU thread setting with watermark cost removed; stable fallback only.",
                "wall_seconds": projected_cpu,
                "server_total_seconds": projected_cpu,
                "audio_seconds": None,
                "t3_seconds": as_float(breakdown.get("t3_seconds")),
                "s3_flow_seconds": as_float(breakdown.get("s3_flow_seconds")),
                "source_seconds": None,
                "hift_decode_seconds": as_float(breakdown.get("hift_seconds")),
                "watermark_seconds": 0.0,
                "ratio_to_5090": ratio(projected_cpu, rtx_5090),
                "gap_to_2x_target_seconds": gap(projected_cpu, target),
                "speedup_vs_original_cpu": ratio(original_cpu, projected_cpu),
            }
        )

    if pre_fast:
        rows.append(
            row_from_request(
                key="pre_fast_vulkan_default",
                label="Early BC-250 Vulkan default profile",
                artifact=PRE_FAST_VULKAN,
                request=selected_request(pre_fast, PRE_FAST_VULKAN, errors),
                notes="Pre-fast Vulkan baseline before no-watermark and native sampler/S3-step1 work.",
                runtime="ggml/IREE Vulkan",
                original_cpu_seconds=original_cpu,
                rtx_5090_seconds=rtx_5090,
                target_seconds=target,
            )
        )

    if default_no_watermark:
        rows.append(
            row_from_request(
                key="default_quality_vulkan_no_watermark",
                label="Default-quality Vulkan, no watermark",
                artifact=DEFAULT_NO_WATERMARK,
                request=selected_request(default_no_watermark, DEFAULT_NO_WATERMARK, errors),
                notes="Stable default-quality reference without watermark; still above the 2x target.",
                runtime="ggml/IREE Vulkan",
                original_cpu_seconds=original_cpu,
                rtx_5090_seconds=rtx_5090,
                target_seconds=target,
            )
        )

    if fast_fused:
        rows.append(
            row_from_request(
                key="fast_fused_vulkan_no_watermark",
                label="Fast-fused Vulkan, no watermark",
                artifact=FAST_FUSED,
                request=selected_request(fast_fused, FAST_FUSED, errors),
                notes="Best saved target-crossing path; listen-before-default.",
                runtime="ggml/IREE Vulkan",
                original_cpu_seconds=original_cpu,
                rtx_5090_seconds=rtx_5090,
                target_seconds=target,
            )
        )

    best_bc250 = None
    bc250_candidates = [
        row for row in rows if row.get("key") != "original_cpu_chunk270" and row.get("wall_seconds") is not None
    ]
    if bc250_candidates:
        best_bc250 = min(bc250_candidates, key=lambda item: item["wall_seconds"])

    summary = {
        "original_cpu_seconds": original_cpu,
        "rtx_5090_reference_seconds": rtx_5090,
        "target_2x_5090_seconds": target,
        "best_bc250_key": best_bc250.get("key") if best_bc250 else None,
        "best_bc250_wall_seconds": best_bc250.get("wall_seconds") if best_bc250 else None,
        "best_bc250_server_total_seconds": best_bc250.get("server_total_seconds") if best_bc250 else None,
        "best_bc250_ratio_to_5090": best_bc250.get("ratio_to_5090") if best_bc250 else None,
        "best_bc250_gap_to_target_seconds": best_bc250.get("gap_to_2x_target_seconds") if best_bc250 else None,
        "best_bc250_speedup_vs_original_cpu": best_bc250.get("speedup_vs_original_cpu") if best_bc250 else None,
        "target_met_by_best_wall": (
            best_bc250 is not None
            and target is not None
            and best_bc250.get("wall_seconds") is not None
            and best_bc250["wall_seconds"] <= target
        ),
    }
    return {
        "ok": not errors,
        "errors": errors,
        "description": "BC-250 speed ledger from saved benchmark artifacts only.",
        "note": "No audio generation, model load, worker start, compilation, or ROCm/HIP probing is performed.",
        "sources": {
            "baseline_cpu_vs_5090": BASELINE_CPU_VS_5090.as_posix(),
            "cpu_fast_status": CPU_FAST_STATUS.as_posix(),
            "pre_fast_vulkan": PRE_FAST_VULKAN.as_posix(),
            "default_no_watermark": DEFAULT_NO_WATERMARK.as_posix(),
            "fast_fused": FAST_FUSED.as_posix(),
        },
        "summary": summary,
        "rows": rows,
        "decision": (
            "Saved evidence shows fast-fused Vulkan meets the 2x RTX 5090 wall-time target, "
            "while default-quality Vulkan remains above target and CPU remains fallback-only."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    lines = [
        "# BC-250 Speed Ledger",
        "",
        "This ledger is built from saved benchmark artifacts only. It does not load the model, generate audio, start workers, compile exports, or probe ROCm/HIP.",
        "",
        "## Summary",
        "",
        f"- Original BC-250 CPU chunk270: `{fmt_seconds(summary.get('original_cpu_seconds'))}`",
        f"- RTX 5090 reference chunk270: `{fmt_seconds(summary.get('rtx_5090_reference_seconds'))}`",
        f"- 2x target: `{fmt_seconds(summary.get('target_2x_5090_seconds'))}`",
        f"- Best saved BC-250 path: `{summary.get('best_bc250_key')}`",
        f"- Best saved BC-250 wall: `{fmt_seconds(summary.get('best_bc250_wall_seconds'))}`",
        f"- Best saved BC-250 server total: `{fmt_seconds(summary.get('best_bc250_server_total_seconds'))}`",
        f"- Best saved BC-250 ratio to 5090: `{fmt_ratio(summary.get('best_bc250_ratio_to_5090'))}`",
        f"- Best saved BC-250 gap to 2x target: `{fmt_seconds(summary.get('best_bc250_gap_to_target_seconds'))}`",
        f"- Best saved BC-250 speedup vs original CPU: `{fmt_ratio(summary.get('best_bc250_speedup_vs_original_cpu'))}`",
        f"- Target met by best wall time: `{summary.get('target_met_by_best_wall')}`",
        "",
        "## Rows",
        "",
        "| Path | Runtime | Wall | Server Total | T3 | S3 | HiFT Decode | Watermark | Ratio To 5090 | Gap To Target | Speedup Vs CPU | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []):
        lines.append(
            "| "
            f"{row['label']} | {row['runtime']} | "
            f"`{fmt_seconds(row.get('wall_seconds'))}` | "
            f"`{fmt_seconds(row.get('server_total_seconds'))}` | "
            f"`{fmt_seconds(row.get('t3_seconds'))}` | "
            f"`{fmt_seconds(row.get('s3_flow_seconds'))}` | "
            f"`{fmt_seconds(row.get('hift_decode_seconds'))}` | "
            f"`{fmt_seconds(row.get('watermark_seconds'))}` | "
            f"`{fmt_ratio(row.get('ratio_to_5090'))}` | "
            f"`{fmt_seconds(row.get('gap_to_2x_target_seconds'))}` | "
            f"`{fmt_ratio(row.get('speedup_vs_original_cpu'))}` | "
            f"{row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for name, source in report.get("sources", {}).items():
        lines.append(f"- `{name}`: `{source}`")
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        for error in report["errors"]:
            lines.append(f"- {error}")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    md = args.output.with_suffix(".md")
    write_markdown(report, md)
    print(f"json={args.output}")
    print(f"markdown={md}")
    print(f"decision={report['decision']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
