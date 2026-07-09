#!/usr/bin/env python3
"""Compare guarded Chatterbox Vulkan benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
RTX_5090_SECONDS = 3.495
TARGET_2X_SECONDS = RTX_5090_SECONDS * 2.0


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def request_rows(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not report or report.get("status") != "ok":
        return []
    return [row for row in report.get("requests", []) if row.get("status") == 200]


def selected_request(report: dict[str, Any] | None, index: int) -> dict[str, Any] | None:
    rows = request_rows(report)
    if not rows:
        return None
    if index < len(rows):
        return rows[index]
    return rows[-1]


def timings(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    body = ((row.get("debug") or {}).get("body") or {}).get("last_request") or {}
    return body


def stage_seconds(row: dict[str, Any] | None) -> dict[str, float | None]:
    data = timings(row)
    return {
        "wall": float(row["wall_seconds"]) if row and row.get("wall_seconds") is not None else None,
        "server_total": data.get("total_seconds"),
        "t3": data.get("t3_seconds"),
        "s3": data.get("s3_flow_seconds"),
        "source": data.get("source_seconds"),
        "hift_decode": data.get("hift_decode_seconds"),
        "watermark": data.get("watermark_seconds"),
        "audio": float(row["duration_seconds"]) if row and row.get("duration_seconds") is not None else data.get("audio_seconds"),
    }


def delta(base: float | None, candidate: float | None) -> dict[str, float | None]:
    if base is None or candidate is None:
        return {"seconds_saved": None, "speedup": None, "percent_saved": None}
    return {
        "seconds_saved": base - candidate,
        "speedup": base / candidate if candidate else None,
        "percent_saved": ((base - candidate) / base) * 100.0 if base else None,
    }


def fmt(value: float | None, suffix: str = "s") -> str:
    if value is None:
        return "pending"
    return f"`{value:.3f}{suffix}`"


def make_report(
    baseline_path: Path,
    candidate_path: Path,
    output_json: Path,
    output_md: Path,
    warm_index: int,
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    baseline_row = selected_request(baseline, warm_index)
    candidate_row = selected_request(candidate, warm_index)
    baseline_stages = stage_seconds(baseline_row)
    candidate_stages = stage_seconds(candidate_row)
    comparisons = {
        key: delta(baseline_stages.get(key), candidate_stages.get(key))
        for key in sorted(set(baseline_stages) | set(candidate_stages))
    }
    report = {
        "description": f"Guarded BC-250 {baseline_label} vs {candidate_label} Chatterbox Vulkan benchmark comparison.",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_path": baseline_path.as_posix(),
        "candidate_path": candidate_path.as_posix(),
        "baseline_exists": baseline is not None,
        "candidate_exists": candidate is not None,
        "baseline_status": baseline.get("status") if baseline else None,
        "candidate_status": candidate.get("status") if candidate else None,
        "baseline_cu": (baseline or {}).get("cu_state", {}).get("num_cu"),
        "candidate_cu": (candidate or {}).get("cu_state", {}).get("num_cu"),
        "warm_request_index": warm_index,
        "rtx_5090_seconds": RTX_5090_SECONDS,
        "target_2x_seconds": TARGET_2X_SECONDS,
        "baseline_stages": baseline_stages,
        "candidate_stages": candidate_stages,
        "comparisons": comparisons,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, output_md)
    return report


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline_label = report.get("baseline_label", "baseline")
    candidate_label = report.get("candidate_label", "candidate")
    base_wall = report["baseline_stages"].get("wall")
    cand_wall = report["candidate_stages"].get("wall")
    wall_delta = report["comparisons"].get("wall", {})
    candidate_gap = cand_wall - TARGET_2X_SECONDS if cand_wall is not None else None
    candidate_ratio = cand_wall / RTX_5090_SECONDS if cand_wall is not None else None
    lines = [
        "# BC-250 CU Benchmark Comparison",
        "",
        f"- {baseline_label}: `{report['baseline_path']}`",
        f"- {candidate_label}: `{report['candidate_path']}`",
        f"- Warm request index: `{report['warm_request_index']}`",
        f"- RTX 5090 reference: `{RTX_5090_SECONDS:.3f}s`",
        f"- 2x target: `{TARGET_2X_SECONDS:.3f}s`",
        "",
        "## Summary",
        "",
        f"- {baseline_label} wall: {fmt(base_wall)}",
        f"- {candidate_label} wall: {fmt(cand_wall)}",
        f"- Seconds saved: {fmt(wall_delta.get('seconds_saved'))}",
        f"- Speedup: {fmt(wall_delta.get('speedup'), 'x')}",
        f"- {candidate_label} ratio to RTX 5090: {fmt(candidate_ratio, 'x')}",
        f"- {candidate_label} gap to 2x target: {fmt(candidate_gap)}",
        "",
        "## Stage Comparison",
        "",
        f"| Stage | {baseline_label} | {candidate_label} | Saved | Speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    names = [
        ("wall", "Wall"),
        ("server_total", "Server total"),
        ("t3", "T3"),
        ("s3", "S3"),
        ("source", "HiFT source/F0"),
        ("hift_decode", "HiFT decode"),
        ("watermark", "Watermark"),
    ]
    for key, label in names:
        comp = report["comparisons"].get(key, {})
        lines.append(
            f"| {label} | {fmt(report['baseline_stages'].get(key))} | "
            f"{fmt(report['candidate_stages'].get(key))} | "
            f"{fmt(comp.get('seconds_saved'))} | {fmt(comp.get('speedup'), 'x')} |"
        )
    if not report.get("candidate_exists"):
        lines.extend(
            [
                "",
                "## Status",
                "",
                f"The {candidate_label} benchmark artifact is not present yet. Run the guarded "
                "candidate benchmark after RADV reports the expected CU count, then rerun this "
                "comparison script.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BENCH / "vulkan_hybrid_api_24cu_current_default_guarded_pre40_2026-07-08.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=BENCH / "vulkan_hybrid_api_40cu_guarded_2026-07-08.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH / "cu_comparison_24_vs_40_2026-07-08.json",
    )
    parser.add_argument("--warm-request-index", type=int, default=1)
    parser.add_argument("--baseline-label", default="24-CU baseline")
    parser.add_argument("--candidate-label", default="40-CU candidate")
    args = parser.parse_args()
    report = make_report(
        args.baseline,
        args.candidate,
        args.output,
        args.output.with_suffix(".md"),
        args.warm_request_index,
        args.baseline_label,
        args.candidate_label,
    )
    print(f"json={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"baseline_wall={report['baseline_stages'].get('wall')}")
    print(f"candidate_wall={report['candidate_stages'].get('wall')}")


if __name__ == "__main__":
    main()
