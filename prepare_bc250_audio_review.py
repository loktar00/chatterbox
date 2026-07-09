#!/usr/bin/env python3
"""Prepare a BC-250 audio review page from existing WAV artifacts.

This script is non-generating. It does not load Chatterbox, start workers,
compile exports, or probe ROCm/HIP. It only inspects existing WAV/JSON evidence
and writes an ignored review manifest plus a local HTML page.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
AUDIO_CHECKS = ROOT / "exports" / "audio_checks"
DEFAULT_OUTPUT_DIR = ROOT / "exports" / "audio_review"


@dataclass(frozen=True)
class ReviewItem:
    key: str
    label: str
    role: str
    path: Path
    notes: str


def benchmark_json_for_wav(path: Path) -> Path:
    name = path.name
    for suffix in ("_chunk270_req0_2026-07-08.wav", "_chunk270_req1_2026-07-08.wav"):
        if name.endswith(suffix):
            return path.with_name(name.removesuffix(suffix) + "_2026-07-08.json")
    return path.with_suffix(".json")


def request_index_for_wav(path: Path) -> int:
    if "_req1_" in path.name:
        return 1
    return 0


REVIEW_ITEMS = [
    ReviewItem(
        key="fast_fused_req1",
        label="Fast fused target path, request 1",
        role="Candidate",
        path=BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_chunk270_req1_2026-07-08.wav",
        notes="Best saved target-crossing fast-fused path.",
    ),
    ReviewItem(
        key="fast_fused_req0",
        label="Fast fused target path, request 0",
        role="Repeatability",
        path=BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_chunk270_req0_2026-07-08.wav",
        notes="Same profile, first request.",
    ),
    ReviewItem(
        key="prior_fast_req1",
        label="Prior fast path, request 1",
        role="Fast reference",
        path=BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_no_watermark_guarded_chunk270_req1_2026-07-08.wav",
        notes="Prior non-fused fast path; waveform sanity says fast-fused is effectively identical.",
    ),
    ReviewItem(
        key="default_quality_req1",
        label="Default-quality no-watermark path, request 1",
        role="Quality reference",
        path=BENCH / "vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_chunk270_req1_2026-07-08.wav",
        notes="Default-quality reference; expected to differ from fast paths.",
    ),
]

SANITY_REPORTS = [
    AUDIO_CHECKS / "native_sampler_request_seeded_s3_step1_fused_basic_audio_sanity_2026-07-09.json",
    AUDIO_CHECKS / "native_sampler_request_seeded_s3_step1_fused_req1_vs_req0_audio_sanity_2026-07-09.json",
    AUDIO_CHECKS / "native_sampler_request_seeded_s3_step1_fused_vs_prior_fast_audio_sanity_2026-07-09.json",
    AUDIO_CHECKS / "native_sampler_request_seeded_s3_step1_fused_vs_default_no_watermark_audio_sanity_2026-07-09.json",
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def wav_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": path.as_posix()}
    try:
        with wave.open(path.as_posix(), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
    except Exception as exc:
        return {"exists": True, "path": path.as_posix(), "error": str(exc)}
    return {
        "exists": True,
        "path": path.as_posix(),
        "sha256": sha256(path),
        "frames": frames,
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "duration_seconds": frames / float(sample_rate) if sample_rate else None,
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": path.as_posix()}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"exists": True, "path": path.as_posix(), "error": str(exc)}
    data["exists"] = True
    data["path"] = path.as_posix()
    return data


def s3_call_summary(calls: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(calls, dict):
        return {"available": False}
    return {
        "available": True,
        "total": calls.get("total"),
        "vulkan": calls.get("vulkan"),
        "fallback_cpu": calls.get("fallback_cpu"),
    }


def provenance_for_wav(path: Path) -> dict[str, Any]:
    benchmark_json = benchmark_json_for_wav(path)
    data = load_json(benchmark_json)
    requests = data.get("requests") if isinstance(data.get("requests"), list) else []
    index = request_index_for_wav(path)
    request = requests[index] if index < len(requests) else {}
    last_request = (
        request.get("debug", {}).get("body", {}).get("last_request", {})
        if isinstance(request, dict)
        else {}
    )
    env = data.get("env_overrides", {}) if isinstance(data.get("env_overrides"), dict) else {}
    return {
        "benchmark_json": benchmark_json.as_posix(),
        "benchmark_json_exists": benchmark_json.exists(),
        "artifact_label": data.get("artifact_label"),
        "request_index": index,
        "generation_path": last_request.get("generation_path"),
        "experimental_vulkan_t3": last_request.get("experimental_vulkan_t3"),
        "experimental_vulkan_s3": last_request.get("experimental_vulkan_s3"),
        "experimental_vulkan_hift": last_request.get("experimental_vulkan_hift"),
        "env_overrides": env,
        "s3_encoder_calls": s3_call_summary(last_request.get("s3_encoder_calls")),
        "s3_estimator_calls": s3_call_summary(last_request.get("s3_estimator_calls")),
        "vulkan_heavy_path": bool(
            last_request.get("experimental_vulkan_t3")
            and last_request.get("experimental_vulkan_s3")
            and last_request.get("experimental_vulkan_hift")
        ),
        "s3_debug_fallback_cpu_zero": (
            s3_call_summary(last_request.get("s3_encoder_calls")).get("fallback_cpu") == 0
            and s3_call_summary(last_request.get("s3_estimator_calls")).get("fallback_cpu") == 0
        )
        if isinstance(last_request.get("s3_encoder_calls"), dict)
        and isinstance(last_request.get("s3_estimator_calls"), dict)
        else None,
    }


def short_metric(report: dict[str, Any]) -> dict[str, Any]:
    audio = report.get("audio", {})
    comparison = report.get("comparison_to_reference", {})
    return {
        "path": report.get("path"),
        "description": report.get("description"),
        "passed_basic_sanity": audio.get("passed_basic_sanity"),
        "duration_seconds": audio.get("duration_seconds"),
        "rms_dbfs": audio.get("rms_dbfs"),
        "clipped_fraction_abs_ge_0_999": audio.get("clipped_fraction_abs_ge_0_999"),
        "correlation": comparison.get("correlation"),
        "rms_diff": comparison.get("rms_diff"),
    }


def build_manifest() -> dict[str, Any]:
    items = []
    for item in REVIEW_ITEMS:
        info = wav_info(item.path)
        info.update(
            {
                "key": item.key,
                "label": item.label,
                "role": item.role,
                "notes": item.notes,
                "provenance": provenance_for_wav(item.path),
            }
        )
        items.append(info)
    reports = [load_json(path) for path in SANITY_REPORTS]
    return {
        "description": "BC-250 listen-test review manifest built from existing artifacts only.",
        "note": "No model load, audio generation, worker start, compilation, or ROCm/HIP probing is performed.",
        "items": items,
        "sanity_reports": [short_metric(report) for report in reports],
        "review_checklist": [
            "No static, buzz, or burst noise.",
            "No severe clipping or silence.",
            "Speech is intelligible across the whole clip.",
            "Pacing and pronunciation are acceptable for the target use.",
            "Fast-fused candidate is acceptable compared with the quality reference.",
        ],
        "decision": "Fast-fused remains listen-before-default until reviewed by ear.",
    }


def rel_to(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve().parent)


def write_html(manifest: dict[str, Any], output: Path) -> None:
    rows = []
    for item in manifest["items"]:
        path = Path(item["path"])
        provenance = item.get("provenance", {})
        rel_path = rel_to(path, output) if item.get("exists") else ""
        duration = item.get("duration_seconds")
        duration_text = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "missing"
        encoder = provenance.get("s3_encoder_calls", {})
        estimator = provenance.get("s3_estimator_calls", {})
        provenance_rows = [
            ("Artifact", provenance.get("artifact_label")),
            ("Generation path", provenance.get("generation_path")),
            ("Vulkan T3/S3/HiFT", f"{provenance.get('experimental_vulkan_t3')}/{provenance.get('experimental_vulkan_s3')}/{provenance.get('experimental_vulkan_hift')}"),
            ("S3 encoder", f"vulkan={encoder.get('vulkan')} fallback_cpu={encoder.get('fallback_cpu')}" if encoder.get("available") else "not recorded in this artifact"),
            ("S3 estimator", f"vulkan={estimator.get('vulkan')} fallback_cpu={estimator.get('fallback_cpu')}" if estimator.get("available") else "not recorded in this artifact"),
            ("Benchmark JSON", provenance.get("benchmark_json")),
        ]
        provenance_html = "".join(
            f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
            for label, value in provenance_rows
        )
        rows.append(
            "<section>"
            f"<h2>{html.escape(item['role'])}: {html.escape(item['label'])}</h2>"
            f"<p>{html.escape(item['notes'])}</p>"
            f"<p><code>{html.escape(path.as_posix())}</code></p>"
            f"<p>Duration: <strong>{html.escape(duration_text)}</strong></p>"
            f"<dl>{provenance_html}</dl>"
            + (
                f'<audio controls preload="metadata" src="{html.escape(rel_path)}"></audio>'
                if item.get("exists")
                else "<p><strong>Missing WAV artifact.</strong></p>"
            )
            + "</section>"
        )
    metrics = []
    for report in manifest["sanity_reports"]:
        metrics.append(
            "<li>"
            f"{html.escape(str(report.get('description')))}: "
            f"passed={html.escape(str(report.get('passed_basic_sanity')))}, "
            f"corr={html.escape(str(report.get('correlation')))}, "
            f"rms_diff={html.escape(str(report.get('rms_diff')))}"
            "</li>"
        )
    checklist = [f"<li>{html.escape(item)}</li>" for item in manifest["review_checklist"]]
    output.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                '<meta charset="utf-8">',
                "<title>BC-250 Audio Review</title>",
                "<style>",
                "body{font-family:system-ui,sans-serif;max-width:960px;margin:32px auto;padding:0 16px;line-height:1.4}",
                "section{border-top:1px solid #ccc;padding:20px 0}",
                "dt{font-weight:700;margin-top:6px}",
                "dd{margin-left:0}",
                "audio{width:100%;display:block;margin-top:8px}",
                "code{word-break:break-all}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>BC-250 Audio Review</h1>",
                f"<p>{html.escape(manifest['description'])}</p>",
                f"<p><strong>Decision:</strong> {html.escape(manifest['decision'])}</p>",
                "<h2>Listen Checklist</h2>",
                "<ul>",
                *checklist,
                "</ul>",
                *rows,
                "<h2>Saved Sanity Metrics</h2>",
                "<ul>",
                *metrics,
                "</ul>",
                "</body>",
                "</html>",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "bc250_audio_review_manifest.json"
    html_path = args.output_dir / "index.html"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_html(manifest, html_path)
    missing = [item for item in manifest["items"] if not item.get("exists")]
    if args.json:
        print(json.dumps({"ok": not missing, "manifest": manifest_path.as_posix(), "html": html_path.as_posix(), "missing": missing}, sort_keys=True))
    else:
        print(f"manifest={manifest_path}")
        print(f"html={html_path}")
        print(f"missing={len(missing)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
