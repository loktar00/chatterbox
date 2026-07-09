#!/usr/bin/env python3
"""Write a compact BC-250 runtime/exportability matrix from saved evidence.

This script is intentionally read-only with respect to runtime behavior: it
does not load Chatterbox, start an API worker, generate audio, or probe
ROCm/HIP. It distills the larger audit artifacts into a short decision matrix
that is easier to review before choosing the next optimization step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "exports" / "benchmarks"
RUNTIME_MATRIX = BENCH / "runtime_export_candidate_matrix_2026-07-08.json"
ACTIVE_GOAL = BENCH / "active_goal_state_2026-07-09.json"
LATEST_PROJECTION = BENCH / "latest_optimization_projection_2026-07-08.json"
DEFAULT_OUTPUT = BENCH / "bc250_runtime_matrix_compact_latest.json"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def fmt_seconds(value: Any) -> str:
    return f"{float(value):.3f}s" if isinstance(value, (int, float)) else "n/a"


def best_benchmark(runtime: dict[str, Any], label_fragment: str) -> dict[str, Any]:
    benches = nested(runtime, "current_performance", "benchmarks", default=[])
    for row in benches:
        if label_fragment in row.get("label", ""):
            return row
    return {}


def latest_t3_fast_token_buffer_validation() -> dict[str, Any]:
    paths = sorted(BENCH.glob("t3_native_fast_token_buffer_validation_*.json"))
    paths = [path for path in paths if not path.name.endswith("_benchmark.json")]
    if not paths:
        return {"exists": False}
    path = paths[-1]
    data = load_json(path)
    data["path"] = path.as_posix()
    data["exists"] = True
    return data


def component_rows(runtime: dict[str, Any], active: dict[str, Any], latest: dict[str, Any]) -> list[dict[str, Any]]:
    safe_validation = runtime.get("safe_validation") or {}
    tooling = runtime.get("tooling") or {}
    modules = tooling.get("python_modules") or {}
    commands = tooling.get("commands") or {}
    t3_bridges = nested(runtime, "t3", "bridges", default=[])
    t3_symbols_ok = bool(t3_bridges) and all(
        all((bridge.get("required_symbols") or {}).values())
        for bridge in t3_bridges
        if bridge.get("exists")
    )
    s3_buckets = nested(runtime, "s3", "buckets", default={})
    s3_all_present = bool(s3_buckets) and all(row.get("missing_count") == 0 for row in s3_buckets.values())
    hift_active = nested(runtime, "hift", "active", default={})
    hift_chunked_ok = nested(hift_active, "t128", "exists") is True and nested(hift_active, "t96", "exists") is True
    fused_live = nested(latest, "native_sampler", "live_request_seeded_s3_step1_fused", default={})
    audio = nested(active, "performance", "audio_sanity", default={})
    t3_fast_token_buffer = latest_t3_fast_token_buffer_validation()
    fast_validation = safe_validation.get("t3_native_sampler_s3_step1_fused") or {}
    fast_request = (fast_validation.get("requests") or [{}])[-1]
    fast_timing = nested(fast_request, "debug", "body", "last_request", default={})
    s3_audit = safe_validation.get("s3_bottleneck_audit") or {}
    s3_api = s3_audit.get("api_debug") or {}
    ncnn_smoke = safe_validation.get("ncnn_pnnx_cpu_smoke") or {}
    ncnn_comparison = nested(ncnn_smoke, "runtime", "cpu", "comparison", default={})
    executorch = safe_validation.get("executorch_tiny_export") or {}
    voice = safe_validation.get("voice_encoder_onnxruntime") or {}
    cpu = active.get("cpu") or {}

    return [
        {
            "component": "Safe CPU API",
            "runtime": "PyTorch CPU",
            "status": "active fallback",
            "evidence": {
                "projected_no_watermark_chunk270_seconds": cpu.get("projected_no_watermark_chunk270_seconds"),
                "best_thread_setting": cpu.get("best_thread_setting"),
            },
            "next_move": "Keep it running on :8000 while probing Vulkan workers separately.",
        },
        {
            "component": "T3 token model",
            "runtime": "ggml + Vulkan",
            "status": "active experimental path",
            "evidence": {
                "bridge_symbols_ok": t3_symbols_ok,
                "weights_exist": nested(runtime, "t3", "weights", "exists"),
                "native_fast_token_buffer_validated": t3_fast_token_buffer.get("native_fast_token_buffer"),
                "native_fast_token_buffer_tokens_equal": t3_fast_token_buffer.get("all_tokens_equal_reference"),
                "t3_seconds_fast_fused": fast_timing.get("t3_seconds"),
            },
            "next_move": "The remaining T3 cost is mostly ggml per-token wall time; avoid redoing IREE one-token prefill.",
        },
        {
            "component": "S3 flow",
            "runtime": "IREE + Vulkan",
            "status": "active for exact buckets; fused split8 is opt-in",
            "evidence": {
                "all_configured_bucket_artifacts_present": s3_all_present,
                "bucket_count": len(s3_buckets),
                "s3_flow_seconds_fast_fused": fast_timing.get("s3_flow_seconds"),
                "s3_flow_seconds_default_debug": s3_api.get("s3_flow_seconds"),
                "fused_live_status": fused_live.get("status"),
            },
            "next_move": "Reduce estimator dispatch/fetch cost or broaden fused estimator chains; keep exact buckets to avoid CPU fallback.",
        },
        {
            "component": "HiFT vocoder",
            "runtime": "IREE + Vulkan chunked core",
            "status": "active chunked path",
            "evidence": {
                "t128_exists": nested(hift_active, "t128", "exists"),
                "t96_exists": nested(hift_active, "t96", "exists"),
                "t722_rejected_probe_exists": nested(runtime, "hift", "rejected_t722_probe_exists"),
                "hift_decode_seconds_fast_fused": fast_timing.get("hift_decode_seconds"),
                "chunked_ready": hift_chunked_ok,
            },
            "next_move": "Do not retry the t722 whole-clip shape; keep 128-frame chunks and compact96 tail.",
        },
        {
            "component": "Voice encoder / conditionals",
            "runtime": "ONNX inspection / CPU fallback",
            "status": "not active in steady-state Vulkan path",
            "evidence": {
                "onnx_module": modules.get("onnx"),
                "onnxruntime_module": modules.get("onnxruntime"),
                "random_weight_onnxruntime_allclose": voice.get("allclose_1e_5"),
            },
            "next_move": "Leave on CPU unless startup or prompt-conditioning profiling shows it matters.",
        },
        {
            "component": "ncnn candidate",
            "runtime": "pnnx/ncnn",
            "status": "CPU conversion validated; ncnn Vulkan rejected on this host",
            "evidence": {
                "ncnn_module": modules.get("ncnn"),
                "pnnx_command": bool(commands.get("pnnx")),
                "cpu_smoke_allclose": ncnn_comparison.get("allclose_1e_5"),
            },
            "next_move": "Do not spend more BC-250 time on ncnn Vulkan unless using a disposable host/container.",
        },
        {
            "component": "ExecuTorch candidate",
            "runtime": "torch.export / ExecuTorch",
            "status": "tiny export probe only; not installed in stable env",
            "evidence": {
                "executorch_module": modules.get("executorch"),
                "executorch_exir_module": modules.get("executorch.exir"),
                "probe_status": executorch.get("status"),
                "torch_export_tiny_ok": nested(executorch, "torch_export", "ok"),
            },
            "next_move": "Use a disposable venv/container before any further ExecuTorch install work.",
        },
        {
            "component": "Request-level parallelism",
            "runtime": "router over one worker per GPU",
            "status": "available for throughput, not single-chunk latency",
            "evidence": {
                "router_script_exists": (ROOT / "chatterbox_router.py").exists(),
                "parallel_doc_exists": (ROOT / "BC250_PARALLEL_SCALING.md").exists(),
            },
            "next_move": "Use for multiple requests or long-text chunks; keep one request in flight per worker.",
        },
        {
            "component": "Audio sanity",
            "runtime": "saved WAV checks",
            "status": "basic sanity passed for fast-fused; listening still required",
            "evidence": {
                "basic_passed": audio.get("basic_passed"),
                "corr_vs_prior_fast": audio.get("corr_vs_prior_fast"),
                "corr_vs_default": audio.get("corr_vs_default"),
            },
            "next_move": "Listen to fast-fused output before making it the default profile.",
        },
    ]


def build_report() -> dict[str, Any]:
    runtime = load_json(RUNTIME_MATRIX)
    active = load_json(ACTIVE_GOAL)
    latest = load_json(LATEST_PROJECTION)
    fast = best_benchmark(runtime, "one-step fused")
    default_no_watermark = best_benchmark(runtime, "default quality, no watermark")
    default_with_watermark = best_benchmark(runtime, "default quality, watermark")
    perf = active.get("performance") or {}
    safe = active.get("safe_runtime") or {}
    guardrails = active.get("guardrails") or []
    rows = component_rows(runtime, active, latest)
    return {
        "description": "Compact BC-250 Chatterbox runtime/exportability matrix from saved evidence. No runtime probing is performed.",
        "sources": {
            "runtime_export_candidate_matrix": RUNTIME_MATRIX.as_posix(),
            "active_goal_state": ACTIVE_GOAL.as_posix(),
            "latest_optimization_projection": LATEST_PROJECTION.as_posix(),
        },
        "safe_runtime": {
            "ports": nested(safe, "ports", "listeners", default=[]),
            "cpu_health_ok": nested(safe, "health", "ok"),
            "cpu_device": nested(safe, "health", "body", "device"),
        },
        "performance": {
            "rtx_5090_reference_seconds": perf.get("rtx_5090_reference_seconds"),
            "target_2x_5090_seconds": perf.get("target_2x_5090_seconds"),
            "fast_fused_seconds": nested(perf, "best_fast_fused_request", "total_seconds"),
            "fast_fused_wall_seconds": nested(perf, "best_fast_fused_request", "wall_seconds"),
            "fast_fused_gap_to_target_seconds": perf.get("fast_fused_gap_to_target_seconds"),
            "default_no_watermark_seconds": default_no_watermark.get("best_total_seconds"),
            "default_with_watermark_seconds": default_with_watermark.get("best_total_seconds"),
            "fast_fused_artifact": fast.get("path"),
        },
        "components": rows,
        "guardrails": guardrails,
        "decision": (
            "Fast-fused BC-250 Vulkan meets the 2x 5090 target from saved benchmark evidence, "
            "but remains listen-before-default. Default-quality Vulkan is still above target."
        ),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    perf = report["performance"]
    lines = [
        "# BC-250 Runtime Matrix",
        "",
        "This is a compact summary from saved artifacts only. It does not load the model, generate audio, start workers, or probe ROCm/HIP.",
        "",
        "## Performance",
        "",
        f"- RTX 5090 reference: `{fmt_seconds(perf.get('rtx_5090_reference_seconds'))}`",
        f"- 2x target: `{fmt_seconds(perf.get('target_2x_5090_seconds'))}`",
        f"- BC-250 fast-fused: `{fmt_seconds(perf.get('fast_fused_seconds'))}`",
        f"- Fast-fused gap to target: `{fmt_seconds(perf.get('fast_fused_gap_to_target_seconds'))}`",
        f"- BC-250 default-quality no-watermark: `{fmt_seconds(perf.get('default_no_watermark_seconds'))}`",
        f"- BC-250 default-quality with watermark: `{fmt_seconds(perf.get('default_with_watermark_seconds'))}`",
        "",
        "## Components",
        "",
        "| Component | Runtime | Status | Next Move |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["components"]:
        lines.append(
            f"| {row['component']} | {row['runtime']} | {row['status']} | {row['next_move']} |"
        )
    lines.extend(["", "## Guardrails", ""])
    for item in report.get("guardrails", []):
        name = item.get("name", "guardrail") if isinstance(item, dict) else "guardrail"
        reason = item.get("reason", item) if isinstance(item, dict) else item
        lines.append(f"- `{name}`: {reason}")
    lines.extend(
        [
            "",
            "## Source Artifacts",
            "",
            f"- Runtime matrix: `{report['sources']['runtime_export_candidate_matrix']}`",
            f"- Active goal state: `{report['sources']['active_goal_state']}`",
            f"- Latest projection: `{report['sources']['latest_optimization_projection']}`",
            "",
        ]
    )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
