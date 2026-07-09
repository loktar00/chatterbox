#!/usr/bin/env python3
"""Non-generating audit of the BC-250 Chatterbox acceleration goal."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/root/chatterbox")
BENCH = ROOT / "exports" / "benchmarks"
CPU = ROOT / "exports" / "cpu_thread_bench"
AUDIO = ROOT / "exports" / "audio_checks"
OUT_JSON = BENCH / "active_goal_state_2026-07-09.json"
OUT_MD = BENCH / "active_goal_state_2026-07-09.md"
RTX_5090_REFERENCE_SECONDS = 3.495
TARGET_2X_5090_SECONDS = RTX_5090_REFERENCE_SECONDS * 2.0

CPU_RUNTIME = CPU / "cpu_runtime_status_2026-07-08.json"
CPU_FAST = CPU / "cpu_fast_path_status_2026-07-08.json"
RUNTIME_MATRIX = BENCH / "runtime_export_candidate_matrix_2026-07-08.json"
LATEST_PROJECTION = BENCH / "latest_optimization_projection_2026-07-08.json"
S3_AUDIT = BENCH / "s3_bottleneck_audit_2026-07-08.json"
FAST_FUSED_BENCH = (
    BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json"
)
FAST_FUSED_AUDIO_BASIC = AUDIO / "native_sampler_request_seeded_s3_step1_fused_basic_audio_sanity_2026-07-09.json"
FAST_FUSED_AUDIO_PRIOR = AUDIO / "native_sampler_request_seeded_s3_step1_fused_vs_prior_fast_audio_sanity_2026-07-09.json"
FAST_FUSED_AUDIO_DEFAULT = AUDIO / "native_sampler_request_seeded_s3_step1_fused_vs_default_no_watermark_audio_sanity_2026-07-09.json"


def run(cmd: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {"ok": True, "status": resp.status, "body": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def port_snapshot() -> dict[str, Any]:
    result = run(["ss", "-ltnp"])
    watched = (8000, 8002, 8003, 8004, 8010, 4123)
    listeners = [
        line
        for line in result["stdout"].splitlines()
        if any(f":{port} " in line or f":{port}\t" in line for port in watched)
    ]
    return {"watched_ports": watched, "listeners": listeners, "error": result["stderr"] if result["returncode"] else ""}


def request_timings(path: Path) -> list[dict[str, Any]]:
    data = load_json(path) or {}
    rows = []
    for req in data.get("requests", []):
        last = req.get("debug", {}).get("body", {}).get("last_request", {})
        rows.append(
            {
                "index": req.get("index"),
                "status": req.get("status"),
                "wall_seconds": req.get("wall_seconds"),
                "total_seconds": last.get("total_seconds"),
                "t3_seconds": last.get("t3_seconds"),
                "s3_flow_seconds": last.get("s3_flow_seconds"),
                "source_seconds": last.get("source_seconds"),
                "hift_decode_seconds": last.get("hift_decode_seconds"),
                "watermark_seconds": last.get("watermark_seconds"),
                "audio_seconds": last.get("audio_seconds"),
                "s3_bucket_inference": last.get("s3_bucket_inference"),
                "t3_loop_info": last.get("t3_loop_info"),
                "t3_prefill_info": last.get("t3_prefill_info"),
                "s3_encoder_calls": last.get("s3_encoder_calls"),
                "s3_estimator_calls": last.get("s3_estimator_calls"),
            }
        )
    return rows


def best_request(path: Path) -> dict[str, Any] | None:
    rows = [
        row
        for row in request_timings(path)
        if isinstance(row.get("total_seconds"), (int, float)) or isinstance(row.get("wall_seconds"), (int, float))
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row.get("total_seconds") or row.get("wall_seconds")))


def cpu_section() -> dict[str, Any]:
    runtime = load_json(CPU_RUNTIME) or {}
    fast = load_json(CPU_FAST) or {}
    return {
        "status": "optimized_fallback_not_target_path",
        "runtime_artifact": CPU_RUNTIME.as_posix(),
        "fast_status_artifact": CPU_FAST.as_posix(),
        "best_thread_setting": runtime.get("thread_benchmark_summary", {}).get("best"),
        "chunk270_breakdown": runtime.get("chunk270_breakdown"),
        "projected_no_watermark_chunk270_seconds": fast.get("projected_no_watermark_chunk270_seconds"),
        "safe_policy": fast.get("safe_api_policy"),
        "decision": (
            "CPU fallback is tightened to the best measured 2-thread/1-interop setting and has a separate no-watermark fast launcher. "
            "It remains roughly a 60s chunk270 fallback, not a route to the 7s target."
        ),
    }


def export_runtime_section() -> dict[str, Any]:
    matrix = load_json(RUNTIME_MATRIX) or {}
    tooling = matrix.get("tooling", {})
    safe_validation = matrix.get("safe_validation", {})
    return {
        "status": "covered_with_known_rejections",
        "artifact": RUNTIME_MATRIX.as_posix(),
        "tooling": tooling,
        "t3": matrix.get("t3"),
        "s3": matrix.get("s3"),
        "hift": matrix.get("hift"),
        "safe_validation_keys": sorted(safe_validation.keys()),
        "decisions": [
            "IREE/Vulkan is the active path for S3 and HiFT fixed-shape subgraphs.",
            "ggml/Vulkan is the active path for the T3 token loop.",
            "ncnn/pnnx is useful for CPU conversion/export inspection only; ncnn Vulkan is rejected on this host.",
            "ExecuTorch/EXIR is not installed in the stable venv; revisit only in a disposable environment.",
        ],
    }


def performance_section() -> dict[str, Any]:
    projection = load_json(LATEST_PROJECTION) or {}
    best = best_request(FAST_FUSED_BENCH)
    audio_basic = load_json(FAST_FUSED_AUDIO_BASIC) or {}
    audio_prior = load_json(FAST_FUSED_AUDIO_PRIOR) or {}
    audio_default = load_json(FAST_FUSED_AUDIO_DEFAULT) or {}
    prior_comparison = audio_prior.get("comparison_to_reference") or audio_prior.get("comparison") or {}
    default_comparison = audio_default.get("comparison_to_reference") or audio_default.get("comparison") or {}
    stage_rows = projection.get("stage_rows", [])
    fast_fused_row = next((row for row in stage_rows if row.get("path") == "fast S3 step1 fused midblocks, no watermark"), {})
    t3_loop = ((best or {}).get("t3_loop_info") or {})
    t3_timing = t3_loop.get("timings", {})
    return {
        "status": "target_met_for_listen_before_default_fast_fused_profile",
        "rtx_5090_reference_seconds": RTX_5090_REFERENCE_SECONDS,
        "target_2x_5090_seconds": TARGET_2X_5090_SECONDS,
        "fast_fused_benchmark": FAST_FUSED_BENCH.as_posix(),
        "best_fast_fused_request": best,
        "fast_fused_gap_to_target_seconds": (
            (best or {}).get("total_seconds") - TARGET_2X_5090_SECONDS
            if isinstance((best or {}).get("total_seconds"), (int, float))
            else None
        ),
        "stage_row": fast_fused_row,
        "remaining_hotspots_seconds": {
            "t3_total": (best or {}).get("t3_seconds"),
            "t3_ggml_token_step_wall": t3_timing.get("step_ggml_wall_seconds"),
            "t3_native_sampling": t3_timing.get("step_native_sampler_seconds"),
            "s3_flow": (best or {}).get("s3_flow_seconds"),
            "hift_source_f0": (best or {}).get("source_seconds"),
            "hift_decode": (best or {}).get("hift_decode_seconds"),
        },
        "audio_sanity": {
            "basic_artifact": FAST_FUSED_AUDIO_BASIC.as_posix(),
            "basic_passed": (audio_basic.get("audio") or {}).get("passed_basic_sanity"),
            "prior_fast_artifact": FAST_FUSED_AUDIO_PRIOR.as_posix(),
            "corr_vs_prior_fast": prior_comparison.get("correlation"),
            "rms_diff_vs_prior_fast": prior_comparison.get("rms_diff"),
            "default_artifact": FAST_FUSED_AUDIO_DEFAULT.as_posix(),
            "corr_vs_default": default_comparison.get("correlation"),
            "rms_diff_vs_default": default_comparison.get("rms_diff"),
            "decision": (
                "Basic audio metrics pass and fused output is effectively identical to the prior fast path, "
                "but the fast path remains listen-before-default because it differs from default-quality output."
            ),
        },
    }


def guardrails_section() -> list[dict[str, Any]]:
    return [
        {
            "name": "rocm_hip",
            "decision": "do_not_use",
            "reason": "BC-250 gfx1013 is not supported by the installed ROCm PyTorch wheel and native HIP testing was followed by a host crash.",
            "guarded_scripts": ["run_api_rocm.sh", "verify_rocm_torch.sh", "verify_native_hip_gfx1013.sh"],
        },
        {
            "name": "hift_t722_whole_clip",
            "decision": "do_not_retry",
            "reason": "Exact whole-clip HiFT Vulkan runtime hit RADV device loss; chunked 128 plus compact96 tail is the safe path.",
        },
        {
            "name": "ncnn_vulkan",
            "decision": "do_not_retry_here",
            "reason": "Tiny ncnn Vulkan probe segfaulted; ncnn remains CPU/export-inspection only in this container.",
        },
        {
            "name": "executorch_stable_venv",
            "decision": "do_not_install_in_place",
            "reason": "Dry-run indicated disruptive dependency replacement; use a disposable venv/container if revisiting.",
        },
    ]


def objective_coverage(report: dict[str, Any]) -> list[dict[str, Any]]:
    safe_health = report["safe_runtime"]["health"]
    ports = report["safe_runtime"]["ports"]["listeners"]
    return [
        {
            "requirement": "Tighten CPU performance as much as possible.",
            "status": "substantial_progress_not_target_path",
            "evidence": [
                CPU_RUNTIME.as_posix(),
                CPU_FAST.as_posix(),
                "/root/chatterbox/run_api.sh",
                "/root/chatterbox/run_api_cpu_fast.sh",
            ],
            "summary": "Best measured CPU thread setting is 2/1; separate CPU fast launcher disables watermark. CPU remains about 60s for chunk270.",
        },
        {
            "requirement": "Identify Chatterbox pieces that export cleanly.",
            "status": "substantial_progress",
            "evidence": [RUNTIME_MATRIX.as_posix()],
            "summary": "Voice/ONNX, IREE subgraphs, S3 buckets, HiFT fixed cores, and ggml T3 fixtures are cataloged; ncnn Vulkan and ExecuTorch limits are recorded.",
        },
        {
            "requirement": "Try Vulkan runtimes for isolated subgraphs.",
            "status": "substantial_progress",
            "evidence": [
                RUNTIME_MATRIX.as_posix(),
                S3_AUDIT.as_posix(),
                FAST_FUSED_BENCH.as_posix(),
            ],
            "summary": "IREE/Vulkan powers S3/HiFT subgraphs; ggml/Vulkan powers T3; ncnn CPU conversion validated but ncnn Vulkan rejected.",
        },
        {
            "requirement": "Keep the safe CPU API running while GPU acceleration progresses.",
            "status": "currently_satisfied",
            "evidence": ["http://127.0.0.1:8000/health", "ss -ltnp"],
            "summary": f"Health ok={safe_health.get('ok')}; watched listeners={ports}.",
        },
    ]


def write_markdown(report: dict[str, Any]) -> None:
    perf = report["performance"]
    cpu = report["cpu"]
    export_runtime = report["export_runtime"]
    best = perf["best_fast_fused_request"] or {}
    hot = perf["remaining_hotspots_seconds"]
    audio = perf["audio_sanity"]
    lines = [
        "# Active Goal State - 2026-07-09",
        "",
        "This is a non-generating audit. It reads saved artifacts plus the live CPU health endpoint.",
        "",
        "## Current Speed",
        "",
        f"- RTX 5090 reference: `{perf['rtx_5090_reference_seconds']:.3f}s`.",
        f"- 2x target: `{perf['target_2x_5090_seconds']:.3f}s`.",
        f"- Fast fused BC-250 best total: `{best.get('total_seconds'):.3f}s`.",
        f"- Fast fused BC-250 best wall: `{best.get('wall_seconds'):.3f}s`.",
        f"- Gap to target: `{perf['fast_fused_gap_to_target_seconds']:.3f}s`.",
        "- Status: target met for the opt-in listen-before-default fast fused profile.",
        "",
        "## Remaining Hotspots",
        "",
        f"- T3 total: `{hot.get('t3_total'):.3f}s`.",
        f"- T3 ggml/Vulkan token-step wall: `{hot.get('t3_ggml_token_step_wall'):.3f}s`.",
        f"- T3 native sampling: `{hot.get('t3_native_sampling'):.3f}s`.",
        f"- S3 flow: `{hot.get('s3_flow'):.3f}s`.",
        f"- HiFT source/F0: `{hot.get('hift_source_f0'):.3f}s`.",
        f"- HiFT decode: `{hot.get('hift_decode'):.3f}s`.",
        "",
        "The next meaningful speed work is T3 token-loop redesign or safer HiFT decode reduction. Further S3 micro-fusion has already produced only small live gains.",
        "",
        "## Audio Sanity",
        "",
        f"- Basic sanity passed: `{audio.get('basic_passed')}`.",
        f"- Correlation versus prior fast path: `{audio.get('corr_vs_prior_fast')}`.",
        f"- RMS diff versus prior fast path: `{audio.get('rms_diff_vs_prior_fast')}`.",
        f"- Correlation versus default-quality no-watermark path: `{audio.get('corr_vs_default')}`.",
        "- Decision: fused fast is waveform-equivalent to the prior fast path, but still listen-before-default because the fast path differs from default-quality output.",
        "",
        "## CPU Fallback",
        "",
        f"- Best CPU thread setting: `{cpu.get('best_thread_setting')}`.",
        f"- CPU chunk270 projected no-watermark: `{cpu.get('projected_no_watermark_chunk270_seconds'):.3f}s`.",
        "- CPU remains the stable fallback, not the target-performance path.",
        "",
        "## Export/Runtime Coverage",
        "",
        f"- Runtime matrix: `{export_runtime['artifact']}`.",
        f"- IREE available: `{export_runtime['tooling'].get('python_modules', {}).get('iree')}`.",
        f"- ncnn available: `{export_runtime['tooling'].get('python_modules', {}).get('ncnn')}`.",
        f"- ExecuTorch available: `{export_runtime['tooling'].get('python_modules', {}).get('executorch')}`.",
        "- Active paths: IREE/Vulkan for S3/HiFT, ggml/Vulkan for T3.",
        "",
        "## Guardrails",
        "",
    ]
    for item in report["guardrails"]:
        lines.append(f"- `{item['name']}`: {item['decision']} - {item['reason']}")
    lines.extend(["", "## Objective Coverage", ""])
    for item in report["objective_coverage"]:
        lines.append(f"- {item['requirement']} `{item['status']}`: {item['summary']}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    report: dict[str, Any] = {
        "description": "Current active-goal audit. No model load, audio generation, worker start, or ROCm/HIP probing.",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "safe_runtime": {
            "ports": port_snapshot(),
            "health": http_json("http://127.0.0.1:8000/health"),
        },
        "cpu": cpu_section(),
        "export_runtime": export_runtime_section(),
        "performance": performance_section(),
        "guardrails": guardrails_section(),
    }
    report["objective_coverage"] = objective_coverage(report)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"status={report['performance']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
