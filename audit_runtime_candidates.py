#!/usr/bin/env python3
"""Audit local runtime/export candidates without starting a model server."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
BENCH = EXPORTS / "benchmarks"
OUT_JSON = BENCH / "runtime_export_candidate_matrix_2026-07-08.json"
OUT_MD = BENCH / "runtime_export_candidate_matrix_2026-07-08.md"
IREE_SMOKE = BENCH / "iree_vulkan_snake_smoke_rerun_2026-07-08.json"
NCNN_SMOKE = BENCH / "ncnn_pnnx_smoke_cpuonly_fixed_2026-07-08.json"
EXECUTORCH_SMOKE = BENCH / "executorch_tiny_export_probe_2026-07-08.json"
VOICE_ONNX_SMOKE = BENCH / "voice_encoder_onnxruntime_random_weights_2026-07-08.json"
T3_LOOP_SUMMARY = BENCH / "vulkan_hybrid_api_t3_loop_info_no_watermark_default_quality_guarded_2026-07-08_t3_loop_summary.json"
FAST_T3_LOOP_SUMMARY = BENCH / "vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08_t3_loop_summary.json"
T3_SEEN_MASK_SAMPLER = BENCH / "t3_seen_mask_sampler_validation_2026-07-08.json"
T3_SAMPLER_CANDIDATES = BENCH / "t3_sampler_candidate_profile_2026-07-08.json"
T3_NATIVE_SAMPLER = BENCH / "t3_native_sampler_microbench_2026-07-08.json"
T3_NATIVE_SAMPLER_BRIDGE = BENCH / "t3_native_sampler_bridge_ctypes_2026-07-08.json"
T3_NATIVE_SAMPLER_REAL_LOGITS = BENCH / "t3_native_sampler_real_logits_validation_2026-07-08.json"
T3_NATIVE_SAMPLER_LIVE = BENCH / "vulkan_hybrid_api_native_sampler_no_watermark_default_quality_guarded_2026-07-08.json"
T3_NATIVE_SAMPLER_PADDED_CAPPED = BENCH / "vulkan_hybrid_api_native_sampler_padded_s3_capped_guarded_2026-07-08.json"
T3_NATIVE_SAMPLER_REQUEST_SEEDED = (
    BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_padded_s3_capped_guarded_2026-07-08.json"
)
T3_NATIVE_SAMPLER_S3_STEP1 = (
    BENCH / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_no_watermark_guarded_2026-07-08.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_FUSED = (
    BENCH
    / "vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json"
)
T3_NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_padded_vs_default_no_watermark_audio_sanity_2026-07-08.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_vs_default_no_watermark_audio_sanity_2026-07-08.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_vs_s3_step1_audio_sanity_2026-07-08.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_fused_basic_audio_sanity_2026-07-09.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_prior_fast_audio_sanity_2026-07-09.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_fused_vs_default_no_watermark_audio_sanity_2026-07-09.json"
)
T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT = (
    EXPORTS / "audio_checks/native_sampler_request_seeded_s3_step1_fused_req1_vs_req0_audio_sanity_2026-07-09.json"
)
FAST_AUDIO_SANITY = (
    EXPORTS / "audio_checks/s3_step1_no_watermark_t3_loop_info_vs_default_no_watermark_audio_sanity_2026-07-08.json"
)
LATEST_PROJECTION = BENCH / "latest_optimization_projection_2026-07-08.json"
CPU_FAST_STATUS = EXPORTS / "cpu_thread_bench/cpu_fast_path_status_2026-07-08.json"
S3_BOTTLENECK_AUDIT = BENCH / "s3_bottleneck_audit_2026-07-08.json"
DEFAULT_WATERMARK_BENCH = BENCH / "vulkan_hybrid_api_32_enabled_24_reported_guarded_2026-07-08.json"
DEFAULT_NO_WATERMARK_BENCH = (
    BENCH / "vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json"
)
S3_STEP1_BENCH = BENCH / "vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08.json"
STATUS_SCRIPT = ROOT / "chatterbox_status.py"
ROUTER_SCRIPT = ROOT / "run_router.sh"
ROUTER_API = ROOT / "chatterbox_router.py"
VULKAN_FAST_SCRIPT = ROOT / "run_api_vulkan_fast.sh"
VULKAN_FAST_FUSED_SCRIPT = ROOT / "run_api_vulkan_fast_fused.sh"
T3_LOOP_INFO_TRIAL = ROOT / "run_t3_loop_info_trial.py"
RTX_5090_REFERENCE_SECONDS = 3.495
TARGET_2X_5090_SECONDS = RTX_5090_REFERENCE_SECONDS * 2.0

T3_WEIGHTS = EXPORTS / "ggml_t3_real_prompt_multistep_chunk270_s4_p935"
T3_LIBS = [
    ROOT / "libt3_ggml_vulkan_bridge_f16weights.so",
    ROOT / "libt3_ggml_vulkan_bridge.so",
]
S3_BUCKETS = {
    605: (605, 1210, 1210),
    611: (611, 1222, 1222),
    615: (615, 1230, 1230),
    629: (629, 1258, 1258),
}
HIFT_CORE = EXPORTS / "iree_vulkan_real_hift_core"


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def command_available(name: str) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    local = ROOT / ".venv" / "bin" / name
    if local.exists():
        return local.as_posix()
    return None


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def dir_size_mb(path: Path) -> float:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return total / (1024.0 * 1024.0)


def bridge_symbols(path: Path) -> dict[str, Any]:
    required = [
        "cb_t3_ggml_run_step",
        "cb_t3_ggml_set_layer_cache",
        "cb_t3_ggml_set_layer_cache_prefix",
        "cb_t3_ggml_set_layer_cache_range",
    ]
    if not path.exists():
        return {"path": path.as_posix(), "exists": False, "required_symbols": {}}
    result = subprocess.run(
        ["nm", "-D", path.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = result.stdout or ""
    return {
        "path": path.as_posix(),
        "exists": True,
        "size_mb": file_size_mb(path),
        "nm_returncode": result.returncode,
        "required_symbols": {symbol: symbol in stdout for symbol in required},
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def request_timing(req: dict[str, Any]) -> dict[str, Any]:
    last_request = req.get("debug", {}).get("body", {}).get("last_request", {})
    return {
        "status": req.get("status"),
        "wall_seconds": req.get("wall_seconds"),
        "duration_seconds": req.get("duration_seconds"),
        "total_seconds": last_request.get("total_seconds"),
        "t3_seconds": last_request.get("t3_seconds"),
        "s3_flow_seconds": last_request.get("s3_flow_seconds"),
        "source_seconds": last_request.get("source_seconds"),
        "hift_decode_seconds": last_request.get("hift_decode_seconds"),
        "watermark_applied": last_request.get("watermark_applied"),
        "watermark_seconds": last_request.get("watermark_seconds"),
        "audio_seconds": last_request.get("audio_seconds"),
        "generation_path": last_request.get("generation_path"),
        "t3_loop_info": last_request.get("t3_loop_info"),
    }


def benchmark_summary(path: Path, label: str, production_quality: bool) -> dict[str, Any]:
    data = load_json(path)
    if data is None:
        return {"label": label, "path": path.as_posix(), "exists": False, "production_quality": production_quality}
    requests = [request_timing(req) for req in data.get("requests", [])]
    totals = [item["total_seconds"] for item in requests if isinstance(item.get("total_seconds"), (int, float))]
    walls = [item["wall_seconds"] for item in requests if isinstance(item.get("wall_seconds"), (int, float))]
    best_total = min(totals) if totals else None
    best_wall = min(walls) if walls else None
    return {
        "label": label,
        "path": path.as_posix(),
        "exists": True,
        "production_quality": production_quality,
        "artifact_label": data.get("artifact_label"),
        "env_overrides": data.get("env_overrides", {}),
        "request_count": len(requests),
        "best_total_seconds": best_total,
        "best_wall_seconds": best_wall,
        "last_request": requests[-1] if requests else None,
        "ratio_to_5090": best_total / RTX_5090_REFERENCE_SECONDS if isinstance(best_total, (int, float)) else None,
        "gap_to_2x_target_seconds": (
            best_total - TARGET_2X_5090_SECONDS if isinstance(best_total, (int, float)) else None
        ),
    }


def current_performance() -> dict[str, Any]:
    return {
        "rtx_5090_reference_seconds": RTX_5090_REFERENCE_SECONDS,
        "target_2x_5090_seconds": TARGET_2X_5090_SECONDS,
        "benchmarks": [
            benchmark_summary(DEFAULT_NO_WATERMARK_BENCH, "default quality, no watermark", True),
            benchmark_summary(DEFAULT_WATERMARK_BENCH, "default quality, watermark retained", True),
            benchmark_summary(
                T3_NATIVE_SAMPLER_REQUEST_SEEDED,
                "request-seeded native sampler, padded S3 no watermark",
                False,
            ),
            benchmark_summary(S3_STEP1_BENCH, "fast listening candidate, one-step S3 no watermark", False),
            benchmark_summary(
                T3_NATIVE_SAMPLER_S3_STEP1,
                "request-seeded native sampler + one-step S3 no watermark",
                False,
            ),
            benchmark_summary(
                T3_NATIVE_SAMPLER_S3_STEP1_FUSED,
                "request-seeded native sampler + one-step fused S3 no watermark",
                False,
            ),
        ],
    }


def s3_bucket_artifacts(bucket: tuple[int, int, int]) -> dict[str, Any]:
    from benchmark_s3_encoder_iree_runtime_chain import encoder_module_names, vmfb_path
    from benchmark_s3_estimator_distinct_iree_runtime_chain import distinct_vmfb, helper_vmfb, module_names

    token_frames, up_frames, estimator_frames = bucket
    encoder = [vmfb_path(name) for name in encoder_module_names(token_frames, up_frames).values()]
    estimator = [distinct_vmfb(name) for name in module_names(estimator_frames).values()]
    helper = [
        helper_vmfb(f"s3_flow_transpose_c256_t{estimator_frames}"),
        helper_vmfb(f"s3_flow_transpose_t{estimator_frames}_c256"),
        helper_vmfb(f"s3_flow_cat_channel_256_256_t{estimator_frames}"),
    ]
    paths = encoder + estimator + helper
    missing = [path.as_posix() for path in paths if not path.exists()]
    return {
        "bucket": {
            "encoder_token_frames": token_frames,
            "encoder_up_frames": up_frames,
            "estimator_frames": estimator_frames,
        },
        "required_vmfb_count": len(paths),
        "missing_count": len(missing),
        "missing": missing[:20],
    }


def hift_artifacts() -> dict[str, Any]:
    active = {
        "t128": HIFT_CORE / "real_hift_core_no_fft_t128_vulkan_gfx1013.vmfb",
        "t96": HIFT_CORE / "real_hift_core_no_fft_t96_vulkan_gfx1013.vmfb",
    }
    rejected = HIFT_CORE / "real_hift_core_no_fft_t722_vulkan_gfx1013.vmfb"
    rejected_probe = HIFT_CORE / "real_hift_core_t722_probe.json"
    return {
        "active": {
            name: {
                "path": path.as_posix(),
                "exists": path.exists(),
                "size_mb": file_size_mb(path) if path.exists() else None,
            }
            for name, path in active.items()
        },
        "rejected_t722_vmfb_exists": rejected.exists(),
        "rejected_t722_probe_exists": rejected_probe.exists(),
    }


def write_markdown(report: dict[str, Any]) -> str:
    def fmt_seconds(value: Any) -> str:
        return f"{value:.3f}s" if isinstance(value, (int, float)) else "n/a"

    def fmt_ratio(value: Any) -> str:
        return f"{value:.2f}x" if isinstance(value, (int, float)) else "n/a"

    lines = [
        "# Runtime Export Candidate Matrix - 2026-07-08",
        "",
        "## Local Tooling",
        "",
        "| Runtime | Import/CLI Status | Local Action |",
        "| --- | --- | --- |",
    ]
    tools = report["tooling"]
    ncnn_ready = bool(tools["python_modules"]["ncnn"] and tools["commands"]["pnnx"])
    onnx_runtime_ready = bool(tools["python_modules"]["onnx"] and tools["python_modules"]["onnxruntime"])
    rows = [
        (
            "IREE",
            f"python={tools['python_modules']['iree']}, runtime={tools['python_modules']['iree.runtime']}, "
            f"compile={bool(tools['commands']['iree-compile'])}, run={bool(tools['commands']['iree-run-module'])}",
            "Usable now for IREE/Vulkan VMFB probes.",
        ),
        (
            "ncnn",
            f"python={tools['python_modules']['ncnn']}, ncnnoptimize={bool(tools['commands']['ncnnoptimize'])}, "
            f"pnnx={bool(tools['commands']['pnnx'])}",
            (
                "Usable now for bounded CPU pnnx-to-ncnn probes; ncnn Vulkan is rejected on this host."
                if ncnn_ready
                else "Not locally testable until ncnn/pnnx are installed."
            ),
        ),
        (
            "ExecuTorch",
            f"python={tools['python_modules']['executorch']}, exir={tools['python_modules']['executorch.exir']}, "
            f"cli={bool(tools['commands']['executorch'])}",
            "Tiny torch.export probe passed, but ExecuTorch/EXIR is not installed; revisit only in a disposable env.",
        ),
        (
            "ONNX",
            f"onnx={tools['python_modules']['onnx']}, onnxruntime={tools['python_modules']['onnxruntime']}",
            (
                "Export inspection and local CPU ONNX Runtime validation are possible."
                if onnx_runtime_ready
                else "Export inspection is possible; local ONNX Runtime execution is not installed."
            ),
        ),
        (
            "ggml/Vulkan",
            "bridge libraries present" if all(item["exists"] for item in report["t3"]["bridges"]) else "missing bridge",
            "Usable now for the T3 token loop.",
        ),
    ]
    for runtime, status, action in rows:
        lines.append(f"| {runtime} | {status} | {action} |")

    smoke = report.get("safe_validation", {}).get("iree_vulkan_snake_smoke")
    if smoke:
        lines.extend(
            [
                "",
                "## Current Safe Validation",
                "",
                "| Check | Result | Evidence |",
                "| --- | --- | --- |",
                (
                    "| IREE/Vulkan saved Snake VMFB rerun | "
                    f"allclose_1e_5={smoke.get('allclose_1e_5')}, "
                    f"max_abs_error={smoke.get('max_abs_error')} | "
                    f"`{IREE_SMOKE.as_posix()}` |"
                ),
            ]
        )
    ncnn_smoke = report.get("safe_validation", {}).get("ncnn_pnnx_cpu_smoke")
    if ncnn_smoke:
        if not smoke:
            lines.extend(
                [
                    "",
                    "## Current Safe Validation",
                    "",
                    "| Check | Result | Evidence |",
                    "| --- | --- | --- |",
                ]
            )
        comparison = ncnn_smoke.get("runtime", {}).get("cpu", {}).get("comparison", {})
        lines.append(
            "| pnnx -> ncnn CPU tiny conv smoke | "
            f"allclose_1e_5={comparison.get('allclose_1e_5')}, "
            f"max_abs_error={comparison.get('max_abs_error')} | "
            f"`{NCNN_SMOKE.as_posix()}` |"
        )
    executorch_smoke = report.get("safe_validation", {}).get("executorch_tiny_export")
    if executorch_smoke:
        if not smoke and not ncnn_smoke:
            lines.extend(
                [
                    "",
                    "## Current Safe Validation",
                    "",
                    "| Check | Result | Evidence |",
                    "| --- | --- | --- |",
                ]
            )
        lines.append(
            "| ExecuTorch tiny export availability probe | "
            f"status={executorch_smoke.get('status')}, "
            f"torch_export_ok={(executorch_smoke.get('torch_export') or {}).get('ok')}, "
            f"edge_export_ok={(executorch_smoke.get('executorch_edge_export') or {}).get('ok')} | "
            f"`{EXECUTORCH_SMOKE.as_posix()}` |"
        )
    voice_smoke = report.get("safe_validation", {}).get("voice_encoder_onnxruntime")
    if voice_smoke:
        if not smoke and not ncnn_smoke and not executorch_smoke:
            lines.extend(
                [
                    "",
                    "## Current Safe Validation",
                    "",
                    "| Check | Result | Evidence |",
                    "| --- | --- | --- |",
                ]
            )
        lines.append(
            "| Voice encoder random-weight ONNX Runtime validation | "
            f"allclose_1e_5={voice_smoke.get('allclose_1e_5')}, "
            f"max_abs_error={voice_smoke.get('max_abs_error')} | "
            f"`{VOICE_ONNX_SMOKE.as_posix()}` |"
        )
    t3_loop_summary = report.get("safe_validation", {}).get("t3_loop_summary")
    if t3_loop_summary:
        warm = (t3_loop_summary.get("requests") or [{}])[-1]
        lines.append(
            "| T3 loop telemetry guarded benchmark | "
            f"status={t3_loop_summary.get('status')}, "
            f"t3={fmt_seconds(warm.get('t3_seconds'))}, "
            f"ggml_wall={fmt_seconds((warm.get('timings') or {}).get('step_ggml_wall_seconds'))}, "
            f"sampling={fmt_seconds((warm.get('timings') or {}).get('step_sampling_seconds'))} | "
            f"`{T3_LOOP_SUMMARY.as_posix()}` |"
        )
    fast_t3_loop_summary = report.get("safe_validation", {}).get("fast_t3_loop_summary")
    if fast_t3_loop_summary:
        warm = (fast_t3_loop_summary.get("requests") or [{}])[-1]
        lines.append(
            "| Fast S3 one-step no-watermark guarded benchmark | "
            f"status={fast_t3_loop_summary.get('status')}, "
            f"wall={fmt_seconds(warm.get('wall_seconds'))}, "
            f"t3={fmt_seconds(warm.get('t3_seconds'))}, "
            f"s3={fmt_seconds(warm.get('s3_flow_seconds'))} | "
            f"`{FAST_T3_LOOP_SUMMARY.as_posix()}` |"
        )
    seen_mask_sampler = report.get("safe_validation", {}).get("t3_seen_mask_sampler")
    if seen_mask_sampler:
        lines.append(
            "| T3 seen-mask/unique-id sampler validation | "
            f"exact={seen_mask_sampler.get('overall_exact')}, "
            f"recommendation={seen_mask_sampler.get('recommended_runtime_change')} | "
            f"`{T3_SEEN_MASK_SAMPLER.as_posix()}` |"
        )
    sampler_candidates = report.get("safe_validation", {}).get("t3_sampler_candidates")
    if sampler_candidates:
        cases = sampler_candidates.get("cases") or []
        real_case = next((case for case in cases if case.get("seen_len") == 358), {})
        real_speedup = (
            (real_case.get("variants") or {})
            .get("compact_no_finite_pack", {})
            .get("speedup_vs_current")
        )
        lines.append(
            "| T3 compact distribution-equivalent sampler profile | "
            f"recommendation={sampler_candidates.get('recommendation')}, "
            f"real_seen_len_speedup={fmt_ratio(real_speedup)} | "
            f"`{T3_SAMPLER_CANDIDATES.as_posix()}` |"
        )
    native_sampler = report.get("safe_validation", {}).get("t3_native_sampler")
    if native_sampler:
        real_case = next((case for case in native_sampler.get("cases", []) if case.get("seen_len") == 358), {})
        lines.append(
            "| T3 native C++ sampler microbench | "
            f"recommendation={native_sampler.get('recommendation')}, "
            f"real_seen_len_speedup={fmt_ratio(real_case.get('native_speedup_vs_python_current'))} | "
            f"`{T3_NATIVE_SAMPLER.as_posix()}` |"
        )
    native_sampler_bridge = report.get("safe_validation", {}).get("t3_native_sampler_bridge")
    if native_sampler_bridge:
        real_case = next((case for case in native_sampler_bridge.get("cases", []) if case.get("seen_len") == 358), {})
        lines.append(
            "| T3 native sampler ctypes bridge prototype | "
            f"recommendation={native_sampler_bridge.get('recommendation')}, "
            f"real_seen_len_speedup={fmt_ratio(real_case.get('native_ctypes_speedup_vs_python_current'))} | "
            f"`{T3_NATIVE_SAMPLER_BRIDGE.as_posix()}` |"
        )
    native_sampler_real_logits = report.get("safe_validation", {}).get("t3_native_sampler_real_logits")
    if native_sampler_real_logits:
        comparisons = [
            step.get("comparison") or {}
            for step in native_sampler_real_logits.get("steps", [])
        ]
        max_abs = max(
            (item.get("max_abs_prob_diff") or 0.0 for item in comparisons),
            default=None,
        )
        lines.append(
            "| T3 native sampler real-logits distribution validation | "
            f"overall_pass={native_sampler_real_logits.get('overall_pass')}, "
            f"max_abs_prob_diff={max_abs} | "
            f"`{T3_NATIVE_SAMPLER_REAL_LOGITS.as_posix()}` |"
        )
    native_sampler_live = report.get("safe_validation", {}).get("t3_native_sampler_live")
    if native_sampler_live:
        requests = native_sampler_live.get("requests") or []
        warm = requests[-1] if requests else {}
        timing = (((warm.get("debug") or {}).get("body") or {}).get("last_request") or {})
        s3_bucket = (timing.get("s3_bucket_inference") or {}).get("selected_bucket")
        lines.append(
            "| T3 native sampler live benchmark | "
            f"status={native_sampler_live.get('status')}, "
            f"wall={fmt_seconds(warm.get('wall_seconds'))}, "
            f"t3={fmt_seconds(timing.get('t3_seconds'))}, "
            f"s3={fmt_seconds(timing.get('s3_flow_seconds'))}, "
            f"selected_bucket={s3_bucket} | "
            f"`{T3_NATIVE_SAMPLER_LIVE.as_posix()}` |"
        )
    native_sampler_padded = report.get("safe_validation", {}).get("t3_native_sampler_padded_capped")
    if native_sampler_padded:
        successes = [row for row in native_sampler_padded.get("requests", []) if row.get("status") == 200]
        first = successes[0] if successes else {}
        timing = (((first.get("debug") or {}).get("body") or {}).get("last_request") or {})
        s3_bucket = (timing.get("s3_bucket_inference") or {}).get("selected_bucket")
        lines.append(
            "| T3 native sampler padded/capped live benchmark | "
            f"status={native_sampler_padded.get('status')}, "
            f"first_success_wall={fmt_seconds(first.get('wall_seconds'))}, "
            f"first_success_s3={fmt_seconds(timing.get('s3_flow_seconds'))}, "
            f"selected_bucket={s3_bucket} | "
            f"`{T3_NATIVE_SAMPLER_PADDED_CAPPED.as_posix()}` |"
        )
    native_sampler_seeded = report.get("safe_validation", {}).get("t3_native_sampler_request_seeded")
    native_sampler_seeded_audio = report.get("safe_validation", {}).get("t3_native_sampler_request_seeded_audio")
    if native_sampler_seeded:
        requests = native_sampler_seeded.get("requests") or []
        warm = requests[-1] if requests else {}
        timing = (((warm.get("debug") or {}).get("body") or {}).get("last_request") or {})
        loop = timing.get("t3_loop_info") or {}
        s3_bucket = (timing.get("s3_bucket_inference") or {}).get("selected_bucket")
        audio = (native_sampler_seeded_audio or {}).get("audio") or {}
        comparison = (native_sampler_seeded_audio or {}).get("comparison_to_reference") or {}
        lines.append(
            "| T3 native sampler request-seeded padded/capped live benchmark | "
            f"status={native_sampler_seeded.get('status')}, "
            f"wall={fmt_seconds(warm.get('wall_seconds'))}, "
            f"t3={fmt_seconds(timing.get('t3_seconds'))}, "
            f"s3={fmt_seconds(timing.get('s3_flow_seconds'))}, "
            f"seed={loop.get('native_sampler_seed')}, "
            f"audio_passed={audio.get('passed_basic_sanity')}, "
            f"corr={comparison.get('correlation')}, "
            f"selected_bucket={s3_bucket} | "
            f"`{T3_NATIVE_SAMPLER_REQUEST_SEEDED.as_posix()}` |"
        )
    native_sampler_s3_step1 = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1")
    native_sampler_s3_step1_audio_default = report.get("safe_validation", {}).get(
        "t3_native_sampler_s3_step1_audio_default"
    )
    native_sampler_s3_step1_audio_fast = report.get("safe_validation", {}).get(
        "t3_native_sampler_s3_step1_audio_fast"
    )
    if native_sampler_s3_step1:
        requests = native_sampler_s3_step1.get("requests") or []
        warm = requests[-1] if requests else {}
        timing = (((warm.get("debug") or {}).get("body") or {}).get("last_request") or {})
        loop = timing.get("t3_loop_info") or {}
        s3_bucket = (timing.get("s3_bucket_inference") or {}).get("selected_bucket")
        audio_default = (native_sampler_s3_step1_audio_default or {}).get("audio") or {}
        comparison_default = (native_sampler_s3_step1_audio_default or {}).get("comparison_to_reference") or {}
        comparison_fast = (native_sampler_s3_step1_audio_fast or {}).get("comparison_to_reference") or {}
        lines.append(
            "| T3 native sampler request-seeded one-step S3 live benchmark | "
            f"status={native_sampler_s3_step1.get('status')}, "
            f"wall={fmt_seconds(warm.get('wall_seconds'))}, "
            f"t3={fmt_seconds(timing.get('t3_seconds'))}, "
            f"s3={fmt_seconds(timing.get('s3_flow_seconds'))}, "
            f"seed={loop.get('native_sampler_seed')}, "
            f"audio_passed={audio_default.get('passed_basic_sanity')}, "
            f"corr_default={comparison_default.get('correlation')}, "
            f"corr_fast={comparison_fast.get('correlation')}, "
            f"selected_bucket={s3_bucket} | "
            f"`{T3_NATIVE_SAMPLER_S3_STEP1.as_posix()}` |"
        )
    native_sampler_s3_step1_fused = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1_fused")
    fused_audio_basic = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1_fused_audio_basic")
    fused_audio_prior_fast = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1_fused_audio_prior_fast")
    fused_audio_default = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1_fused_audio_default")
    fused_audio_repeat = report.get("safe_validation", {}).get("t3_native_sampler_s3_step1_fused_audio_repeat")
    if native_sampler_s3_step1_fused:
        requests = native_sampler_s3_step1_fused.get("requests") or []
        warm = requests[-1] if requests else {}
        timing = (((warm.get("debug") or {}).get("body") or {}).get("last_request") or {})
        loop = timing.get("t3_loop_info") or {}
        s3_bucket = (timing.get("s3_bucket_inference") or {}).get("selected_bucket")
        estimator = timing.get("s3_estimator_calls") or {}
        audio = (fused_audio_basic or {}).get("audio") or {}
        comparison_prior = (fused_audio_prior_fast or {}).get("comparison_to_reference") or {}
        comparison_default = (fused_audio_default or {}).get("comparison_to_reference") or {}
        comparison_repeat = (fused_audio_repeat or {}).get("comparison_to_reference") or {}
        chain_types = [
            record.get("chain_type")
            for record in (estimator.get("records") or [])
        ]
        lines.append(
            "| T3 native sampler request-seeded one-step fused S3 live benchmark | "
            f"status={native_sampler_s3_step1_fused.get('status')}, "
            f"wall={fmt_seconds(warm.get('wall_seconds'))}, "
            f"t3={fmt_seconds(timing.get('t3_seconds'))}, "
            f"s3={fmt_seconds(timing.get('s3_flow_seconds'))}, "
            f"seed={loop.get('native_sampler_seed')}, "
            f"audio_passed={audio.get('passed_basic_sanity')}, "
            f"corr_prior_fast={comparison_prior.get('correlation')}, "
            f"corr_default={comparison_default.get('correlation')}, "
            f"corr_repeat={comparison_repeat.get('correlation')}, "
            f"chain_types={chain_types}, "
            f"selected_bucket={s3_bucket} | "
            f"`{T3_NATIVE_SAMPLER_S3_STEP1_FUSED.as_posix()}` |"
        )
    fast_audio = report.get("safe_validation", {}).get("fast_audio_sanity")
    if fast_audio:
        audio = fast_audio.get("audio") or {}
        comparison = fast_audio.get("comparison_to_reference") or {}
        lines.append(
            "| Fast S3 one-step no-watermark audio sanity | "
            f"passed={audio.get('passed_basic_sanity')}, "
            f"corr={comparison.get('correlation')}, "
            f"rms_diff={comparison.get('rms_diff')} | "
            f"`{FAST_AUDIO_SANITY.as_posix()}` |"
        )
    cpu_fast = report.get("safe_validation", {}).get("cpu_fast_path")
    if cpu_fast:
        lines.append(
            "| CPU fast fallback projection | "
            f"chunk270_no_watermark={fmt_seconds(cpu_fast.get('projected_no_watermark_chunk270_seconds'))}, "
            f"gain={fmt_seconds(cpu_fast.get('projected_no_watermark_gain_seconds'))} | "
            f"`{CPU_FAST_STATUS.as_posix()}` |"
        )
    s3_audit = report.get("safe_validation", {}).get("s3_bottleneck_audit")
    if s3_audit:
        api = s3_audit.get("api_debug") or {}
        active_fused = s3_audit.get("active_shape_fused_midblock") or {}
        flag_matrix = s3_audit.get("active_shape_fused_midblock_flag_matrix") or {}
        fused_chain = s3_audit.get("active_shape_fused_estimator_chain") or {}
        lines.append(
            "| S3 Vulkan bottleneck audit | "
            f"s3={fmt_seconds(api.get('s3_flow_seconds'))}, "
            f"encoder={fmt_seconds((api.get('encoder') or {}).get('vulkan_chain_fetch_seconds'))}, "
            f"estimator={fmt_seconds((api.get('estimator') or {}).get('vulkan_chain_fetch_seconds'))}, "
            f"active_t1222_credit={fmt_seconds(active_fused.get('recommended_full_s3_savings_seconds'))}, "
            f"split8_candidate={fmt_seconds(flag_matrix.get('candidate_full_s3_fetch_output_savings_seconds'))}, "
            f"fused_chain_candidate={fmt_seconds(fused_chain.get('fetch_output_seconds_for_two_s3_estimator_calls'))} | "
            f"`{S3_BOTTLENECK_AUDIT.as_posix()}` |"
        )

    perf = report["current_performance"]
    lines.extend(
        [
            "",
            "## Current Single-Worker Performance",
            "",
            f"- RTX 5090 reference chunk270: `{fmt_seconds(perf['rtx_5090_reference_seconds'])}`.",
            f"- 2x slower target: `{fmt_seconds(perf['target_2x_5090_seconds'])}`.",
            "",
            "| Path | Quality Gate | Best Server Time | Ratio To 5090 | Gap To 2x Target | Evidence |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for bench in perf["benchmarks"]:
        if not bench.get("exists"):
            lines.append(
                f"| {bench['label']} | {'default-quality' if bench['production_quality'] else 'listen-before-default'} | "
                "n/a | n/a | n/a | missing artifact |"
            )
            continue
        gap = bench.get("gap_to_2x_target_seconds")
        gap_text = fmt_seconds(gap)
        quality = "default-quality" if bench["production_quality"] else "listen-before-default"
        lines.append(
            f"| {bench['label']} | {quality} | {fmt_seconds(bench.get('best_total_seconds'))} | "
            f"{fmt_ratio(bench.get('ratio_to_5090'))} | {gap_text} | `{bench['path']}` |"
        )

    lines.extend(
        [
            "",
            "## Deployment Controls",
            "",
            "| Control | Status | Evidence |",
            "| --- | --- | --- |",
            (
                "| Safe CPU API | Keep running on `:8000`; do not restart unless intentionally picking up code changes | "
                "`./run_api.sh` defaults `CHATTERBOX_APPLY_WATERMARK=1` |"
            ),
            (
                "| CPU fast fallback | Available on intentional separate launch, best measured CPU threads plus no watermark | "
                "`./run_api_cpu_fast.sh` defaults `PORT=8002`, `CHATTERBOX_APPLY_WATERMARK=0` |"
            ),
            (
                "| Experimental Vulkan API | Opt-in worker; no-watermark default accepted for speed | "
                "`./run_api_vulkan_hybrid.sh` defaults `CHATTERBOX_APPLY_WATERMARK=0` |"
            ),
            (
                "| Experimental Vulkan fast API | Opt-in listen-before-default worker matching the 6.944s recipe | "
                f"`{VULKAN_FAST_SCRIPT.as_posix()}` sets one-step S3, native T3 sampler, padded required Vulkan S3, and no watermark |"
            ),
            (
                "| Experimental Vulkan fast fused API | Opt-in listen-before-default worker matching the 6.903s recipe | "
                f"`{VULKAN_FAST_FUSED_SCRIPT.as_posix()}` adds fused split8 S3 midblocks to the fast recipe |"
            ),
            (
                "| Multi-GPU throughput router | Available; single-flight per backend | "
                f"`{ROUTER_API.as_posix()}`, `{ROUTER_SCRIPT.as_posix()}` |"
            ),
            (
                "| Non-generating status snapshot | Available; checks ports, health, resources, GPU nodes, and saved timings | "
                f"`{STATUS_SCRIPT.as_posix()} --pretty` |"
            ),
            (
                "| Vulkan worker preflight | Available; profile-aware non-generating gate for default-quality or fast-target workers | "
                "`./preflight_vulkan_worker.py --profile default-quality`, `--profile fast-target`, or `--profile fast-fused-target` |"
            ),
            (
                "| T3 loop telemetry benchmark | Available; dry-run by default, guarded run with `--run` | "
                f"`{T3_LOOP_INFO_TRIAL.as_posix()}` |"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Chatterbox Components",
            "",
            "| Component | Best Local Runtime | Status | Evidence | Next Useful Move |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    t3_symbols_ok = all(
        all(item["required_symbols"].values()) for item in report["t3"]["bridges"] if item["exists"]
    )
    s3_all_present = all(item["missing_count"] == 0 for item in report["s3"]["buckets"].values())
    hift_active = report["hift"]["active"]
    hift_ok = hift_active["t128"]["exists"] and hift_active["t96"]["exists"]
    voice_onnx_validated = report.get("safe_validation", {}).get("voice_encoder_onnxruntime") is not None
    executorch_probe = report.get("safe_validation", {}).get("executorch_tiny_export") or {}
    executorch_status = executorch_probe.get("status", "missing_probe")
    executorch_torch_export_ok = (executorch_probe.get("torch_export") or {}).get("ok")
    lines.extend(
        [
            (
                "| T3 token model | ggml + Vulkan | Active experimental path | "
                f"weights={report['t3']['weights']['exists']}, symbols_ok={t3_symbols_ok} | "
                "Request-seeded native sampling is stable in guarded runs. Combined with one-step fused S3 it crosses the 2x target at 6.903s, but remains opt-in/listen-before-default because waveform output differs from default and the fused variant still needs listening validation. |"
            ),
            (
                "| T3 full-sequence/prefill | PyTorch CPU + prefix cache | Partially optimized | "
                "Prefix cache and startup prewarm are implemented; IREE one-token prefill was slower than CPU. | "
                "Avoid repeating one-token IREE prefill; investigate true full-sequence export only if bounded. |"
            ),
            (
                "| S3 encoder/estimator | IREE + Vulkan | Active for exact buckets | "
                f"configured_buckets={len(report['s3']['buckets'])}, all_required_vmfb_present={s3_all_present} | "
                "Telemetry shows estimator execution dominates S3; active-shape split8 midblock fusion validates across all midblocks, wins in the fixed-shape fused estimator-chain benchmark, and is now integrated behind an opt-in API flag. Production/default credit still requires listening validation. Prioritize larger estimator fusion or dispatch reduction. Export exact buckets only to prevent real CPU fallback. |"
            ),
            (
                "| HiFT vocoder core | IREE + Vulkan | Active chunked path | "
                f"t128={hift_active['t128']['exists']}, compact_t96={hift_active['t96']['exists']}, "
                f"t722_rejected_probe={report['hift']['rejected_t722_probe_exists']} | "
                "Do not retry t722 whole-clip shape; keep chunked 128 plus compact96 tail. |"
            ),
            (
                "| Voice encoder / conditionals | ONNX export path only | Not in active Vulkan path | "
                f"onnx_module_available={tools['python_modules']['onnx']}, "
                f"onnxruntime={tools['python_modules']['onnxruntime']}, "
                f"random_weight_validation={voice_onnx_validated} | "
                "Leave as CPU unless profiling shows conditionals matter for steady-state requests. |"
            ),
            (
                f"| ncnn candidate path | ncnn/pnnx | {'CPU conversion validated, Vulkan rejected' if ncnn_ready else 'Blocked by missing local runtime'} | "
                f"ncnn={tools['python_modules']['ncnn']}, pnnx={bool(tools['commands']['pnnx'])} | "
                + (
                    "Do not use ncnn Vulkan for Chatterbox on this host; use pnnx only for CPU conversion/export inspection. |"
                    if ncnn_ready
                    else "Install ncnn/pnnx before spending export time here. |"
                )
            ),
            (
                f"| ExecuTorch candidate path | ExecuTorch | {executorch_status} | "
                f"executorch={tools['python_modules']['executorch']}, exir={tools['python_modules']['executorch.exir']}, "
                f"torch_export_tiny_ok={executorch_torch_export_ok} | "
                "Do not install in this env unless a separate disposable venv/container is created. |"
            ),
            (
                "| Request-level parallelism | Router over one worker per BC-250 | Available for throughput, not single-chunk latency | "
                f"router_exists={ROUTER_API.exists()}, launcher_exists={ROUTER_SCRIPT.exists()} | "
                "Use for independent requests or long-text chunks; keep one request in flight per worker. |"
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Do not use ROCm/HIP on the BC-250.",
            "- Keep the safe CPU API on port `8000` running while probing.",
            "- Do not retry the 722-frame whole-clip HiFT Vulkan shape; it previously triggered device loss.",
            "- Do not repeat ncnn Vulkan probes on this host; the tiny graph already segfaulted.",
            "- Do not install ExecuTorch into the stable venv; use a disposable environment if revisiting it.",
            "- Prefer bounded isolated probes over full-pipeline runtime experiments.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    return OUT_MD.as_posix()


def main() -> None:
    BENCH.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "description": "Local runtime/export candidate audit without model load or server start.",
        "tooling": {
            "python_modules": {
                name: module_available(name)
                for name in [
                    "iree",
                    "iree.runtime",
                    "iree.turbine",
                    "ncnn",
                    "executorch",
                    "executorch.exir",
                    "onnx",
                    "onnxruntime",
                ]
            },
            "commands": {
                name: command_available(name)
                for name in [
                    "iree-compile",
                    "iree-run-module",
                    "ncnnoptimize",
                    "pnnx",
                    "executorch",
                ]
            },
        },
        "t3": {
            "weights": {
                "path": T3_WEIGHTS.as_posix(),
                "exists": T3_WEIGHTS.exists(),
                "manifest_exists": (T3_WEIGHTS / "manifest.json").exists(),
                "f32_file_count": len(list(T3_WEIGHTS.glob("*.f32"))) if T3_WEIGHTS.exists() else 0,
                "size_mb": dir_size_mb(T3_WEIGHTS) if T3_WEIGHTS.exists() else None,
            },
            "bridges": [bridge_symbols(path) for path in T3_LIBS],
        },
        "s3": {
            "buckets": {str(key): s3_bucket_artifacts(bucket) for key, bucket in S3_BUCKETS.items()},
            "artifact_root_size_mb": dir_size_mb(EXPORTS / "s3_flow_vulkan_components"),
        },
        "hift": hift_artifacts(),
        "safe_validation": {
            "iree_vulkan_snake_smoke": load_json(IREE_SMOKE),
            "ncnn_pnnx_cpu_smoke": load_json(NCNN_SMOKE),
            "executorch_tiny_export": load_json(EXECUTORCH_SMOKE),
            "voice_encoder_onnxruntime": load_json(VOICE_ONNX_SMOKE),
            "t3_loop_summary": load_json(T3_LOOP_SUMMARY),
            "fast_t3_loop_summary": load_json(FAST_T3_LOOP_SUMMARY),
            "t3_seen_mask_sampler": load_json(T3_SEEN_MASK_SAMPLER),
            "t3_sampler_candidates": load_json(T3_SAMPLER_CANDIDATES),
            "t3_native_sampler": load_json(T3_NATIVE_SAMPLER),
            "t3_native_sampler_bridge": load_json(T3_NATIVE_SAMPLER_BRIDGE),
            "t3_native_sampler_real_logits": load_json(T3_NATIVE_SAMPLER_REAL_LOGITS),
            "t3_native_sampler_live": load_json(T3_NATIVE_SAMPLER_LIVE),
            "t3_native_sampler_padded_capped": load_json(T3_NATIVE_SAMPLER_PADDED_CAPPED),
            "t3_native_sampler_request_seeded": load_json(T3_NATIVE_SAMPLER_REQUEST_SEEDED),
            "t3_native_sampler_request_seeded_audio": load_json(T3_NATIVE_SAMPLER_REQUEST_SEEDED_AUDIO),
            "t3_native_sampler_s3_step1": load_json(T3_NATIVE_SAMPLER_S3_STEP1),
            "t3_native_sampler_s3_step1_fused": load_json(T3_NATIVE_SAMPLER_S3_STEP1_FUSED),
            "t3_native_sampler_s3_step1_audio_default": load_json(T3_NATIVE_SAMPLER_S3_STEP1_AUDIO_DEFAULT),
            "t3_native_sampler_s3_step1_audio_fast": load_json(T3_NATIVE_SAMPLER_S3_STEP1_AUDIO_FAST),
            "t3_native_sampler_s3_step1_fused_audio_basic": load_json(T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_BASIC),
            "t3_native_sampler_s3_step1_fused_audio_prior_fast": load_json(T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_PRIOR_FAST),
            "t3_native_sampler_s3_step1_fused_audio_default": load_json(T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_DEFAULT),
            "t3_native_sampler_s3_step1_fused_audio_repeat": load_json(T3_NATIVE_SAMPLER_S3_STEP1_FUSED_AUDIO_REPEAT),
            "fast_audio_sanity": load_json(FAST_AUDIO_SANITY),
            "latest_projection": load_json(LATEST_PROJECTION),
            "cpu_fast_path": load_json(CPU_FAST_STATUS),
            "s3_bottleneck_audit": load_json(S3_BOTTLENECK_AUDIT),
        },
        "current_performance": current_performance(),
        "notes": [
            "This audit intentionally does not import the full Chatterbox model.",
            "No ROCm/HIP command is run.",
            "ncnn and ExecuTorch are only reported as available if installed locally.",
            "ExecuTorch is recorded via a tiny CPU-only probe; it is not installed in the stable venv.",
            "ncnn Vulkan is not treated as viable on this host after a tiny smoke graph segfaulted.",
            "Current performance is read from saved benchmark artifacts; this audit performs no audio generation.",
        ],
    }
    report["markdown"] = write_markdown(report)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")


if __name__ == "__main__":
    main()
