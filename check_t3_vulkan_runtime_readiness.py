#!/usr/bin/env python3
"""Check readiness for assembling the BC-250 Vulkan T3 runtime."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
T3 = EXPORTS / "t3_exportability"
MASKED = T3 / "masked_cache"
CACHE_UPDATE = T3 / "cache_update"
BENCHMARKS = EXPORTS / "benchmarks"

MASKED_CHUNK_TEMPLATE = "gpt2_masked_cache_stack_s{start}_l4_p128_valid42_t1_vulkan_gfx1013.vmfb"
MASKED_CHUNK_MLIR_TEMPLATE = "gpt2_masked_cache_stack_s{start}_l4_p128_valid42_t1.mlir"
P1024_CHUNK_TEMPLATE = "gpt2_masked_cache_stack_s{start}_l4_p1024_valid935_t1_vulkan_gfx1013.vmfb"
P1024_CHUNK_MLIR_TEMPLATE = "gpt2_masked_cache_stack_s{start}_l4_p1024_valid935_t1.mlir"
CHUNK_STARTS = (0, 4, 8, 12, 16, 20)


def size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def human(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {
                "ok": response.status == 200,
                "status": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "size_bytes": size(path),
        "size": human(size(path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS / "t3_vulkan_runtime_readiness_2026-07-08.json",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:8000/health",
        help="Safe CPU API health endpoint to check.",
    )
    args = parser.parse_args()

    disk = shutil.disk_usage(ROOT)
    cleanup_conservative = load_json(T3 / "artifact_cleanup_conservative_2026-07-08-v2.json")
    cleanup_aggressive = load_json(T3 / "artifact_cleanup_aggressive_2026-07-08-v2.json")
    projection = load_json(BENCHMARKS / "t3_vulkan_runtime_projection_2026-07-08.json")

    masked_chunks = []
    p1024_chunks = []
    for start in CHUNK_STARTS:
        vmfb = MASKED / MASKED_CHUNK_TEMPLATE.format(start=start)
        mlir = MASKED / MASKED_CHUNK_MLIR_TEMPLATE.format(start=start)
        masked_chunks.append(
            {
                "start_layer": start,
                "end_layer": start + 4,
                "vmfb": artifact(vmfb),
                "mlir": artifact(mlir),
            }
        )
        p1024_vmfb = MASKED / P1024_CHUNK_TEMPLATE.format(start=start)
        p1024_mlir = MASKED / P1024_CHUNK_MLIR_TEMPLATE.format(start=start)
        p1024_chunks.append(
            {
                "start_layer": start,
                "end_layer": start + 4,
                "vmfb": artifact(p1024_vmfb),
                "mlir": artifact(p1024_mlir),
            }
        )

    existing_chunk_vmfbs = [chunk for chunk in masked_chunks if chunk["vmfb"]["exists"]]
    missing_chunk_vmfbs = [chunk for chunk in masked_chunks if not chunk["vmfb"]["exists"]]
    exemplar_chunk = next((chunk for chunk in masked_chunks if chunk["vmfb"]["exists"]), None)
    exemplar_vmfb_size = exemplar_chunk["vmfb"]["size_bytes"] if exemplar_chunk else 201_702_428
    exemplar_mlir_size = exemplar_chunk["mlir"]["size_bytes"] if exemplar_chunk else 403_148_734
    missing_count = len(missing_chunk_vmfbs)
    estimated_missing_vmfb_bytes = missing_count * exemplar_vmfb_size
    estimated_missing_mlir_bytes = missing_count * exemplar_mlir_size
    estimated_missing_with_mlir_bytes = estimated_missing_vmfb_bytes + estimated_missing_mlir_bytes

    required = {
        "masked_chunks": masked_chunks,
        "cache_update_slot_vmfb": artifact(
            CACHE_UPDATE / "t3_kv_cache_slot_update_p128_t1_vulkan_gfx1013.vmfb"
        ),
        "speech_head_vmfb": artifact(T3 / "speech_head_t1_vulkan_gfx1013.vmfb"),
        "final_norm_speech_head_vmfb": artifact(T3 / "t3_final_norm_speech_head_t1_vulkan_gfx1013.vmfb"),
    }
    p1024_missing_chunks = [chunk for chunk in p1024_chunks if not chunk["vmfb"]["exists"]]
    real_bucket_required = {
        "masked_chunks_p1024": p1024_chunks,
        "cache_update_slot_p1024_vmfb": artifact(
            CACHE_UPDATE / "t3_kv_cache_slot_update_p1024_t1_vulkan_gfx1013.vmfb"
        ),
        "full_loop_p1024_32step_report": artifact(
            MASKED / "t3_full_masked_vulkan_loop_p1024_32step_2026-07-08.json"
        ),
        "full_loop_p1024_finalhead_report": artifact(
            MASKED / "t3_full_masked_vulkan_loop_p1024_finalhead_8step_2026-07-08.json"
        ),
        "real_prefill_followon_report": artifact(
            MASKED / "t3_real_prefill_vulkan_followon_chunk270_finalhead_32step_fetchlogits_2026-07-08.json"
        ),
        "prompt_length_budget": artifact(T3 / "t3_real_prompt_length_budget_2026-07-08.json"),
    }
    all_core_runtime_ready = (
        missing_count == 0
        and required["cache_update_slot_vmfb"]["exists"]
        and required["speech_head_vmfb"]["exists"]
        and required["final_norm_speech_head_vmfb"]["exists"]
    )
    real_bucket_runtime_ready = (
        len(p1024_missing_chunks) == 0
        and real_bucket_required["cache_update_slot_p1024_vmfb"]["exists"]
        and real_bucket_required["full_loop_p1024_32step_report"]["exists"]
        and real_bucket_required["full_loop_p1024_finalhead_report"]["exists"]
        and real_bucket_required["real_prefill_followon_report"]["exists"]
        and real_bucket_required["prompt_length_budget"]["exists"]
        and required["final_norm_speech_head_vmfb"]["exists"]
    )

    report = {
        "health": http_json(args.health_url),
        "disk": {
            "root": ROOT.as_posix(),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "total": human(disk.total),
            "used": human(disk.used),
            "free": human(disk.free),
            "percent_used": round(disk.used * 100.0 / disk.total, 2),
        },
        "required_runtime_artifacts": required,
        "real_bucket_runtime_artifacts": real_bucket_required,
        "readiness": {
            "masked_chunks_ready": len(existing_chunk_vmfbs),
            "masked_chunks_required": len(CHUNK_STARTS),
            "missing_masked_chunks": [
                {"start_layer": chunk["start_layer"], "end_layer": chunk["end_layer"]}
                for chunk in missing_chunk_vmfbs
            ],
            "cache_update_ready": required["cache_update_slot_vmfb"]["exists"],
            "speech_head_ready": required["speech_head_vmfb"]["exists"],
            "final_norm_speech_head_ready": required["final_norm_speech_head_vmfb"]["exists"],
            "all_core_runtime_ready": all_core_runtime_ready,
            "p1024_masked_chunks_ready": len(CHUNK_STARTS) - len(p1024_missing_chunks),
            "p1024_masked_chunks_required": len(CHUNK_STARTS),
            "p1024_missing_masked_chunks": [
                {"start_layer": chunk["start_layer"], "end_layer": chunk["end_layer"]}
                for chunk in p1024_missing_chunks
            ],
            "p1024_cache_update_ready": real_bucket_required["cache_update_slot_p1024_vmfb"]["exists"],
            "p1024_full_loop_report_ready": real_bucket_required["full_loop_p1024_32step_report"]["exists"],
            "p1024_finalhead_report_ready": real_bucket_required["full_loop_p1024_finalhead_report"]["exists"],
            "real_prefill_followon_report_ready": real_bucket_required["real_prefill_followon_report"]["exists"],
            "prompt_length_budget_ready": real_bucket_required["prompt_length_budget"]["exists"],
            "real_bucket_runtime_ready": real_bucket_runtime_ready,
        },
        "compile_budget_estimate": {
            "exemplar_vmfb_size_bytes": exemplar_vmfb_size,
            "exemplar_vmfb_size": human(exemplar_vmfb_size),
            "exemplar_mlir_size_bytes": exemplar_mlir_size,
            "exemplar_mlir_size": human(exemplar_mlir_size),
            "missing_chunk_count": missing_count,
            "estimated_missing_vmfb_bytes": estimated_missing_vmfb_bytes,
            "estimated_missing_vmfb": human(estimated_missing_vmfb_bytes),
            "estimated_missing_mlir_bytes": estimated_missing_mlir_bytes,
            "estimated_missing_mlir": human(estimated_missing_mlir_bytes),
            "estimated_missing_with_mlir_bytes": estimated_missing_with_mlir_bytes,
            "estimated_missing_with_mlir": human(estimated_missing_with_mlir_bytes),
            "current_free_after_missing_with_mlir_bytes": disk.free - estimated_missing_with_mlir_bytes,
            "current_free_after_missing_vmfbs_only_bytes": disk.free - estimated_missing_vmfb_bytes,
            "can_store_missing_vmfbs_only_now": disk.free > estimated_missing_vmfb_bytes,
            "can_store_missing_vmfbs_and_mlir_now": disk.free > estimated_missing_with_mlir_bytes,
        },
        "cleanup": {
            "conservative_report": (T3 / "artifact_cleanup_conservative_2026-07-08-v2.json").as_posix(),
            "conservative_reclaim_bytes": cleanup_conservative["summary"]["bytes"] if cleanup_conservative else None,
            "conservative_reclaim": cleanup_conservative["summary"]["human"] if cleanup_conservative else None,
            "aggressive_report": (T3 / "artifact_cleanup_aggressive_2026-07-08-v2.json").as_posix(),
            "aggressive_reclaim_bytes": cleanup_aggressive["summary"]["bytes"] if cleanup_aggressive else None,
            "aggressive_reclaim": cleanup_aggressive["summary"]["human"] if cleanup_aggressive else None,
        },
        "projection": {
            "path": (BENCHMARKS / "t3_vulkan_runtime_projection_2026-07-08.json").as_posix(),
            "optimistic_full_api_270_s": (
                projection["projection"]["optimistic_full_api_270_s_with_vulkan_t3_and_hift"]
                if projection
                else None
            ),
            "optimistic_slowdown_vs_5090": (
                projection["projection"]["optimistic_slowdown_vs_5090"] if projection else None
            ),
            "python_loop_full_api_270_s": (
                projection["projection"]["python_loop_full_api_270_s_with_vulkan_t3_and_hift"]
                if projection
                else None
            ),
            "python_loop_slowdown_vs_5090": (
                projection["projection"]["python_loop_slowdown_vs_5090"] if projection else None
            ),
        },
        "next_actions": [
            "Do not use ROCm/HIP on this BC-250 host.",
            "Run artifact cleanup or expand disk before compiling the five missing masked 4-layer chunks.",
            "Compile masked chunks for start layers 4, 8, 12, 16, and 20.",
            "Assemble the full six-chunk token loop with fixed-slot cache updates and speech-head logits.",
            "Validate generated-token drift before wiring any Vulkan T3 path into the API.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"safe_api_ok={report['health'].get('ok')}")
    print(
        f"masked_chunks_ready={len(existing_chunk_vmfbs)}/{len(CHUNK_STARTS)} "
        f"missing={missing_count}"
    )
    print(f"disk_free={report['disk']['free']} used={report['disk']['percent_used']}%")
    print(
        "estimated_missing_with_mlir="
        f"{report['compile_budget_estimate']['estimated_missing_with_mlir']}"
    )
    print(
        "estimated_missing_vmfbs_only="
        f"{report['compile_budget_estimate']['estimated_missing_vmfb']}"
    )
    print(
        "core_runtime_ready="
        f"{report['readiness']['all_core_runtime_ready']}"
    )
    print(
        "real_bucket_runtime_ready="
        f"{report['readiness']['real_bucket_runtime_ready']} "
        f"p1024_chunks={report['readiness']['p1024_masked_chunks_ready']}/"
        f"{report['readiness']['p1024_masked_chunks_required']}"
    )


if __name__ == "__main__":
    main()
