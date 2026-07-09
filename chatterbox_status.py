#!/usr/bin/env python3
"""Non-generating status snapshot for the local Chatterbox workbench."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/root/chatterbox")
BENCHMARKS = [
    ROOT / "exports/benchmarks/vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_t3_loop_info_no_watermark_default_quality_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_s3_debug_records_no_watermark_default_quality_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_native_sampler_no_watermark_default_quality_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_native_sampler_padded_s3_capped_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_native_sampler_request_seeded_padded_s3_capped_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_no_watermark_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_guarded_2026-07-08.json",
    ROOT / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_s3_step1_guarded_2026-07-08.json",
]
CPU_FAST_STATUS = ROOT / "exports/cpu_thread_bench/cpu_fast_path_status_2026-07-08.json"
PORTS = (8000, 8003, 8004, 8010, 4123)
RTX_5090_REFERENCE_SECONDS = 3.495
TARGET_2X_5090_SECONDS = RTX_5090_REFERENCE_SECONDS * 2.0


def gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def run(cmd: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return {"ok": True, "status": resp.status, "body": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def port_snapshot() -> dict[str, Any]:
    result = run(["ss", "-ltnp"])
    lines = []
    for line in result["stdout"].splitlines():
        if any(f":{port} " in line or f":{port}\t" in line for port in PORTS):
            lines.append(line)
    return {"ports": PORTS, "listeners": lines, "ss_error": result["stderr"] if result["returncode"] else ""}


def meminfo() -> dict[str, Any]:
    data: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw_value = line.split(":", 1)
            value = raw_value.strip().split()[0]
            data[key] = int(value) * 1024
    except Exception:
        return {}
    keys = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    return {key: gb(data[key]) for key in keys if key in data}


def diskinfo() -> dict[str, float]:
    usage = shutil.disk_usage(ROOT)
    return {"total_gb": gb(usage.total), "used_gb": gb(usage.used), "free_gb": gb(usage.free)}


def gpu_snapshot() -> dict[str, Any]:
    dri = Path("/dev/dri")
    nodes = sorted(path.name for path in dri.iterdir()) if dri.exists() else []
    cards = []
    for path in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        device = path / "device"
        uevent = device / "uevent"
        info: dict[str, Any] = {"name": path.name}
        if uevent.exists():
            for line in uevent.read_text(errors="replace").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key in {"PCI_ID", "DRIVER", "PCI_SLOT_NAME"}:
                        info[key.lower()] = value
        cards.append(info)
    return {"dev_dri": nodes, "kfd_present": Path("/dev/kfd").exists(), "drm_cards": cards}


def request_timing(req: dict[str, Any]) -> dict[str, Any]:
    last_request = req.get("debug", {}).get("body", {}).get("last_request", {}) if req.get("status") == 200 else {}
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
        "raw_t3_tokens": last_request.get("raw_t3_tokens"),
        "s3_bucket_inference": last_request.get("s3_bucket_inference"),
        "s3_encoder_calls": last_request.get("s3_encoder_calls"),
        "s3_estimator_calls": last_request.get("s3_estimator_calls"),
        "t3_loop_info": last_request.get("t3_loop_info"),
    }


def benchmark_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"path": str(path), "exists": True, "error": str(exc)}
    requests = [request_timing(req) for req in data.get("requests", [])]
    totals = [req["total_seconds"] for req in requests if isinstance(req.get("total_seconds"), (int, float))]
    return {
        "path": str(path),
        "exists": True,
        "artifact_label": data.get("artifact_label"),
        "description": data.get("description"),
        "env_overrides": data.get("env_overrides", {}),
        "request_count": len(requests),
        "best_total_seconds": min(totals) if totals else None,
        "last_total_seconds": totals[-1] if totals else None,
        "last_request": requests[-1] if requests else None,
    }


def performance_target_summary(benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        item
        for item in benchmarks
        if item.get("exists") and isinstance(item.get("best_total_seconds"), (int, float))
    ]
    best = min(measured, key=lambda item: float(item["best_total_seconds"])) if measured else None
    if best is None:
        return {
            "rtx_5090_reference_seconds": RTX_5090_REFERENCE_SECONDS,
            "target_2x_5090_seconds": TARGET_2X_5090_SECONDS,
            "best_saved_benchmark": None,
            "target_met": False,
        }
    best_total = float(best["best_total_seconds"])
    return {
        "rtx_5090_reference_seconds": RTX_5090_REFERENCE_SECONDS,
        "target_2x_5090_seconds": TARGET_2X_5090_SECONDS,
        "best_saved_benchmark": {
            "path": best.get("path"),
            "artifact_label": best.get("artifact_label"),
            "best_total_seconds": best_total,
            "gap_to_2x_target_seconds": best_total - TARGET_2X_5090_SECONDS,
            "ratio_to_5090": best_total / RTX_5090_REFERENCE_SECONDS,
            "listen_before_default": best.get("artifact_label")
            == "native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark",
        },
        "target_met": best_total <= TARGET_2X_5090_SECONDS,
    }


def cpu_fast_summary() -> dict[str, Any]:
    if not CPU_FAST_STATUS.exists():
        return {"path": str(CPU_FAST_STATUS), "exists": False}
    try:
        data = json.loads(CPU_FAST_STATUS.read_text())
    except Exception as exc:
        return {"path": str(CPU_FAST_STATUS), "exists": True, "error": str(exc)}
    return {
        "path": str(CPU_FAST_STATUS),
        "exists": True,
        "launcher": data.get("launcher"),
        "best_thread_setting": data.get("best_thread_setting"),
        "projected_no_watermark_chunk270_seconds": data.get("projected_no_watermark_chunk270_seconds"),
        "projected_no_watermark_gain_seconds": data.get("projected_no_watermark_gain_seconds"),
        "projected_no_watermark_gain_percent": data.get("projected_no_watermark_gain_percent"),
        "safe_api_policy": data.get("safe_api_policy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true", help="print indented JSON")
    args = parser.parse_args()

    benchmarks = [benchmark_summary(path) for path in BENCHMARKS]
    snapshot = {
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "note": "No audio generation is performed by this script.",
        "ports": port_snapshot(),
        "health": {
            str(port): http_json(f"http://127.0.0.1:{port}/health")
            for port in PORTS
            if port != 4123
        },
        "resources": {"disk_root": diskinfo(), "memory": meminfo()},
        "gpu": gpu_snapshot(),
        "benchmarks": benchmarks,
        "performance_target": performance_target_summary(benchmarks),
        "cpu_fast_path": cpu_fast_summary(),
        "watermark_defaults": {
            "safe_cpu_launcher": "CHATTERBOX_APPLY_WATERMARK defaults to 1",
            "cpu_fast_launcher": "CHATTERBOX_APPLY_WATERMARK defaults to 0 and PORT defaults to 8002",
            "vulkan_hybrid_launcher": "CHATTERBOX_APPLY_WATERMARK defaults to 0",
            "vulkan_fast_launcher": "CHATTERBOX_APPLY_WATERMARK defaults to 0 with one-step S3 and native T3 sampler",
            "vulkan_fast_fused_launcher": "CHATTERBOX_APPLY_WATERMARK defaults to 0 with one-step S3, native T3 sampler, and fused split8 S3 midblocks",
        },
    }
    print(json.dumps(snapshot, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
