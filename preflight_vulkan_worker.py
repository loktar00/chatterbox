#!/usr/bin/env python3
"""Non-generating Vulkan worker preflight for BC-250 Chatterbox workers.

This intentionally avoids ROCm/HIP tools. It checks the pieces that matter
before starting a Vulkan worker: device selection, resource headroom, port state,
selected launch profile, and whether the safe CPU API is still alive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/root/chatterbox")
DEFAULT_PORTS = (8000, 8003, 8004, 8010, 4123)
FAST_LAUNCHER = ROOT / "run_api_vulkan_fast.sh"
FAST_FUSED_LAUNCHER = ROOT / "run_api_vulkan_fast_fused.sh"
HYBRID_LAUNCHER = ROOT / "run_api_vulkan_hybrid.sh"
BENCHMARKS = {
    "target_crossing_native_sampler_s3_step1_no_watermark": ROOT
    / "exports/benchmarks/vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_no_watermark_guarded_2026-07-08.json",
    "target_crossing_native_sampler_s3_step1_fused_midblocks_no_watermark": ROOT
    / "exports/benchmarks/vulkan_hybrid_api_native_sampler_request_seeded_s3_step1_fused_midblocks_no_watermark_guarded_2026-07-08.json",
    "fast_s3_step1_no_watermark": ROOT
    / "exports/benchmarks/vulkan_hybrid_api_s3_step1_no_watermark_t3_loop_info_guarded_2026-07-08.json",
    "default_quality_no_watermark": ROOT
    / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_no_watermark_default_quality_guarded_2026-07-08.json",
    "default_quality_with_watermark": ROOT
    / "exports/benchmarks/vulkan_hybrid_api_32_enabled_24_reported_guarded_2026-07-08.json",
}
CPU_STATUS = ROOT / "exports/cpu_thread_bench/cpu_runtime_status_2026-07-08.json"
RTX_5090_REFERENCE_SECONDS = 3.495
TARGET_2X_5090_SECONDS = RTX_5090_REFERENCE_SECONDS * 2.0
PROFILES = {
    "default-quality": {
        "description": "default-quality Vulkan worker, no watermark",
        "launcher": HYBRID_LAUNCHER,
        "benchmark": "default_quality_no_watermark",
        "launcher_defaults": {
            "CHATTERBOX_APPLY_WATERMARK": "0",
            "CHATTERBOX_S3_TIMESTEPS": "2",
            "CHATTERBOX_T3_NATIVE_SAMPLER": "0",
            "CHATTERBOX_REQUIRE_VULKAN_S3": "0",
            "CHATTERBOX_T3_MAX_GEN_LEN": "420",
        },
        "benchmark_env": {"CHATTERBOX_APPLY_WATERMARK": "0"},
        "max_best_total_seconds": None,
    },
    "fast-target": {
        "description": "listen-before-default fast Vulkan worker matching the 6.944s recipe",
        "launcher": FAST_LAUNCHER,
        "benchmark": "target_crossing_native_sampler_s3_step1_no_watermark",
        "launcher_defaults": {
            "CHATTERBOX_APPLY_WATERMARK": "0",
            "CHATTERBOX_S3_TIMESTEPS": "1",
            "CHATTERBOX_T3_NATIVE_SAMPLER": "1",
            "CHATTERBOX_ALLOW_VULKAN_S3_PADDING": "1",
            "CHATTERBOX_REQUIRE_VULKAN_S3": "1",
            "CHATTERBOX_T3_MAX_GEN_LEN": "376",
            "CHATTERBOX_DEFAULT_SEED": "20260708",
        },
        "benchmark_env": {
            "CHATTERBOX_APPLY_WATERMARK": "0",
            "CHATTERBOX_S3_TIMESTEPS": "1",
            "CHATTERBOX_T3_NATIVE_SAMPLER": "1",
            "CHATTERBOX_ALLOW_VULKAN_S3_PADDING": "1",
            "CHATTERBOX_REQUIRE_VULKAN_S3": "1",
            "CHATTERBOX_T3_MAX_GEN_LEN": "376",
        },
        "max_best_total_seconds": TARGET_2X_5090_SECONDS,
    },
    "fast-fused-target": {
        "description": "listen-before-default fast Vulkan worker with fused split8 S3 midblocks",
        "launcher": FAST_FUSED_LAUNCHER,
        "benchmark": "target_crossing_native_sampler_s3_step1_fused_midblocks_no_watermark",
        "launcher_defaults": {
            "CHATTERBOX_APPLY_WATERMARK": "0",
            "CHATTERBOX_S3_TIMESTEPS": "1",
            "CHATTERBOX_T3_NATIVE_SAMPLER": "1",
            "CHATTERBOX_ALLOW_VULKAN_S3_PADDING": "1",
            "CHATTERBOX_REQUIRE_VULKAN_S3": "1",
            "CHATTERBOX_T3_MAX_GEN_LEN": "376",
            "CHATTERBOX_DEFAULT_SEED": "20260708",
            "CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS": "1",
            "CHATTERBOX_VULKAN_S3_FUSED_VARIANT": "split8",
        },
        "benchmark_env": {
            "CHATTERBOX_APPLY_WATERMARK": "0",
            "CHATTERBOX_S3_TIMESTEPS": "1",
            "CHATTERBOX_T3_NATIVE_SAMPLER": "1",
            "CHATTERBOX_ALLOW_VULKAN_S3_PADDING": "1",
            "CHATTERBOX_REQUIRE_VULKAN_S3": "1",
            "CHATTERBOX_T3_MAX_GEN_LEN": "376",
            "CHATTERBOX_DEFAULT_SEED": "20260708",
            "CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS": "1",
            "CHATTERBOX_VULKAN_S3_FUSED_VARIANT": "split8",
        },
        "max_best_total_seconds": TARGET_2X_5090_SECONDS,
    },
}


def gb(value: int | float) -> float:
    return round(float(value) / (1024**3), 3)


def run(cmd: list[str], *, env: dict[str, str] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
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


def meminfo() -> dict[str, float]:
    data: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw_value = line.split(":", 1)
            data[key] = int(raw_value.strip().split()[0]) * 1024
    except Exception:
        return {}
    return {
        "total_gb": gb(data.get("MemTotal", 0)),
        "available_gb": gb(data.get("MemAvailable", 0)),
        "swap_total_gb": gb(data.get("SwapTotal", 0)),
        "swap_free_gb": gb(data.get("SwapFree", 0)),
    }


def diskinfo() -> dict[str, float]:
    usage = shutil.disk_usage(ROOT)
    return {"total_gb": gb(usage.total), "used_gb": gb(usage.used), "free_gb": gb(usage.free)}


def port_snapshot() -> dict[str, Any]:
    result = run(["ss", "-ltnp"], timeout=5.0)
    listeners = []
    for line in result["stdout"].splitlines():
        if any(f":{port} " in line or f":{port}\t" in line for port in DEFAULT_PORTS):
            listeners.append(line)
    return {"listeners": listeners, "error": result["stderr"] if result["returncode"] else ""}


def port_is_listening(snapshot: dict[str, Any], port: int) -> bool:
    return any(f":{port} " in line or f":{port}\t" in line for line in snapshot.get("listeners", []))


def vulkan_summary(selector: str | None) -> dict[str, Any]:
    env = dict(os.environ)
    if selector:
        env["MESA_VK_DEVICE_SELECT"] = selector
        env["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"
    result = run(["vulkaninfo", "--summary"], env=env, timeout=20.0)
    text = f"{result['stdout']}\n{result['stderr']}"
    devices = []
    current: dict[str, Any] | None = None
    for line in result["stdout"].splitlines():
        gpu_match = re.match(r"GPU(\d+):", line)
        if gpu_match:
            if current is not None:
                devices.append(current)
            current = {"index": int(gpu_match.group(1))}
            continue
        if current is None:
            continue
        stripped = line.strip()
        if "=" in stripped:
            key, value = [part.strip() for part in stripped.split("=", 1)]
            if key in {"vendorID", "deviceID", "deviceName", "driverName", "driverInfo", "deviceType"}:
                current[key] = value
    if current is not None:
        devices.append(current)
    return {
        "selector": selector,
        "returncode": result["returncode"],
        "devices": devices,
        "has_bc250": "AMD BC-250" in text and "driverName         = radv" in text,
        "has_llvmpipe": "llvmpipe" in text,
        "stderr": result["stderr"],
    }


def benchmark_summary() -> dict[str, Any]:
    out = {}
    for name, path in BENCHMARKS.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                data = json.loads(path.read_text())
                totals = []
                for req in data.get("requests", []):
                    total = req.get("debug", {}).get("body", {}).get("last_request", {}).get("total_seconds")
                    if isinstance(total, (int, float)):
                        totals.append(total)
                item.update(
                    {
                        "artifact_label": data.get("artifact_label"),
                        "env_overrides": data.get("env_overrides", {}),
                        "request_count": len(data.get("requests", [])),
                        "best_total_seconds": min(totals) if totals else None,
                        "last_total_seconds": totals[-1] if totals else None,
                    }
                )
            except Exception as exc:
                item["error"] = str(exc)
        out[name] = item
    return out


def launcher_default_report(path: Path, expected: dict[str, str]) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False, "checks": []}
    text = path.read_text(errors="replace")
    checks = []
    for key, value in expected.items():
        needle = f'export {key}="${{{key}:-{value}}}"'
        checks.append({"key": key, "expected_default": value, "ok": needle in text})
    return {
        "path": path.as_posix(),
        "exists": True,
        "checks": checks,
        "ok": all(item["ok"] for item in checks),
    }


def env_expectation_report(actual: dict[str, Any], expected: dict[str, str]) -> dict[str, Any]:
    checks = [
        {"key": key, "expected": value, "actual": actual.get(key), "ok": actual.get(key) == value}
        for key, value in expected.items()
    ]
    return {"checks": checks, "ok": all(item["ok"] for item in checks)}


def profile_report(name: str, benchmarks: dict[str, Any]) -> dict[str, Any]:
    profile = PROFILES[name]
    launcher = profile["launcher"]
    benchmark = benchmarks.get(profile["benchmark"], {})
    best_total = benchmark.get("best_total_seconds")
    max_best = profile["max_best_total_seconds"]
    benchmark_threshold_ok = (
        True
        if max_best is None
        else isinstance(best_total, (int, float)) and best_total <= max_best
    )
    return {
        "name": name,
        "description": profile["description"],
        "launcher": launcher.as_posix(),
        "benchmark_key": profile["benchmark"],
        "benchmark": benchmark,
        "target_2x_5090_seconds": TARGET_2X_5090_SECONDS,
        "launcher_defaults": launcher_default_report(launcher, profile["launcher_defaults"]),
        "benchmark_env": env_expectation_report(benchmark.get("env_overrides") or {}, profile["benchmark_env"]),
        "benchmark_threshold": {
            "best_total_seconds": best_total,
            "max_best_total_seconds": max_best,
            "ok": benchmark_threshold_ok,
        },
    }


def cpu_status_summary() -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(CPU_STATUS), "exists": CPU_STATUS.exists()}
    if not CPU_STATUS.exists():
        return item
    try:
        data = json.loads(CPU_STATUS.read_text())
    except Exception as exc:
        item["error"] = str(exc)
        return item
    health = data.get("health", {}).get("json", {})
    process = data.get("process", {})
    item.update(
        {
            "device": health.get("device"),
            "torch_threads": health.get("torch_threads"),
            "torch_interop_threads": health.get("torch_interop_threads"),
            "model_loaded": health.get("model_loaded"),
            "pid": process.get("pid"),
            "best_thread_setting": data.get("thread_benchmark_summary", {}).get("best"),
            "chunk270_breakdown": data.get("chunk270_breakdown"),
            "current_short_smoke": data.get("current_short_smoke"),
        }
    )
    return item


def append_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any, severity: str = "error") -> None:
    checks.append({"name": name, "ok": ok, "severity": severity, "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-selector", default=os.getenv("CHATTERBOX_VK_DEVICE_SELECT", ""))
    parser.add_argument("--worker-port", type=int, default=int(os.getenv("PORT", "8003")))
    parser.add_argument("--safe-port", type=int, default=8000)
    parser.add_argument("--min-free-disk-gb", type=float, default=8.0)
    parser.add_argument("--min-mem-available-gb", type=float, default=2.0)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="default-quality")
    parser.add_argument("--json", action="store_true", help="print raw JSON only")
    args = parser.parse_args()

    selector = args.device_selector or None
    ports = port_snapshot()
    memory = meminfo()
    disk = diskinfo()
    vulkan = vulkan_summary(selector)
    safe_health = http_json(f"http://127.0.0.1:{args.safe_port}/health")
    benchmarks = benchmark_summary()
    selected_profile = profile_report(args.profile, benchmarks)
    env_state = {
        "CHATTERBOX_VK_DEVICE_SELECT": selector,
        "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE": "1" if selector else os.getenv("MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE", ""),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "HIP_VISIBLE_DEVICES": os.getenv("HIP_VISIBLE_DEVICES", ""),
        "ROCR_VISIBLE_DEVICES": os.getenv("ROCR_VISIBLE_DEVICES", ""),
    }

    checks: list[dict[str, Any]] = []
    append_check(checks, "bc250_vulkan_radv_visible", bool(vulkan["has_bc250"]), vulkan["devices"])
    if selector:
        append_check(
            checks,
            "selected_device_forced_single_visible_gpu",
            len(vulkan["devices"]) == 1 and not vulkan["has_llvmpipe"],
            {"device_count": len(vulkan["devices"]), "has_llvmpipe": vulkan["has_llvmpipe"]},
        )
    else:
        append_check(
            checks,
            "device_selector_supplied",
            False,
            "Set CHATTERBOX_VK_DEVICE_SELECT for multi-GPU worker isolation; optional for one visible BC-250.",
            severity="warning",
        )
    append_check(checks, "safe_api_alive", bool(safe_health.get("ok")), safe_health)
    safe_body = safe_health.get("body", {}) if safe_health.get("ok") else {}
    append_check(
        checks,
        "safe_api_is_cpu_fallback",
        safe_body.get("device") == "cpu" and not safe_body.get("experimental_vulkan_t3", True),
        {"device": safe_body.get("device"), "experimental_vulkan_t3": safe_body.get("experimental_vulkan_t3")},
    )
    append_check(
        checks,
        "worker_port_free",
        not port_is_listening(ports, args.worker_port),
        {"worker_port": args.worker_port, "listeners": ports["listeners"]},
    )
    append_check(
        checks,
        "disk_headroom",
        disk.get("free_gb", 0.0) >= args.min_free_disk_gb,
        {"free_gb": disk.get("free_gb"), "min_free_disk_gb": args.min_free_disk_gb},
    )
    append_check(
        checks,
        "memory_headroom",
        memory.get("available_gb", 0.0) >= args.min_mem_available_gb,
        {"available_gb": memory.get("available_gb"), "min_mem_available_gb": args.min_mem_available_gb},
    )
    append_check(
        checks,
        "hybrid_launcher_executable",
        HYBRID_LAUNCHER.exists() and os.access(HYBRID_LAUNCHER, os.X_OK),
        {"path": HYBRID_LAUNCHER.as_posix()},
    )
    append_check(
        checks,
        "fast_launcher_executable",
        FAST_LAUNCHER.exists() and os.access(FAST_LAUNCHER, os.X_OK),
        {"path": FAST_LAUNCHER.as_posix()},
    )
    append_check(
        checks,
        "fast_fused_launcher_executable",
        FAST_FUSED_LAUNCHER.exists() and os.access(FAST_FUSED_LAUNCHER, os.X_OK),
        {"path": FAST_FUSED_LAUNCHER.as_posix()},
    )
    append_check(
        checks,
        "selected_profile_launcher_executable",
        Path(selected_profile["launcher"]).exists() and os.access(selected_profile["launcher"], os.X_OK),
        {"path": selected_profile["launcher"]},
    )
    append_check(
        checks,
        "selected_profile_launcher_defaults",
        bool((selected_profile.get("launcher_defaults") or {}).get("ok")),
        selected_profile.get("launcher_defaults"),
    )
    append_check(
        checks,
        "selected_profile_benchmark_env",
        bool((selected_profile.get("benchmark_env") or {}).get("ok")),
        selected_profile.get("benchmark_env"),
    )
    append_check(
        checks,
        "selected_profile_benchmark_threshold",
        bool((selected_profile.get("benchmark_threshold") or {}).get("ok")),
        selected_profile.get("benchmark_threshold"),
    )
    append_check(
        checks,
        "rocm_hip_masked",
        env_state["HIP_VISIBLE_DEVICES"] == "" and env_state["ROCR_VISIBLE_DEVICES"] == "",
        {"HIP_VISIBLE_DEVICES": env_state["HIP_VISIBLE_DEVICES"], "ROCR_VISIBLE_DEVICES": env_state["ROCR_VISIBLE_DEVICES"]},
        severity="warning",
    )

    errors = [check for check in checks if not check["ok"] and check["severity"] == "error"]
    warnings = [check for check in checks if not check["ok"] and check["severity"] == "warning"]
    report = {
        "ok": not errors,
        "note": "No audio generation, ROCm/HIP probing, or model loading is performed.",
        "selector": selector,
        "worker_port": args.worker_port,
        "checks": checks,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "resources": {"disk_root": disk, "memory": memory},
        "ports": ports,
        "safe_health": safe_health,
        "vulkan": vulkan,
        "env": env_state,
        "profile": selected_profile,
        "benchmarks": benchmarks,
        "cpu_status": cpu_status_summary(),
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "ok" if report["ok"] else "failed"
        print(f"preflight={status}")
        print(report["note"])
        print(f"profile={args.profile}")
        print(f"profile_launcher={selected_profile['launcher']}")
        for check in checks:
            marker = "ok" if check["ok"] else check["severity"]
            print(f"- {marker}: {check['name']} :: {check['detail']}")
        print(f"disk_free_gb={disk.get('free_gb')}")
        print(f"mem_available_gb={memory.get('available_gb')}")
        for name, item in report["benchmarks"].items():
            best = item.get("best_total_seconds")
            if best is not None:
                print(f"benchmark_{name}_best_seconds={best:.3f}")
        cpu = report["cpu_status"]
        best_cpu = cpu.get("best_thread_setting") or {}
        chunk270 = cpu.get("chunk270_breakdown") or {}
        if best_cpu:
            print(
                "cpu_best_threads="
                f"{best_cpu.get('threads')} interop={best_cpu.get('interop_threads')} "
                f"warm_wall_seconds={best_cpu.get('warm_wall_seconds')}"
            )
        if chunk270:
            print(f"cpu_chunk270_total_seconds={chunk270.get('total_seconds')}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
