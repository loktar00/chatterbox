#!/usr/bin/env python3
"""Guarded active-shape S3 midblock split8 sweep.

This helper intentionally stays narrow:
- no audio generation
- no ROCm/HIP probing
- no live API worker changes
- one midblock at a time with memory/disk/service checks
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
BASE = ROOT / "exports" / "s3_flow_vulkan_components"
DISTINCT = BASE / "distinct_estimator"
BENCH = ROOT / "exports" / "benchmarks"


def mem_available_gb() -> float:
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024.0 / 1024.0 / 1024.0


def run_cmd(cmd: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env.setdefault("CHATTERBOX_PROGRESS", "0")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": completed.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_tail": completed.stdout.splitlines()[-40:],
            "stderr_tail": completed.stderr.splitlines()[-80:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": 124,
            "seconds": time.perf_counter() - started,
            "timeout": timeout,
            "stdout_tail": (exc.stdout or "").splitlines()[-40:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-80:] if isinstance(exc.stderr, str) else [],
        }


def safe_api_health() -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {"ok": response.status == 200, "status": response.status, "body": body}
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


def listeners() -> list[str]:
    result = run_cmd(["ss", "-ltnp"], timeout=10)
    if result["returncode"] != 0:
        return []
    return result["stdout_tail"]


def guardrails(min_mem_gb: float, min_disk_gb: float) -> dict[str, Any]:
    health = safe_api_health()
    active_listeners = listeners()
    mem_gb = mem_available_gb()
    free_gb = disk_free_gb(ROOT)
    worker_ports_open = [
        port
        for port in (8003, 8004, 8010, 4123)
        if any(f":{port} " in line for line in active_listeners)
    ]
    ok = (
        health.get("ok") is True
        and (health.get("body") or {}).get("device") == "cpu"
        and not (health.get("body") or {}).get("experimental_vulkan_t3")
        and not worker_ports_open
        and mem_gb >= min_mem_gb
        and free_gb >= min_disk_gb
    )
    return {
        "ok": ok,
        "safe_api": health,
        "listeners": active_listeners,
        "worker_ports_open": worker_ports_open,
        "mem_available_gb": mem_gb,
        "disk_free_gb": free_gb,
        "min_mem_gb": min_mem_gb,
        "min_disk_gb": min_disk_gb,
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def distinct_vmfb(name: str) -> Path:
    return DISTINCT / name / f"{name}_vulkan_gfx1013.vmfb"


def distinct_ready(frames: int, mid_index: int) -> bool:
    names = [f"s3_distinct_mid_resnet{mid_index}_t{frames}"]
    names.extend(f"s3_distinct_mid{mid_index}_transformer{index}_t{frames}" for index in range(4))
    return all(distinct_vmfb(name).exists() for name in names)


def split8_report_path(frames: int, mid_index: int) -> Path:
    return BASE / f"s3_fused_midblock{mid_index}_t{frames}_compile_flag_matrix_split8_20iter_2026-07-08.json"


def fused_probe_path(frames: int, mid_index: int) -> Path:
    return BASE / f"s3_fused_midblock{mid_index}_t{frames}_probe_5iter_2026-07-08.json"


def split8_ready(frames: int, mid_index: int) -> bool:
    report = load_json(split8_report_path(frames, mid_index))
    if report.get("status") != "ok":
        return False
    return any(
        variant.get("name") == "split8"
        and variant.get("status") == "ok"
        and (variant.get("validation") or {}).get("allclose_1e_4") is True
        for variant in report.get("variants") or []
    )


def run_midblock(args: argparse.Namespace, mid_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mid_index": mid_index,
        "frames": args.frames,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }

    guard = guardrails(args.min_mem_gb, args.min_disk_gb)
    row["pre_guardrails"] = guard
    if not guard["ok"]:
        row["status"] = "guardrail_failed"
        return row

    if split8_ready(args.frames, mid_index) and not args.force:
        row["status"] = "already_complete"
        row["split8_report"] = split8_report_path(args.frames, mid_index).as_posix()
        return row

    if args.dry_run:
        row["status"] = "dry_run"
        row["distinct_ready"] = distinct_ready(args.frames, mid_index)
        row["fused_probe_exists"] = fused_probe_path(args.frames, mid_index).exists()
        row["split8_ready"] = split8_ready(args.frames, mid_index)
        return row

    if not distinct_ready(args.frames, mid_index) or args.force_distinct:
        probes = ",".join(
            [f"mid_resnet_{mid_index}"]
            + [f"mid_transformer_{mid_index}_{index}" for index in range(4)]
        )
        output = DISTINCT / f"midblock{mid_index}_t{args.frames}_distinct_components_2026-07-08.json"
        cmd = [
            PYTHON.as_posix(),
            "export_s3_estimator_distinct_vulkan_components.py",
            "--frames",
            str(args.frames),
            "--probes",
            probes,
            "--compile-timeout",
            str(args.compile_timeout),
            "--run-timeout",
            str(args.run_timeout),
            "--output",
            output.as_posix(),
        ]
        step = {"name": "distinct_components", "output": output.as_posix()}
        step["result"] = run_cmd(cmd, timeout=args.stage_timeout)
        row["steps"].append(step)
        if step["result"]["returncode"] != 0:
            row["status"] = "distinct_failed"
            return row
    else:
        row["steps"].append({"name": "distinct_components", "status": "exists"})

    guard = guardrails(args.min_mem_gb, args.min_disk_gb)
    row["post_distinct_guardrails"] = guard
    if not guard["ok"]:
        row["status"] = "guardrail_failed_after_distinct"
        return row

    probe_output = fused_probe_path(args.frames, mid_index)
    if not probe_output.exists() or args.force_fused:
        cmd = [
            PYTHON.as_posix(),
            "probe_s3_fused_midblock_vulkan.py",
            "--frames",
            str(args.frames),
            "--mid-index",
            str(mid_index),
            "--iterations",
            str(args.probe_iterations),
            "--warmup",
            str(args.probe_warmup),
            "--compile-timeout",
            str(args.compile_timeout),
            "--run-timeout",
            str(args.run_timeout),
            "--output",
            probe_output.as_posix(),
        ]
        step = {"name": "fused_probe", "output": probe_output.as_posix()}
        step["result"] = run_cmd(cmd, timeout=args.stage_timeout)
        row["steps"].append(step)
        if step["result"]["returncode"] != 0:
            row["status"] = "fused_probe_failed"
            return row
    else:
        row["steps"].append({"name": "fused_probe", "status": "exists", "output": probe_output.as_posix()})

    probe_report = load_json(probe_output)
    if probe_report.get("status") != "ok" or (probe_report.get("validation") or {}).get("allclose_1e_4") is not True:
        row["status"] = "fused_probe_invalid"
        row["fused_probe"] = probe_report
        return row

    guard = guardrails(args.min_mem_gb, args.min_disk_gb)
    row["post_fused_guardrails"] = guard
    if not guard["ok"]:
        row["status"] = "guardrail_failed_after_fused"
        return row

    split8_output = split8_report_path(args.frames, mid_index)
    if not split8_ready(args.frames, mid_index) or args.force_split8:
        cmd = [
            PYTHON.as_posix(),
            "probe_s3_fused_midblock_flag_matrix.py",
            "--frames",
            str(args.frames),
            "--mid-index",
            str(mid_index),
            "--iterations",
            str(args.split8_iterations),
            "--warmup",
            str(args.split8_warmup),
            "--variants",
            "split8",
            "--compile-timeout",
            str(args.compile_timeout),
            "--output",
            split8_output.as_posix(),
        ]
        step = {"name": "split8_matrix", "output": split8_output.as_posix()}
        step["result"] = run_cmd(cmd, timeout=args.stage_timeout)
        row["steps"].append(step)
        if step["result"]["returncode"] != 0:
            row["status"] = "split8_failed"
            return row
    else:
        row["steps"].append({"name": "split8_matrix", "status": "exists", "output": split8_output.as_posix()})

    split8_report = load_json(split8_output)
    row["split8_report"] = split8_output.as_posix()
    row["best_no_fetch_save_ms"] = split8_report.get("best_no_fetch_save_ms")
    row["best_fetch_output_save_ms"] = split8_report.get("best_fetch_output_save_ms")
    row["split8_ready"] = split8_ready(args.frames, mid_index)
    row["status"] = "ok" if row["split8_ready"] else "split8_invalid"
    row["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return row


def parse_mid_indices(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=1222)
    parser.add_argument("--mid-indices", default="3", help="Comma/range list, e.g. 3,4,5 or 3-5.")
    parser.add_argument("--min-mem-gb", type=float, default=3.0)
    parser.add_argument("--min-disk-gb", type=float, default=8.0)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--run-timeout", type=int, default=90)
    parser.add_argument("--stage-timeout", type=int, default=480)
    parser.add_argument("--probe-iterations", type=int, default=5)
    parser.add_argument("--probe-warmup", type=int, default=1)
    parser.add_argument("--split8-iterations", type=int, default=20)
    parser.add_argument("--split8-warmup", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Re-run everything except unchanged script code.")
    parser.add_argument("--force-distinct", action="store_true")
    parser.add_argument("--force-fused", action="store_true")
    parser.add_argument("--force-split8", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH / "s3_active_midblock_sweep_2026-07-08.json",
    )
    args = parser.parse_args()

    mid_indices = parse_mid_indices(args.mid_indices)
    report: dict[str, Any] = {
        "description": "Guarded active-shape S3 midblock split8 sweep.",
        "frames": args.frames,
        "mid_indices": mid_indices,
        "dry_run": args.dry_run,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": [],
    }
    for mid_index in mid_indices:
        if mid_index < 0 or mid_index >= 12:
            report["rows"].append({"mid_index": mid_index, "status": "invalid_mid_index"})
            continue
        row = run_midblock(args, mid_index)
        report["rows"].append(row)
        print(
            f"midblock={mid_index} status={row.get('status')} "
            f"fetch_save_ms={row.get('best_fetch_output_save_ms')} "
            f"no_fetch_save_ms={row.get('best_no_fetch_save_ms')}"
        )
        if row.get("status") not in {"ok", "already_complete", "dry_run"}:
            break
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"json={args.output}")
    return 0 if all(row.get("status") in {"ok", "already_complete", "dry_run"} for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
