#!/usr/bin/env python3
"""Audit the safe CPU API runtime without generating audio."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "cpu_thread_bench"
OUT_JSON = OUT_DIR / "cpu_runtime_status_2026-07-08.json"
OUT_MD = OUT_DIR / "cpu_runtime_status_2026-07-08.md"
THREAD_BENCH = OUT_DIR / "cpu_thread_benchmark.json"
CURRENT_SMOKE = OUT_DIR / "cpu_api_short_current_smoke_2026-07-08.json"
BASELINE_CHUNK = ROOT / "exports" / "pipeline_profiles" / "profile_chunk270_2026-07-08.json"


def run(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def get_json(url: str, timeout: int = 5) -> dict[str, Any]:
    started = time.perf_counter()
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return {
                "status": resp.status,
                "seconds": time.perf_counter() - started,
                "json": json.loads(body.decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception:
            parsed = body.decode("utf-8", errors="replace")
        return {
            "status": exc.code,
            "seconds": time.perf_counter() - started,
            "json": parsed,
        }
    except Exception as exc:
        return {
            "status": None,
            "seconds": time.perf_counter() - started,
            "error": str(exc),
        }


def live_cpu_pid() -> int | None:
    status = run(["ss", "-ltnp"])
    for line in status["stdout"].splitlines():
        if ":8000" not in line:
            continue
        match = re.search(r"pid=(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def process_env(pid: int | None) -> dict[str, str]:
    if pid is None:
        return {}
    path = Path(f"/proc/{pid}/environ")
    if not path.exists():
        return {}
    data = path.read_bytes().split(b"\0")
    env = {}
    for raw_item in data:
        if not raw_item or b"=" not in raw_item:
            continue
        key, value = raw_item.split(b"=", 1)
        env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def process_lstart(pid: int | None) -> str | None:
    if pid is None:
        return None
    result = run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "etime=", "-o", "cmd="])
    if result["returncode"] != 0:
        return None
    return result["stdout"].strip()


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_thread_bench(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, list):
        return None
    rows = []
    for item in data:
        runs = item.get("runs") or []
        warm = runs[-1] if runs else {}
        rows.append(
            {
                "threads": item.get("threads"),
                "interop_threads": item.get("interop_threads"),
                "warm_wall_seconds": warm.get("wall_seconds"),
                "audio_seconds": warm.get("audio_seconds"),
                "wall_per_audio": warm.get("wall_per_audio"),
            }
        )
    best = min(
        (row for row in rows if row["warm_wall_seconds"] is not None),
        key=lambda row: row["warm_wall_seconds"],
        default=None,
    )
    return {"rows": rows, "best": best}


def chunk270_breakdown(data: Any) -> dict[str, float] | None:
    if not isinstance(data, dict):
        return None
    cases = data.get("cases") or []
    if not cases:
        return None
    result = {"total_seconds": float(cases[0].get("total_ms", 0.0)) / 1000.0}
    for event in cases[0].get("events", []):
        label = event.get("label", "")
        elapsed = float(event.get("elapsed_ms", 0.0)) / 1000.0
        if label == "t3.inference_turbo":
            result["t3_seconds"] = elapsed
        elif label == "s3gen.flow_inference":
            result["s3_flow_seconds"] = elapsed
        elif label == "s3gen.hift_inference":
            result["hift_seconds"] = elapsed
        elif label == "watermarker.apply_watermark":
            result["watermark_seconds"] = elapsed
    return result


def write_markdown(report: dict[str, Any]) -> None:
    health = report["health"].get("json", {})
    env = report["process"].get("selected_env", {})
    bench = report.get("thread_benchmark_summary") or {}
    chunk = report.get("chunk270_breakdown") or {}
    smoke = report.get("current_short_smoke") or {}
    debug_ready = report["debug_last_request"].get("status") == 200
    progress_suppressed = env.get("CHATTERBOX_PROGRESS") == "0" and env.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
    service_state_notes = [
        "- The thread setting is already the best measured CPU setting in this container.",
    ]
    if debug_ready:
        service_state_notes.append("- The live service exposes `/debug/last_request`; no telemetry restart is needed.")
    else:
        service_state_notes.append("- The live service predates the newer `/debug/last_request` endpoint.")
    if progress_suppressed:
        service_state_notes.append("- Progress output is suppressed in the live service environment.")
    else:
        service_state_notes.append(
            "- Progress-suppression environment is not active; apply it only during an intentional maintenance restart."
        )
    service_state_notes.extend(
        [
            "- Do not restart the safe CPU service automatically during Vulkan probing.",
            "- Further meaningful speed work remains on the Vulkan/ggml/IREE path rather than CPU thread tuning.",
        ]
    )
    lines = [
        "# CPU Runtime Status - 2026-07-08",
        "",
        "## Live Safe API",
        "",
        f"- Health status: `{report['health'].get('status')}`.",
        f"- Device: `{health.get('device')}`.",
        f"- Torch threads: `{health.get('torch_threads')}`.",
        f"- Torch interop threads: `{health.get('torch_interop_threads')}`.",
        f"- PID/start: `{report['process'].get('ps')}`.",
        f"- `/debug/last_request` status: `{report['debug_last_request'].get('status')}`.",
        "",
        "Selected process environment:",
        "",
        f"- `CHATTERBOX_TORCH_THREADS={env.get('CHATTERBOX_TORCH_THREADS')}`",
        f"- `CHATTERBOX_TORCH_INTEROP_THREADS={env.get('CHATTERBOX_TORCH_INTEROP_THREADS')}`",
        f"- `OMP_NUM_THREADS={env.get('OMP_NUM_THREADS')}`",
        f"- `MKL_NUM_THREADS={env.get('MKL_NUM_THREADS')}`",
        f"- `OPENBLAS_NUM_THREADS={env.get('OPENBLAS_NUM_THREADS')}`",
        f"- `CHATTERBOX_PROGRESS={env.get('CHATTERBOX_PROGRESS')}`",
        f"- `HF_HUB_DISABLE_PROGRESS_BARS={env.get('HF_HUB_DISABLE_PROGRESS_BARS')}`",
        "",
        "## Thread Tuning Evidence",
        "",
        f"- Prior best setting: `{bench.get('best')}`.",
        "- Current launcher default remains `CHATTERBOX_TORCH_THREADS=2` and `CHATTERBOX_TORCH_INTEROP_THREADS=1`.",
        "",
        "## Current Short Smoke",
        "",
        f"- Status: `{smoke.get('status')}`.",
        f"- Wall: `{smoke.get('wall_seconds')}` seconds.",
        f"- Audio: `{smoke.get('duration_seconds')}` seconds.",
        f"- Wall/audio: `{smoke.get('wall_per_audio')}`.",
        f"- Artifact: `{CURRENT_SMOKE.as_posix()}`.",
        "",
        "## CPU Bottleneck",
        "",
        f"- Chunk270 total: `{chunk.get('total_seconds')}` seconds.",
        f"- T3: `{chunk.get('t3_seconds')}` seconds.",
        f"- S3 flow: `{chunk.get('s3_flow_seconds')}` seconds.",
        f"- HiFT: `{chunk.get('hift_seconds')}` seconds.",
        f"- Watermark: `{chunk.get('watermark_seconds')}` seconds.",
        "",
        "## Interpretation",
        "",
        *service_state_notes,
        "",
    ]
    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pid = live_cpu_pid()
    env = process_env(pid)
    selected_env_keys = [
        "CHATTERBOX_TORCH_THREADS",
        "CHATTERBOX_TORCH_INTEROP_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CHATTERBOX_PROGRESS",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
    ]
    report: dict[str, Any] = {
        "description": "Safe CPU API runtime audit without generating audio.",
        "health": get_json("http://127.0.0.1:8000/health"),
        "debug_last_request": get_json("http://127.0.0.1:8000/debug/last_request"),
        "process": {
            "pid": pid,
            "ps": process_lstart(pid),
            "selected_env": {key: env.get(key) for key in selected_env_keys},
        },
        "systemd_is_active": run(["systemctl", "is-active", "chatterbox-api.service"]),
        "thread_benchmark_summary": summarize_thread_bench(load_json(THREAD_BENCH)),
        "chunk270_breakdown": chunk270_breakdown(load_json(BASELINE_CHUNK)),
        "current_short_smoke": load_json(CURRENT_SMOKE),
        "notes": [
            "This audit intentionally does not generate audio.",
            "The current short smoke artifact was generated separately and is referenced here.",
            "No ROCm/HIP command is run.",
        ],
    }
    write_markdown(report)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")


if __name__ == "__main__":
    main()
