#!/usr/bin/env python3
"""Guarded BC-250 Vulkan benchmark for CU configuration trials.

This script refuses to run the chunk270 Vulkan API benchmark unless RADV reports
the expected CU count. It keeps the safe CPU API on :8000 untouched and starts a
temporary experimental Vulkan API on :8003 only when the CU gate passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

from benchmark_tts_servers import CHUNK_270


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "output": completed.stdout,
    }


def radv_cu_state() -> dict[str, Any]:
    env = os.environ.copy()
    env["RADV_DEBUG"] = "info"
    result = run_cmd(["vulkaninfo", "--summary"], env=env, timeout=45)
    text = result["output"]
    num_cu = None
    match = re.search(r"^\s*num_cu\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if match:
        num_cu = int(match.group(1))
    masks = []
    for match in re.finditer(
        r"cu_mask\[(SE\d+)\]\[(SA\d+)\]\s*=\s*(0x[0-9a-fA-F]+)\s*\((\d+)\)",
        text,
    ):
        masks.append(
            {
                "shader_engine": match.group(1),
                "shader_array": match.group(2),
                "mask": match.group(3),
                "enabled_cus": int(match.group(4)),
            }
        )
    return {
        "num_cu": num_cu,
        "cu_masks": masks,
        "returncode": result["returncode"],
        "seconds": result["seconds"],
        "output_tail": text.splitlines()[-80:],
    }


def http_request(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> tuple[int | None, bytes, float, str | None]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), time.perf_counter() - started, None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), time.perf_counter() - started, f"HTTP Error {exc.code}: {exc.reason}"
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return None, b"", time.perf_counter() - started, str(exc)


def parse_json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def lookup_path(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def wav_info(path: Path) -> dict[str, Any]:
    with wave.open(path.as_posix(), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
        channels = wav.getnchannels()
    return {
        "duration_seconds": frames / float(rate),
        "sample_rate": rate,
        "channels": channels,
    }


def port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def wait_health(base_url: str, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts = []
    while time.monotonic() < deadline:
        status, body, elapsed, err = http_request("GET", f"{base_url}/health", timeout=5)
        attempts.append({"status": status, "seconds": elapsed, "error": err})
        if status == 200:
            return {
                "ok": True,
                "attempts": attempts,
                "body": parse_json(body),
            }
        time.sleep(1.0)
    return {
        "ok": False,
        "attempts": attempts[-10:],
    }


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --env value {value!r}; expected KEY=VALUE")
        key, env_value = value.split("=", 1)
        if not key:
            raise SystemExit(f"Invalid --env value {value!r}; empty key")
        overrides[key] = env_value
    return overrides


def start_vulkan_api(port: int, env_overrides: dict[str, str]) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "PORT": str(port),
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CHATTERBOX_PROGRESS": "0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    env.update(env_overrides)
    return subprocess.Popen(
        ["/bin/bash", "./run_api_vulkan_hybrid.sh"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen[str]) -> dict[str, Any]:
    if proc.poll() is not None:
        output = proc.stdout.read() if proc.stdout is not None else ""
        return {"already_exited": True, "returncode": proc.returncode, "output_tail": output.splitlines()[-80:]}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    output = proc.stdout.read() if proc.stdout is not None else ""
    return {"already_exited": False, "returncode": proc.returncode, "output_tail": output.splitlines()[-80:]}


def run_api_benchmark(args: argparse.Namespace, cu_state: dict[str, Any]) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{args.port}"
    if port_is_open("127.0.0.1", args.port):
        raise SystemExit(f"Refusing to start benchmark: port {args.port} is already listening")

    env_overrides = parse_env_overrides(args.env)
    proc = start_vulkan_api(args.port, env_overrides)
    report: dict[str, Any] = {
        "description": "Guarded default-quality Vulkan hybrid API benchmark after BC-250 CU verification.",
        "expected_cu": args.expected_cu,
        "artifact_label": args.artifact_label,
        "env_overrides": env_overrides,
        "required_debug_fields": args.require_debug_field,
        "cu_state": cu_state,
        "base_url": base_url,
        "case": "chunk270",
        "chars": len(CHUNK_270),
        "requests": [],
    }
    try:
        health = wait_health(base_url, timeout=args.startup_timeout)
        report["startup_health"] = health
        if not health.get("ok"):
            report["status"] = "startup_failed"
            return report

        for index in range(args.requests):
            payload = {
                "input": CHUNK_270,
                "voice": "morgan",
                "exaggeration": 0.33,
                "cfg_weight": 0.67,
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 1000,
                "repetition_penalty": 1.2,
                "seed": args.seed,
            }
            status, body, wall, err = http_request(
                "POST",
                f"{base_url}/audio/speech",
                payload=payload,
                timeout=args.request_timeout,
            )
            row: dict[str, Any] = {
                "index": index,
                "status": status,
                "wall_seconds": wall,
                "bytes": len(body),
                "error": err,
            }
            if status == 200 and body:
                wav_path = (
                    OUT_DIR
                    / f"vulkan_hybrid_api_{args.artifact_label}_guarded_chunk270_req{index}_2026-07-08.wav"
                )
                wav_path.write_bytes(body)
                row["wav_path"] = wav_path.as_posix()
                row.update(wav_info(wav_path))
                row["wall_per_audio"] = row["wall_seconds"] / row["duration_seconds"]
            elif body:
                row["response_json"] = parse_json(body)
                row["response_text_tail"] = body.decode("utf-8", errors="replace")[-4000:]

            debug_status, debug_body, debug_wall, debug_err = http_request(
                "GET",
                f"{base_url}/debug/last_request",
                timeout=10,
            )
            row["debug"] = {
                "status": debug_status,
                "seconds": debug_wall,
                "error": debug_err,
                "body": parse_json(debug_body),
            }
            last_request = (((row["debug"].get("body") or {}).get("last_request") or {}))
            missing_fields = [
                field
                for field in args.require_debug_field
                if lookup_path(last_request, field) is None
            ]
            if missing_fields:
                row["required_debug_field_errors"] = missing_fields
            report["requests"].append(row)
        if not all(row.get("status") == 200 for row in report["requests"]):
            report["status"] = "request_failed"
        elif any(row.get("required_debug_field_errors") for row in report["requests"]):
            report["status"] = "missing_required_debug_fields"
        else:
            report["status"] = "ok"
        return report
    finally:
        report["shutdown"] = stop_process(proc)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    artifact_label = report.get("artifact_label") or f"{report.get('expected_cu')}cu"
    lines = [
        f"# Guarded {artifact_label} Vulkan Benchmark",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Artifact label: `{artifact_label}`",
        f"- Expected CUs: `{report.get('expected_cu')}`",
        f"- RADV reported CUs: `{report.get('cu_state', {}).get('num_cu')}`",
        f"- Base URL: `{report.get('base_url')}`",
        "",
    ]
    env_overrides = report.get("env_overrides") or {}
    if env_overrides:
        lines.extend(["## Environment Overrides", ""])
        for key in sorted(env_overrides):
            lines.append(f"- `{key}={env_overrides[key]}`")
        lines.append("")
    required_debug_fields = report.get("required_debug_fields") or []
    if required_debug_fields:
        lines.extend(["## Required Debug Fields", ""])
        for field in required_debug_fields:
            lines.append(f"- `{field}`")
        lines.append("")
    if report.get("requests"):
        lines.extend(
            [
                "| Request | HTTP | Wall | Audio | Wall/audio | T3 | S3 | HiFT | Watermark | Debug Gate |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in report["requests"]:
            timing = (((row.get("debug") or {}).get("body") or {}).get("last_request") or {})
            missing_fields = row.get("required_debug_field_errors") or []
            debug_gate = "ok" if not missing_fields else "missing " + ", ".join(missing_fields)
            lines.append(
                "| "
                f"{row.get('index')} | "
                f"{row.get('status')} | "
                f"`{row.get('wall_seconds', 0.0):.3f}s` | "
                f"`{row.get('duration_seconds', 0.0):.3f}s` | "
                f"`{row.get('wall_per_audio', 0.0):.3f}x` | "
                f"`{timing.get('t3_seconds', 0.0):.3f}s` | "
                f"`{timing.get('s3_flow_seconds', 0.0):.3f}s` | "
                f"`{timing.get('hift_decode_seconds', 0.0):.3f}s` | "
                f"`{timing.get('watermark_seconds', 0.0):.3f}s` | "
                f"{debug_gate} |"
            )
    else:
        lines.append("No benchmark requests were run.")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-cu", type=int, default=40)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run", action="store_true", help="Run the benchmark when the CU gate passes.")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument(
        "--artifact-label",
        default=None,
        help="Label used in saved WAV and markdown artifacts. Defaults to '<expected-cu>cu'.",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--request-timeout", type=int, default=420)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment override for the temporary API, formatted KEY=VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--require-debug-field",
        action="append",
        default=[],
        help="Require this dot-path under /debug/last_request.last_request for every successful request.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUT_DIR / "vulkan_hybrid_api_40cu_guarded_2026-07-08.json",
    )
    args = parser.parse_args()
    if args.artifact_label is None:
        args.artifact_label = f"{args.expected_cu}cu"

    cu_state = radv_cu_state()
    if args.check_only or not args.run:
        print(json.dumps({"cu_state": cu_state, "gate_passed": cu_state.get("num_cu") == args.expected_cu}, indent=2))
        return

    if cu_state.get("num_cu") != args.expected_cu:
        report = {
            "description": "Guarded default-quality Vulkan hybrid API benchmark was skipped because the CU gate did not pass.",
            "status": "cu_gate_failed",
            "expected_cu": args.expected_cu,
            "artifact_label": args.artifact_label,
            "env_overrides": parse_env_overrides(args.env),
            "required_debug_fields": args.require_debug_field,
            "cu_state": cu_state,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report, args.output.with_suffix(".md"))
        print(f"cu_gate_failed expected={args.expected_cu} actual={cu_state.get('num_cu')}")
        print(f"results={args.output}")
        return

    report = run_api_benchmark(args, cu_state)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"status={report.get('status')}")
    print(f"results={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
