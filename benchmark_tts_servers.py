#!/usr/bin/env python3
"""Sequential Chatterbox TTS benchmark for local CPU and remote GPU APIs."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "exports" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


SHORT_TEXT = "This is a test of the text to speech system."
CHUNK_270 = (
    "This benchmark uses a longer prompt to approximate one safe chunk for the "
    "CUDA server while staying below the reported reliability limit. It should "
    "produce a longer clip and expose the throughput difference between CPU and "
    "GPU inference without sending concurrent requests."
)
CHUNK_270 = CHUNK_270[:270]


@dataclass
class Server:
    name: str
    base_url: str


SERVERS = [
    Server("local_cpu", "http://127.0.0.1:8000"),
    Server("remote_5090", os.getenv("REMOTE_TTS_URL", "http://192.168.1.120:4123")),
]


def request(
    method: str,
    url: str,
    *,
    payload: dict | None = None,
    timeout: int = 30,
) -> tuple[int | None, bytes, float, str | None]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return resp.status, body, time.monotonic() - started, None
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        return None, b"", time.monotonic() - started, str(exc)


def wav_info(path: Path) -> tuple[float | None, int | None, int | None]:
    try:
        with wave.open(path.as_posix(), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            return frames / float(rate), rate, channels
    except wave.Error:
        return None, None, None


def print_json(prefix: str, value: bytes) -> None:
    try:
        parsed = json.loads(value.decode("utf-8"))
    except Exception:
        print(f"{prefix}: raw={value[:200]!r}")
    else:
        print(f"{prefix}: {json.dumps(parsed, sort_keys=True)}")


def bench_server(server: Server) -> None:
    print(f"\n== {server.name} ({server.base_url}) ==")

    status, body, elapsed, err = request("GET", f"{server.base_url}/health", timeout=10)
    print(f"health: status={status} time={elapsed:.3f}s error={err}")
    if body:
        print_json("health_body", body)

    status, body, elapsed, err = request("GET", f"{server.base_url}/voices", timeout=10)
    print(f"voices: status={status} time={elapsed:.3f}s bytes={len(body)} error={err}")
    if body:
        print_json("voices_body", body)

    cases = [
        ("short_1", SHORT_TEXT),
        ("short_2", SHORT_TEXT),
        ("chunk270_1", CHUNK_270),
    ]

    for case_name, text in cases:
        payload = {
            "input": text,
            "voice": "morgan",
            "exaggeration": 0.33,
            "cfg_weight": 0.67,
        }
        status, body, elapsed, err = request(
            "POST",
            f"{server.base_url}/audio/speech",
            payload=payload,
            timeout=420,
        )
        print(
            f"{case_name}: status={status} time={elapsed:.3f}s "
            f"chars={len(text)} bytes={len(body)} error={err}"
        )
        if status == 200 and body:
            out_path = OUT_DIR / f"{server.name}_{case_name}.wav"
            out_path.write_bytes(body)
            duration, sample_rate, channels = wav_info(out_path)
            if duration is not None:
                realtime = elapsed / duration if duration > 0 else None
                print(
                    f"{case_name}_wav: file={out_path} duration={duration:.3f}s "
                    f"sample_rate={sample_rate} channels={channels} "
                    f"wall_per_audio={realtime:.2f}x"
                )
            else:
                print(f"{case_name}_wav: file={out_path} parse_failed=true")


def main() -> None:
    print(f"output_dir={OUT_DIR}")
    print(f"chunk270_chars={len(CHUNK_270)}")
    for server in SERVERS:
        bench_server(server)


if __name__ == "__main__":
    main()
