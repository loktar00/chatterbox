"""Small request-level router for multiple Chatterbox worker APIs.

Each backend is treated as single-flight: the router sends at most one generation
request to a worker at a time. This matches the current BC-250 guidance.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    return float(raw) if raw else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    return int(raw) if raw else default


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def backend_urls() -> list[str]:
    raw = os.getenv("CHATTERBOX_ROUTER_BACKENDS", "http://127.0.0.1:8000")
    urls = [part.strip().rstrip("/") for part in raw.split(",") if part.strip()]
    if not urls:
        raise RuntimeError("CHATTERBOX_ROUTER_BACKENDS must contain at least one backend URL")
    return urls


REQUEST_TIMEOUT = env_float("CHATTERBOX_ROUTER_REQUEST_TIMEOUT", 600.0)
HEALTH_TIMEOUT = env_float("CHATTERBOX_ROUTER_HEALTH_TIMEOUT", 5.0)
QUEUE_TIMEOUT = env_float("CHATTERBOX_ROUTER_QUEUE_TIMEOUT", 900.0)
CHUNK_LONG_TEXT = env_flag("CHATTERBOX_ROUTER_CHUNK_LONG_TEXT", False)
CHUNK_CHARS = max(1, min(3000, env_int("CHATTERBOX_ROUTER_CHUNK_CHARS", 270)))
CHUNK_SILENCE_MS = max(0, env_int("CHATTERBOX_ROUTER_CHUNK_SILENCE_MS", 0))


@dataclass
class Backend:
    url: str
    index: int
    busy: bool = False
    requests: int = 0
    failures: int = 0
    last_seconds: float | None = None
    last_error: str | None = None


BACKENDS = [Backend(url=url, index=index) for index, url in enumerate(backend_urls())]
SELECT_LOCK = threading.Lock()
NEXT_INDEX = 0

app = FastAPI(title="Chatterbox Router", version="0.1.0")


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=3000)
    voice: str = "default"
    exaggeration: float = 0.0
    cfg_weight: float = 0.0
    temperature: float = 0.8
    min_p: float = 0.0
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2
    norm_loudness: bool = True
    seed: int = 0


@dataclass
class ChunkResult:
    index: int
    backend: Backend
    seconds: float
    body: bytes
    content_type: str


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = REQUEST_TIMEOUT) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_bytes(method: str, url: str, payload: dict[str, Any], timeout: float = REQUEST_TIMEOUT) -> tuple[int, bytes, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read(), resp.headers.get("Content-Type", "application/octet-stream")


def backend_health(backend: Backend) -> dict[str, Any]:
    started = time.perf_counter()
    with SELECT_LOCK:
        busy = backend.busy
        requests = backend.requests
        failures = backend.failures
        last_seconds = backend.last_seconds
        last_error = backend.last_error
    try:
        body = http_json("GET", f"{backend.url}/health", timeout=HEALTH_TIMEOUT)
        return {
            "ok": True,
            "url": backend.url,
            "index": backend.index,
            "seconds": time.perf_counter() - started,
            "busy": busy,
            "requests": requests,
            "failures": failures,
            "last_seconds": last_seconds,
            "last_error": last_error,
            "body": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "url": backend.url,
            "index": backend.index,
            "seconds": time.perf_counter() - started,
            "busy": busy,
            "requests": requests,
            "failures": failures,
            "last_seconds": last_seconds,
            "last_error": last_error,
            "error": str(exc),
        }


def acquire_backend() -> Backend:
    global NEXT_INDEX
    deadline = time.monotonic() + QUEUE_TIMEOUT
    while time.monotonic() < deadline:
        with SELECT_LOCK:
            order = [(NEXT_INDEX + offset) % len(BACKENDS) for offset in range(len(BACKENDS))]
            for index in order:
                backend = BACKENDS[index]
                if not backend.busy:
                    backend.busy = True
                    NEXT_INDEX = (index + 1) % len(BACKENDS)
                    return backend
        time.sleep(0.05)
    raise HTTPException(status_code=503, detail="No Chatterbox backend became available before queue timeout")


def record_backend_result(backend: Backend, seconds: float, error: str | None = None) -> None:
    with SELECT_LOCK:
        backend.last_seconds = seconds
        backend.last_error = error
        if error is None:
            backend.requests += 1
        else:
            backend.failures += 1


def release_backend(backend: Backend) -> None:
    with SELECT_LOCK:
        backend.busy = False


def split_text_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for word in text.split(" "):
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(word), max_chars):
                chunks.append(word[start : start + max_chars])
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = word
    if current:
        chunks.append(current)
    return chunks


def join_wav_bytes(parts: list[bytes], silence_ms: int = CHUNK_SILENCE_MS) -> bytes:
    if not parts:
        raise ValueError("No WAV parts to join")

    params = None
    frames: list[bytes] = []
    silence = b""
    for index, body in enumerate(parts):
        with wave.open(io.BytesIO(body), "rb") as wav:
            current_params = wav.getparams()
            comparable_params = (
                current_params.nchannels,
                current_params.sampwidth,
                current_params.framerate,
                current_params.comptype,
                current_params.compname,
            )
            if params is None:
                params = current_params
                if silence_ms:
                    silence_frames = int(current_params.framerate * silence_ms / 1000)
                    silence = b"\x00" * silence_frames * current_params.nchannels * current_params.sampwidth
            else:
                first_comparable = (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    params.comptype,
                    params.compname,
                )
                if comparable_params != first_comparable:
                    raise ValueError(f"WAV part {index} format does not match the first part")
                if silence:
                    frames.append(silence)
            frames.append(wav.readframes(current_params.nframes))

    out = io.BytesIO()
    with wave.open(out, "wb") as wav_out:
        assert params is not None
        wav_out.setnchannels(params.nchannels)
        wav_out.setsampwidth(params.sampwidth)
        wav_out.setframerate(params.framerate)
        wav_out.writeframes(b"".join(frames))
    return out.getvalue()


def generate_on_backend(payload: dict[str, Any], chunk_index: int = 0) -> ChunkResult:
    backend = acquire_backend()
    started = time.perf_counter()
    try:
        status, body, content_type = http_bytes("POST", f"{backend.url}/audio/speech", payload)
        seconds = time.perf_counter() - started
        if status != 200:
            record_backend_result(backend, seconds, f"HTTP {status}")
            raise HTTPException(status_code=status, detail={"backend": backend.url, "error": f"HTTP {status}"})
        record_backend_result(backend, seconds)
        return ChunkResult(
            index=chunk_index,
            backend=backend,
            seconds=seconds,
            body=body,
            content_type=content_type,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        record_backend_result(backend, time.perf_counter() - started, detail or str(exc))
        raise HTTPException(status_code=exc.code, detail={"backend": backend.url, "error": detail}) from exc
    except HTTPException:
        raise
    except Exception as exc:
        record_backend_result(backend, time.perf_counter() - started, str(exc))
        raise HTTPException(status_code=502, detail={"backend": backend.url, "error": str(exc)}) from exc
    finally:
        release_backend(backend)


def generate_chunked(req: SpeechRequest) -> Response:
    chunks = split_text_chunks(req.input, CHUNK_CHARS)
    if len(chunks) == 1:
        result = generate_on_backend(req.model_dump(), 0)
        return Response(
            content=result.body,
            media_type=result.content_type.split(";")[0] or "audio/wav",
            headers={
                "X-Chatterbox-Backend": result.backend.url,
                "X-Chatterbox-Backend-Index": str(result.backend.index),
                "X-Chatterbox-Chunked": "false",
                "X-Chatterbox-Chunk-Count": "1",
            },
        )

    payloads = []
    base_payload = req.model_dump()
    for index, chunk in enumerate(chunks):
        payload = dict(base_payload)
        payload["input"] = chunk
        payloads.append((index, payload))

    max_workers = min(len(BACKENDS), len(payloads))
    results: list[ChunkResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_on_backend, payload, index) for index, payload in payloads]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.index)
    content_types = {item.content_type.split(";")[0] or "audio/wav" for item in results}
    if content_types != {"audio/wav"} and content_types != {"audio/x-wav"}:
        raise HTTPException(
            status_code=502,
            detail={"message": "Chunked router can only stitch WAV responses", "content_types": sorted(content_types)},
        )
    try:
        body = join_wav_bytes([item.body for item in results], CHUNK_SILENCE_MS)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=body,
        media_type="audio/wav",
        headers={
            "X-Chatterbox-Chunked": "true",
            "X-Chatterbox-Chunk-Count": str(len(results)),
            "X-Chatterbox-Backends": ",".join(item.backend.url for item in results),
            "X-Chatterbox-Backend-Indices": ",".join(str(item.backend.index) for item in results),
            "X-Chatterbox-Chunk-Seconds": ",".join(f"{item.seconds:.3f}" for item in results),
        },
    )


@app.get("/health")
def health() -> dict[str, Any]:
    backends = [backend_health(backend) for backend in BACKENDS]
    return {
        "ok": any(item["ok"] for item in backends),
        "backend_count": len(BACKENDS),
        "available_count": sum(1 for item in backends if item["ok"] and not item["busy"]),
        "single_flight_per_backend": True,
        "chunk_long_text": CHUNK_LONG_TEXT,
        "chunk_chars": CHUNK_CHARS,
        "chunk_silence_ms": CHUNK_SILENCE_MS,
        "backends": backends,
    }


@app.get("/voices")
def voices() -> Any:
    errors = []
    for backend in BACKENDS:
        try:
            return http_json("GET", f"{backend.url}/voices", timeout=HEALTH_TIMEOUT)
        except Exception as exc:
            errors.append({"url": backend.url, "error": str(exc)})
    raise HTTPException(status_code=503, detail={"message": "No backend returned voices", "errors": errors})


@app.post("/audio/speech")
def audio_speech(req: SpeechRequest) -> Response:
    if CHUNK_LONG_TEXT and len(split_text_chunks(req.input, CHUNK_CHARS)) > 1:
        return generate_chunked(req)

    result = generate_on_backend(req.model_dump(), 0)
    return Response(
        content=result.body,
        status_code=200,
        media_type=result.content_type.split(";")[0] or "audio/wav",
        headers={"X-Chatterbox-Backend": result.backend.url, "X-Chatterbox-Backend-Index": str(result.backend.index)},
    )


@app.post("/audio/speech/chunked")
def audio_speech_chunked(req: SpeechRequest) -> Response:
    return generate_chunked(req)
