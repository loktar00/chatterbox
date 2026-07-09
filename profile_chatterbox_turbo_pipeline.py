#!/usr/bin/env python3
"""Profile Chatterbox Turbo CPU generation boundaries and emitted frame sizes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.tts_turbo import ChatterboxTurboTTS


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "pipeline_profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tensor_shape(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return list(value.shape)
    return None


def array_shape(value: Any) -> list[int] | None:
    if isinstance(value, np.ndarray):
        return list(value.shape)
    return None


def value_summary(value: Any) -> dict[str, Any]:
    shape = tensor_shape(value)
    if shape is not None:
        result: dict[str, Any] = {"type": "torch.Tensor", "shape": shape, "dtype": str(value.dtype)}
        if value.numel() == 1:
            result["value"] = float(value.detach().cpu().item())
        return result

    shape = array_shape(value)
    if shape is not None:
        return {"type": "numpy.ndarray", "shape": shape, "dtype": str(value.dtype)}

    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [value_summary(item) for item in value[:4]]}

    return {"type": type(value).__name__}


def find_tensor(args: tuple[Any, ...], kwargs: dict[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        value = kwargs.get(name)
        if torch.is_tensor(value):
            return value
    for value in args:
        if torch.is_tensor(value):
            return value
    return None


class Recorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def wrap(
        self,
        obj: Any,
        method_name: str,
        label: str,
        detail_fn: Callable[[tuple[Any, ...], dict[str, Any], Any], dict[str, Any]] | None = None,
    ) -> None:
        original = getattr(obj, method_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            output = original(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            event: dict[str, Any] = {
                "label": label,
                "elapsed_ms": elapsed_ms,
                "output": value_summary(output),
            }
            if detail_fn is not None:
                event.update(detail_fn(args, kwargs, output))
            self.events.append(event)
            return output

        setattr(obj, method_name, wrapped)


def t3_details(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> dict[str, Any]:
    text_tokens = find_tensor(args, kwargs, "text_tokens")
    result: dict[str, Any] = {}
    if text_tokens is not None:
        result["text_token_shape"] = list(text_tokens.shape)
        result["text_token_count"] = int(text_tokens.numel())
    if torch.is_tensor(output):
        result["speech_token_shape_raw"] = list(output.shape)
        result["speech_token_count_raw"] = int(output.numel())
    return result


def flow_details(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> dict[str, Any]:
    speech_tokens = find_tensor(args, kwargs, "speech_tokens")
    result: dict[str, Any] = {}
    if speech_tokens is not None:
        result["speech_token_shape"] = list(speech_tokens.shape)
        result["speech_token_count"] = int(speech_tokens.numel())
    if torch.is_tensor(output):
        result["mel_shape"] = list(output.shape)
        result["mel_frames"] = int(output.shape[-1])
    return result


def hift_details(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> dict[str, Any]:
    speech_feat = find_tensor(args, kwargs, "speech_feat")
    result: dict[str, Any] = {}
    if speech_feat is not None:
        result["speech_feat_shape"] = list(speech_feat.shape)
        result["mel_frames"] = int(speech_feat.shape[-1])
    if isinstance(output, tuple) and output:
        wav = output[0]
        if torch.is_tensor(wav):
            result["wav_shape"] = list(wav.shape)
            result["wav_samples"] = int(wav.shape[-1])
            result["wav_seconds_at_24k"] = float(wav.shape[-1] / 24000.0)
        if len(output) > 1 and torch.is_tensor(output[1]):
            result["source_shape"] = list(output[1].shape)
    return result


def decode_details(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> dict[str, Any]:
    speech_feat = kwargs.get("x")
    source = kwargs.get("s")
    result: dict[str, Any] = {}
    if torch.is_tensor(speech_feat):
        result["speech_feat_shape"] = list(speech_feat.shape)
        result["mel_frames"] = int(speech_feat.shape[-1])
    if torch.is_tensor(source):
        result["source_shape"] = list(source.shape)
        result["source_samples"] = int(source.shape[-1])
    if torch.is_tensor(output):
        result["wav_shape"] = list(output.shape)
        result["wav_samples"] = int(output.shape[-1])
        result["wav_seconds_at_24k"] = float(output.shape[-1] / 24000.0)
    return result


def watermark_details(args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if args:
        shape = array_shape(args[0])
        if shape is not None:
            result["input_shape"] = shape
            result["input_samples"] = int(args[0].shape[-1])
    shape = array_shape(output)
    if shape is not None:
        result["output_shape"] = shape
        result["output_samples"] = int(output.shape[-1])
        result["output_seconds_at_24k"] = float(output.shape[-1] / 24000.0)
    return result


def install_hooks(model: ChatterboxTurboTTS, recorder: Recorder) -> None:
    recorder.wrap(model.t3, "inference_turbo", "t3.inference_turbo", t3_details)
    recorder.wrap(model.s3gen, "flow_inference", "s3gen.flow_inference", flow_details)
    recorder.wrap(model.s3gen, "hift_inference", "s3gen.hift_inference", hift_details)
    recorder.wrap(model.s3gen.mel2wav, "inference", "mel2wav.inference", hift_details)
    recorder.wrap(model.s3gen.mel2wav, "decode", "mel2wav.decode", decode_details)
    recorder.wrap(model.watermarker, "apply_watermark", "watermarker.apply_watermark", watermark_details)


def profile_case(model: ChatterboxTurboTTS, recorder: Recorder, name: str, text: str) -> dict[str, Any]:
    event_start = len(recorder.events)
    started = time.perf_counter()
    with torch.inference_mode():
        wav = model.generate(text)
    total_ms = (time.perf_counter() - started) * 1000.0
    return {
        "case": name,
        "chars": len(text),
        "text": text,
        "total_ms": total_ms,
        "wav_shape": list(wav.shape),
        "wav_samples": int(wav.shape[-1]),
        "wav_seconds_at_24k": float(wav.shape[-1] / 24000.0),
        "wall_per_audio": float((total_ms / 1000.0) / (wav.shape[-1] / 24000.0)),
        "events": recorder.events[event_start:],
    }


def selected_cases(names: list[str]) -> list[tuple[str, str]]:
    available = {
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
        "hello": "Hello world, this is a test.",
    }
    cases = []
    for name in names:
        if name not in available:
            raise ValueError(f"Unknown case {name!r}; available={sorted(available)}")
        cases.append((name, available[name]))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="short", help="Comma-separated cases: hello,short,chunk270")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "pipeline_profile_latest.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.interop_threads > 0:
        torch.set_num_interop_threads(args.interop_threads)

    selected = selected_cases([part.strip() for part in args.cases.split(",") if part.strip()])

    load_started = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_ms = (time.perf_counter() - load_started) * 1000.0
    recorder = Recorder()
    install_hooks(model, recorder)

    results = {
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "model_load_ms": load_ms,
        "cases": [],
    }

    for name, text in selected:
        print(f"profiling {name} chars={len(text)}")
        case = profile_case(model, recorder, name, text)
        results["cases"].append(case)
        print(
            f"{name}: total={case['total_ms']:.1f}ms audio={case['wav_seconds_at_24k']:.3f}s "
            f"wall/audio={case['wall_per_audio']:.2f}x"
        )
        for event in case["events"]:
            print(f"  {event['label']}: {event['elapsed_ms']:.1f}ms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results={args.output}")


if __name__ == "__main__":
    main()
