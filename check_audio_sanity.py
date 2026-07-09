#!/usr/bin/env python3
"""Basic WAV sanity and optional pairwise comparison for benchmark outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path.as_posix(), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float64), int(sample_rate)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(path: Path) -> dict[str, Any]:
    data, sample_rate = read_audio(path)
    finite = np.isfinite(data)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    return {
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "sample_rate": sample_rate,
        "frames": int(data.shape[0]),
        "duration_seconds": float(data.shape[0] / sample_rate) if sample_rate else None,
        "peak_abs": peak,
        "rms": rms,
        "rms_dbfs": float(20.0 * np.log10(max(rms, 1e-12))),
        "finite_fraction": float(np.mean(finite)) if data.size else 0.0,
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
        "clipped_fraction_abs_ge_0_999": float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0,
        "near_silence_fraction_abs_lt_1e_4": float(np.mean(np.abs(data) < 1e-4)) if data.size else 0.0,
        "non_silent": bool(rms > 1e-4),
        "passed_basic_sanity": bool(
            data.size
            and sample_rate > 0
            and np.all(finite)
            and peak < 0.999
            and rms > 1e-4
        ),
    }


def compare(actual: Path, reference: Path) -> dict[str, Any]:
    actual_data, actual_sr = read_audio(actual)
    reference_data, reference_sr = read_audio(reference)
    n = min(actual_data.shape[0], reference_data.shape[0])
    if n == 0:
        return {"samples_compared": 0}
    a = actual_data[:n]
    b = reference_data[:n]
    diff = a - b
    if np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = None
    return {
        "reference_path": reference.as_posix(),
        "sample_rates_match": actual_sr == reference_sr,
        "samples_compared": int(n),
        "rms_diff": float(np.sqrt(np.mean(np.square(diff)))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "correlation": corr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--description", default="Audio sanity check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {
        "description": args.description,
        "audio": summarize(args.audio),
    }
    if args.reference is not None:
        report["reference"] = summarize(args.reference)
        report["comparison_to_reference"] = compare(args.audio, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"json={args.output}")
    print(f"passed={report['audio']['passed_basic_sanity']}")
    return 0 if report["audio"]["passed_basic_sanity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
