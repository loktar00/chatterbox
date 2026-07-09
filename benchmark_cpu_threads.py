#!/usr/bin/env python3
"""Benchmark Chatterbox Turbo CPU generation under different thread counts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


TEXT = "This is a test of the text to speech system."
OUT_DIR = Path(__file__).resolve().parent / "exports" / "cpu_thread_bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def worker(threads: int, interop_threads: int, runs: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""

    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download

    from chatterbox.tts_turbo import ChatterboxTurboTTS, REPO_ID

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop_threads)
    torch.manual_seed(0)

    ckpt_dir = snapshot_download(
        repo_id=REPO_ID,
        local_files_only=True,
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"],
    )

    load_started = time.monotonic()
    model = ChatterboxTurboTTS.from_local(ckpt_dir, "cpu")
    load_seconds = time.monotonic() - load_started

    results = []
    for run_idx in range(1, runs + 1):
        torch.manual_seed(run_idx)
        started = time.monotonic()
        wav = model.generate(
            TEXT,
            temperature=0.8,
            top_p=0.95,
            top_k=1000,
            repetition_penalty=1.2,
            norm_loudness=True,
        )
        wall_seconds = time.monotonic() - started
        arr = wav.detach().cpu().squeeze().numpy()
        duration = float(arr.shape[0]) / float(model.sr)
        out_path = OUT_DIR / f"threads{threads}_interop{interop_threads}_run{run_idx}.wav"
        sf.write(out_path, arr, model.sr, format="WAV")
        results.append(
            {
                "run": run_idx,
                "wall_seconds": wall_seconds,
                "audio_seconds": duration,
                "wall_per_audio": wall_seconds / duration,
                "wav": out_path.as_posix(),
            }
        )

    print(json.dumps({
        "threads": threads,
        "interop_threads": interop_threads,
        "load_seconds": load_seconds,
        "runs": results,
    }))


def driver(thread_counts: list[int], interop_threads: int, runs: int) -> None:
    rows = []
    for threads in thread_counts:
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--threads",
            str(threads),
            "--interop-threads",
            str(interop_threads),
            "--runs",
            str(runs),
        ]
        started = time.monotonic()
        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            print(f"threads={threads} failed rc={completed.returncode}", file=sys.stderr)
            print(completed.stderr, file=sys.stderr)
            rows.append({"threads": threads, "error": completed.stderr[-2000:], "elapsed": elapsed})
            continue

        line = completed.stdout.strip().splitlines()[-1]
        row = json.loads(line)
        row["process_seconds"] = elapsed
        if completed.stderr.strip():
            row["stderr_tail"] = completed.stderr.strip().splitlines()[-5:]
        rows.append(row)
        warm = row["runs"][-1]
        print(
            f"threads={threads} interop={interop_threads} "
            f"load={row['load_seconds']:.3f}s warm={warm['wall_seconds']:.3f}s "
            f"audio={warm['audio_seconds']:.3f}s wall/audio={warm['wall_per_audio']:.2f}x",
            flush=True,
        )

    output_path = OUT_DIR / "cpu_thread_benchmark.json"
    output_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"results={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--thread-counts", default="1,2,4")
    args = parser.parse_args()

    if args.worker:
        worker(args.threads, args.interop_threads, args.runs)
    else:
        thread_counts = [int(part) for part in args.thread_counts.split(",") if part.strip()]
        driver(thread_counts, args.interop_threads, args.runs)


if __name__ == "__main__":
    main()
