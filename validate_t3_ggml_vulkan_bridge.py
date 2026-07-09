#!/usr/bin/env python3
"""Validate the ctypes ggml/Vulkan T3 bridge against the saved real-prompt fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from t3_ggml_vulkan_runtime import DEFAULT_WEIGHTS_DIR, T3GGMLVulkanRuntime


ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "exports" / "benchmarks" / "ggml_t3_ctypes_bridge_chunk270_s4_p935_2026-07-08.json"
OUT_MD = ROOT / "exports" / "benchmarks" / "ggml_t3_ctypes_bridge_chunk270_s4_p935_2026-07-08.md"


def read_f32(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32)


def topk(values: np.ndarray, k: int) -> list[int]:
    return np.argsort(values)[-k:][::-1].astype(int).tolist()


def overlap(a: list[int], b: list[int]) -> int:
    return len(set(a).intersection(b))


def load_value_update_layout(path: Path, max_len: int, head_dim: int, heads: int) -> np.ndarray:
    src = read_f32(path).reshape((max_len, head_dim, heads), order="F")
    return np.ascontiguousarray(src.transpose(2, 0, 1))


def load_key_update_layout(path: Path, max_len: int, head_dim: int, heads: int) -> np.ndarray:
    src = read_f32(path).reshape((head_dim, max_len, heads), order="F")
    return np.ascontiguousarray(src.transpose(2, 1, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--initial-context-len", type=int, default=423)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = T3GGMLVulkanRuntime(weights_dir=args.fixture_dir)
    fixture = args.fixture_dir

    for layer in range(runtime.layers):
        key = load_key_update_layout(
            fixture / f"step_00_layer_{layer:02d}_past_k.f32",
            runtime.max_len,
            runtime.head_dim,
            runtime.heads,
        )
        value = load_value_update_layout(
            fixture / f"step_00_layer_{layer:02d}_past_v.f32",
            runtime.max_len,
            runtime.head_dim,
            runtime.heads,
        )
        runtime.set_layer_cache_arrays(layer, key, value)

    steps = []
    for step in range(args.steps):
        prefix = f"step_{step:02d}"
        input_hidden = read_f32(fixture / f"{prefix}_input_hidden.f32")
        attn_mask = read_f32(fixture / f"{prefix}_attn_mask.f32")
        ref_logits = read_f32(fixture / f"{prefix}_ref_logits.f32")
        logits, elapsed_ms = runtime.run_step_with_mask(
            input_hidden,
            attn_mask,
            args.initial_context_len + step,
        )
        max_abs = float(np.max(np.abs(logits - ref_logits)))
        actual_argmax = int(np.argmax(logits))
        expected_argmax = int(np.argmax(ref_logits))
        top10_overlap = overlap(topk(logits, 10), topk(ref_logits, 10))
        ok = bool(np.allclose(logits, ref_logits, atol=1e-3, rtol=1e-3) and actual_argmax == expected_argmax and top10_overlap == 10)
        steps.append(
            {
                "step": step,
                "ms": elapsed_ms,
                "ok": ok,
                "logits_max_abs_error": max_abs,
                "expected_argmax": expected_argmax,
                "actual_argmax": actual_argmax,
                "top10_overlap": top10_overlap,
            }
        )

    mean_ms = float(sum(s["ms"] for s in steps) / len(steps))
    ok = all(s["ok"] for s in steps)
    result = {
        "description": "Python ctypes validation for ggml Vulkan T3 bridge.",
        "fixture_manifest": str(fixture / "manifest.json"),
        "device": runtime.device,
        "ok": ok,
        "mean_ms": mean_ms,
        "steps": steps,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# ggml Vulkan T3 ctypes Bridge Validation",
        "",
        f"Fixture: `{fixture / 'manifest.json'}`",
        "",
        f"- Device: `{runtime.device}`",
        f"- Mean runtime: `{mean_ms:.4f} ms/step`",
        f"- Steps correct: `{sum(1 for s in steps if s['ok'])}/{len(steps)}`",
        "",
        "| Step | ms | Logits max abs | Argmax | Top10 overlap | Status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for s in steps:
        lines.append(
            f"| {s['step']} | {s['ms']:.4f} | {s['logits_max_abs_error']:.6g} | "
            f"{s['actual_argmax']}/{s['expected_argmax']} | {s['top10_overlap']}/10 | {'ok' if s['ok'] else 'fail'} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"device={runtime.device}")
    print(f"mean_ms={mean_ms:.4f}")
    print(f"ok={ok}")
    print(f"wrote={OUT_JSON}")
    print(f"wrote={OUT_MD}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
