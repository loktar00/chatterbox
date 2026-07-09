#!/usr/bin/env python3
"""Summarize T3 loop telemetry from a guarded Chatterbox benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "exports/benchmarks/vulkan_hybrid_api_t3_loop_info_no_watermark_default_quality_guarded_2026-07-08.json"
)


def last_request(row: dict[str, Any]) -> dict[str, Any]:
    return (((row.get("debug") or {}).get("body") or {}).get("last_request") or {})


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def fmt(value: Any, suffix: str = "s") -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}{suffix}"
    return "n/a"


def request_summary(row: dict[str, Any]) -> dict[str, Any]:
    lr = last_request(row)
    loop = lr.get("t3_loop_info") or {}
    timings = loop.get("timings") or {}
    per_step = loop.get("per_step") or {}
    t3_seconds = number(lr.get("t3_seconds"))
    loop_total = sum(
        number(timings.get(key)) or 0.0
        for key in (
            "fast_loop_setup_seconds",
            "step_hidden_seconds",
            "step_ggml_wall_seconds",
            "step_logits_to_torch_seconds",
            "step_sampling_seconds",
        )
    )
    non_loop_t3 = t3_seconds - loop_total if t3_seconds is not None else None
    return {
        "index": row.get("index"),
        "http_status": row.get("status"),
        "wall_seconds": row.get("wall_seconds"),
        "server_total_seconds": lr.get("total_seconds"),
        "t3_seconds": t3_seconds,
        "s3_flow_seconds": lr.get("s3_flow_seconds"),
        "hift_decode_seconds": lr.get("hift_decode_seconds"),
        "watermark_seconds": lr.get("watermark_seconds"),
        "t3_loop_info_present": bool(loop),
        "fast_loop": loop.get("fast_loop"),
        "fast_sampler": loop.get("fast_sampler"),
        "initial_context_len": loop.get("initial_context_len"),
        "loop_iterations": loop.get("loop_iterations"),
        "generated_tokens": loop.get("generated_tokens"),
        "loop_total_seconds": loop_total if loop else None,
        "non_loop_t3_seconds": non_loop_t3,
        "timings": timings,
        "per_step": per_step,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# T3 Loop Telemetry Summary",
        "",
        f"- Source: `{summary['source']}`",
        f"- Benchmark status: `{summary.get('status')}`",
        f"- Requests: `{len(summary['requests'])}`",
        "",
        "| Req | Wall | T3 | Loop | Non-loop T3 | ggml wall | logits->Torch | sampling | Iterations | ggml mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["requests"]:
        timings = row.get("timings") or {}
        per_step = row.get("per_step") or {}
        lines.append(
            "| "
            f"{row.get('index')} | "
            f"`{fmt(row.get('wall_seconds'))}` | "
            f"`{fmt(row.get('t3_seconds'))}` | "
            f"`{fmt(row.get('loop_total_seconds'))}` | "
            f"`{fmt(row.get('non_loop_t3_seconds'))}` | "
            f"`{fmt(timings.get('step_ggml_wall_seconds'))}` | "
            f"`{fmt(timings.get('step_logits_to_torch_seconds'))}` | "
            f"`{fmt(timings.get('step_sampling_seconds'))}` | "
            f"`{row.get('loop_iterations')}` | "
            f"`{fmt(per_step.get('ggml_wall_mean_ms'), 'ms')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ggml wall` is the per-token bridge call wall time accumulated by Python.",
            "- `logits->Torch` and `sampling` are the host-side costs that a larger native token loop could reduce.",
            "- `non-loop T3` includes tokenization-adjacent setup, CPU prefill/suffix prefill, initial speech head/sample, and cache upload.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    summary = {
        "source": args.input.as_posix(),
        "status": data.get("status"),
        "artifact_label": data.get("artifact_label"),
        "requests": [request_summary(row) for row in data.get("requests", [])],
    }
    output = args.output or args.input.with_name(args.input.stem + "_t3_loop_summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, output.with_suffix(".md"))
    print(f"json={output}")
    print(f"markdown={output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
