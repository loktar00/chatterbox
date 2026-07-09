#!/usr/bin/env python3
"""Summarize the fastest safe CPU fallback configuration without generation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/chatterbox")
CPU_STATUS = ROOT / "exports/cpu_thread_bench/cpu_runtime_status_2026-07-08.json"
OUT_JSON = ROOT / "exports/cpu_thread_bench/cpu_fast_path_status_2026-07-08.json"
OUT_MD = ROOT / "exports/cpu_thread_bench/cpu_fast_path_status_2026-07-08.md"


def main() -> int:
    data = json.loads(CPU_STATUS.read_text())
    best = data.get("thread_benchmark_summary", {}).get("best", {})
    chunk = data.get("chunk270_breakdown", {})
    total = float(chunk.get("total_seconds", 0.0) or 0.0)
    watermark = float(chunk.get("watermark_seconds", 0.0) or 0.0)
    projected_no_watermark = total - watermark if total else None
    projected_gain_pct = (watermark / total * 100.0) if total else None
    report = {
        "description": "CPU fast fallback status derived from saved CPU telemetry; no audio generation.",
        "source": str(CPU_STATUS),
        "launcher": str(ROOT / "run_api_cpu_fast.sh"),
        "best_thread_setting": best,
        "chunk270_breakdown": chunk,
        "projected_no_watermark_chunk270_seconds": projected_no_watermark,
        "projected_no_watermark_gain_seconds": watermark if total else None,
        "projected_no_watermark_gain_percent": projected_gain_pct,
        "safe_api_policy": {
            "run_api_sh_default_watermark": 1,
            "run_api_cpu_fast_sh_default_watermark": 0,
            "run_api_cpu_fast_default_port": 8002,
            "do_not_restart_live_safe_api_automatically": True,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# CPU Fast Path Status - 2026-07-08",
        "",
        "- Source telemetry: `" + str(CPU_STATUS) + "`.",
        "- No audio generation was performed for this audit.",
        f"- Best measured CPU thread setting: `{best}`.",
        f"- Chunk270 CPU baseline: `{total:.3f}s`.",
        f"- Chunk270 watermark cost: `{watermark:.3f}s`.",
        (
            f"- Projected no-watermark CPU chunk270: `{projected_no_watermark:.3f}s` "
            f"({projected_gain_pct:.2f}% faster)."
            if projected_no_watermark is not None and projected_gain_pct is not None
            else "- Projected no-watermark CPU chunk270: `n/a`."
        ),
        "",
        "## Launcher",
        "",
        "`./run_api_cpu_fast.sh` keeps the best measured thread settings and defaults "
        "to `CHATTERBOX_APPLY_WATERMARK=0` on port `8002`.",
        "",
        "The live safe API on `:8000` should not be restarted automatically during Vulkan work.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    if projected_no_watermark is not None:
        print(f"projected_no_watermark_chunk270_seconds={projected_no_watermark:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
