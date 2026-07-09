#!/usr/bin/env python3
"""Summarize ignored BC-250 runtime artifacts.

This script only inspects files. It does not load Chatterbox, start workers,
generate audio, compile exports, or probe ROCm/HIP.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
BENCH = EXPORTS / "benchmarks"
DEFAULT_OUTPUT = BENCH / "bc250_artifact_manifest_latest.json"

HELPER_LIBS = [
    ROOT / "libt3_native_sampler_bridge.so",
    ROOT / "libt3_ggml_vulkan_bridge.so",
    ROOT / "libt3_ggml_vulkan_bridge_f16weights.so",
    ROOT / "libt3_ggml_vulkan_bridge_f16weights_range.so",
]
T3_WEIGHTS = EXPORTS / "ggml_t3_real_prompt_multistep_chunk270_s4_p935"
S3_ROOT = EXPORTS / "s3_flow_vulkan_components"
HIFT_ROOT = EXPORTS / "iree_vulkan_real_hift_core"
ACTIVE_HIFT = [
    HIFT_ROOT / "real_hift_core_no_fft_t128_vulkan_gfx1013.vmfb",
    HIFT_ROOT / "real_hift_core_no_fft_t96_vulkan_gfx1013.vmfb",
]
S3_BUCKETS = {
    "605_1210_1210": ("t605", "t1210"),
    "611_1222_1222": ("t611", "t1222"),
    "615_1230_1230": ("t615", "t1230"),
    "629_1258_1258": ("t629", "t1258"),
}


def run(cmd: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def human(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() and path.is_file() else 0


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def count_files(path: Path, pattern: str = "*") -> int:
    return sum(1 for item in path.rglob(pattern) if item.is_file()) if path.exists() else 0


def ignored(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    return run(["git", "check-ignore", "-q", rel], timeout=5.0)["returncode"] == 0


def helper_lib_summary() -> list[dict[str, Any]]:
    rows = []
    for path in HELPER_LIBS:
        ldd = run(["ldd", path.as_posix()], timeout=10.0) if path.exists() else {"returncode": None, "stdout": ""}
        rows.append(
            {
                "path": path.as_posix(),
                "exists": path.exists(),
                "ignored": ignored(path),
                "size_bytes": file_size(path),
                "size": human(file_size(path)),
                "linked": ldd["returncode"] == 0 and "not found" not in ldd["stdout"],
            }
        )
    return rows


def t3_summary() -> dict[str, Any]:
    return {
        "path": T3_WEIGHTS.as_posix(),
        "exists": T3_WEIGHTS.exists(),
        "ignored": ignored(T3_WEIGHTS),
        "file_count": count_files(T3_WEIGHTS),
        "f32_count": count_files(T3_WEIGHTS, "*.f32"),
        "manifest_exists": (T3_WEIGHTS / "manifest.json").exists(),
        "size_bytes": dir_size(T3_WEIGHTS),
        "size": human(dir_size(T3_WEIGHTS)),
    }


def s3_summary() -> dict[str, Any]:
    vmfbs = list(S3_ROOT.rglob("*.vmfb")) if S3_ROOT.exists() else []
    buckets = {}
    for name, tokens in S3_BUCKETS.items():
        matched = [
            path
            for path in vmfbs
            if any(token in path.name or token in path.as_posix() for token in tokens)
        ]
        buckets[name] = {
            "vmfb_count": len(matched),
            "size_bytes": sum(path.stat().st_size for path in matched),
            "size": human(sum(path.stat().st_size for path in matched)),
        }
    return {
        "path": S3_ROOT.as_posix(),
        "exists": S3_ROOT.exists(),
        "ignored": ignored(S3_ROOT),
        "vmfb_count": len(vmfbs),
        "json_count": count_files(S3_ROOT, "*.json"),
        "size_bytes": dir_size(S3_ROOT),
        "size": human(dir_size(S3_ROOT)),
        "buckets": buckets,
    }


def hift_summary() -> dict[str, Any]:
    vmfbs = list(HIFT_ROOT.glob("*.vmfb")) if HIFT_ROOT.exists() else []
    return {
        "path": HIFT_ROOT.as_posix(),
        "exists": HIFT_ROOT.exists(),
        "ignored": ignored(HIFT_ROOT),
        "vmfb_count": len(vmfbs),
        "size_bytes": dir_size(HIFT_ROOT),
        "size": human(dir_size(HIFT_ROOT)),
        "active": [
            {
                "path": path.as_posix(),
                "exists": path.exists(),
                "size_bytes": file_size(path),
                "size": human(file_size(path)),
            }
            for path in ACTIVE_HIFT
        ],
        "rejected_t722_probe_exists": (HIFT_ROOT / "real_hift_core_t722_probe.json").exists(),
        "rejected_t722_vmfb_exists": (HIFT_ROOT / "real_hift_core_no_fft_t722_vulkan_gfx1013.vmfb").exists(),
    }


def generated_output_summary() -> dict[str, Any]:
    wavs = list(EXPORTS.rglob("*.wav")) if EXPORTS.exists() else []
    return {
        "benchmarks": {
            "path": BENCH.as_posix(),
            "exists": BENCH.exists(),
            "ignored": ignored(BENCH),
            "file_count": count_files(BENCH),
            "size_bytes": dir_size(BENCH),
            "size": human(dir_size(BENCH)),
        },
        "audio_checks": {
            "path": (EXPORTS / "audio_checks").as_posix(),
            "exists": (EXPORTS / "audio_checks").exists(),
            "ignored": ignored(EXPORTS / "audio_checks"),
            "file_count": count_files(EXPORTS / "audio_checks"),
            "size_bytes": dir_size(EXPORTS / "audio_checks"),
            "size": human(dir_size(EXPORTS / "audio_checks")),
        },
        "wav_count": len(wavs),
        "wav_bytes": sum(path.stat().st_size for path in wavs),
        "wav_size": human(sum(path.stat().st_size for path in wavs)),
    }


def build_report() -> dict[str, Any]:
    return {
        "description": "BC-250 generated runtime artifact manifest. Generated by file inspection only.",
        "note": "No model load, audio generation, worker start, compilation, or ROCm/HIP probing is performed.",
        "helper_libraries": helper_lib_summary(),
        "t3_ggml_weights": t3_summary(),
        "s3_iree_vulkan": s3_summary(),
        "hift_iree_vulkan": hift_summary(),
        "generated_outputs": generated_output_summary(),
        "rebuild": {
            "helper_libraries": "./build_vulkan_helpers.sh",
            "summary_gate": "./verify_bc250_safe_stack.py --require-t3-validation-artifact",
            "runtime_matrix": "./summarize_bc250_runtime_matrix.py --pretty",
        },
        "committed_manifest": "BC250_ARTIFACT_MANIFEST.md",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    t3 = report["t3_ggml_weights"]
    s3 = report["s3_iree_vulkan"]
    hift = report["hift_iree_vulkan"]
    outputs = report["generated_outputs"]
    lines = [
        "# BC-250 Artifact Manifest",
        "",
        "Generated by file inspection only. No model load, audio generation, worker start, compilation, or ROCm/HIP probing is performed.",
        "",
        "## Runtime Artifacts",
        "",
        "| Category | Path | Count | Size | Required For |",
        "| --- | --- | ---: | ---: | --- |",
        f"| T3 ggml weights | `{t3['path']}` | {t3['file_count']} files | `{t3['size']}` | ggml/Vulkan T3 runtime |",
        f"| S3 IREE/Vulkan VMFBs | `{s3['path']}` | {s3['vmfb_count']} VMFBs | `{s3['size']}` | Vulkan S3 exact buckets/fused midblocks |",
        f"| HiFT IREE/Vulkan VMFBs | `{hift['path']}` | {hift['vmfb_count']} VMFBs | `{hift['size']}` | chunked HiFT decode |",
        f"| Benchmark reports | `{outputs['benchmarks']['path']}` | {outputs['benchmarks']['file_count']} files | `{outputs['benchmarks']['size']}` | saved evidence only |",
        f"| Audio checks | `{outputs['audio_checks']['path']}` | {outputs['audio_checks']['file_count']} files | `{outputs['audio_checks']['size']}` | saved audio sanity evidence only |",
        f"| WAV outputs | `exports/**/*.wav` | {outputs['wav_count']} files | `{outputs['wav_size']}` | listen/debug evidence only |",
        "",
        "## Helper Libraries",
        "",
        "| Library | Exists | Ignored | Linked | Size |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["helper_libraries"]:
        lines.append(
            f"| `{row['path']}` | {row['exists']} | {row['ignored']} | {row['linked']} | `{row['size']}` |"
        )
    lines.extend(
        [
            "",
            "## Rebuild / Verify",
            "",
            "```bash",
            report["rebuild"]["helper_libraries"],
            report["rebuild"]["summary_gate"],
            report["rebuild"]["runtime_matrix"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"json={args.output}")
    print(f"markdown={args.output.with_suffix('.md')}")
    print(f"t3_size={report['t3_ggml_weights']['size']}")
    print(f"s3_size={report['s3_iree_vulkan']['size']}")
    print(f"hift_size={report['hift_iree_vulkan']['size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
