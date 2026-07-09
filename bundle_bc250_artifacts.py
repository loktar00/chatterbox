#!/usr/bin/env python3
"""Plan or create a portable BC-250 Chatterbox artifact bundle.

The default mode is a dry run. This script inspects files only unless
``--create`` is supplied. It does not load Chatterbox, start workers, generate
audio, compile exports, or probe ROCm/HIP.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import summarize_bc250_artifacts as artifact_manifest


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
DEFAULT_CREATE_PATH = Path("/root/chatterbox-bc250-artifacts-runtime-evidence.tar.zst")


@dataclass(frozen=True)
class ArtifactGroup:
    name: str
    description: str
    paths: tuple[Path, ...]
    required: bool
    mode: str


def human(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size_bytes}B"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return [item for item in path.rglob("*") if item.is_file()]
    return []


def path_size(path: Path) -> int:
    return sum(item.stat().st_size for item in iter_files(path))


def path_count(path: Path) -> int:
    return len(iter_files(path))


def wav_evidence_paths() -> tuple[Path, ...]:
    if not EXPORTS.exists():
        return ()
    selected = []
    covered_roots = (artifact_manifest.BENCH, EXPORTS / "audio_checks")
    for path in EXPORTS.rglob("*.wav"):
        if any(root in path.parents or path == root for root in covered_roots):
            continue
        selected.append(path)
    return tuple(sorted(selected))


def artifact_groups() -> list[ArtifactGroup]:
    return [
        ArtifactGroup(
            "helper_libraries",
            "Native T3 helper libraries built from committed C++ sources",
            tuple(artifact_manifest.HELPER_LIBS),
            True,
            "runtime",
        ),
        ArtifactGroup(
            "t3_ggml_weights",
            "T3 ggml/Vulkan weights and manifest",
            (artifact_manifest.T3_WEIGHTS,),
            True,
            "runtime",
        ),
        ArtifactGroup(
            "s3_iree_vulkan",
            "S3 IREE/Vulkan VMFBs and descriptors",
            (artifact_manifest.S3_ROOT,),
            True,
            "runtime",
        ),
        ArtifactGroup(
            "hift_iree_vulkan",
            "HiFT IREE/Vulkan VMFBs and descriptors",
            (artifact_manifest.HIFT_ROOT,),
            True,
            "runtime",
        ),
        ArtifactGroup(
            "benchmark_reports",
            "Saved benchmark, verifier, runtime matrix, and manifest evidence",
            (artifact_manifest.BENCH,),
            False,
            "evidence",
        ),
        ArtifactGroup(
            "audio_checks",
            "Saved waveform sanity/comparison evidence",
            (EXPORTS / "audio_checks",),
            False,
            "evidence",
        ),
        ArtifactGroup(
            "wav_outputs",
            "Listening/debug WAV evidence outside benchmark and audio-check folders",
            wav_evidence_paths(),
            False,
            "all",
        ),
    ]


def selected_groups(mode: str) -> list[ArtifactGroup]:
    allowed = {"runtime"}
    if mode in {"runtime-evidence", "all"}:
        allowed.add("evidence")
    if mode == "all":
        allowed.add("all")
    return [group for group in artifact_groups() if group.mode in allowed]


def prune_nested(paths: list[Path]) -> list[Path]:
    pruned: list[Path] = []
    for path in sorted(paths, key=lambda item: (len(item.parts), item.as_posix())):
        resolved = path.resolve()
        if any(parent == resolved or parent in resolved.parents for parent in pruned):
            continue
        pruned.append(resolved)
    return pruned


def build_plan(mode: str) -> dict[str, Any]:
    groups = []
    all_paths: list[Path] = []
    missing_required = []
    for group in selected_groups(mode):
        rows = []
        group_size = 0
        group_count = 0
        for path in group.paths:
            exists = path.exists()
            size = path_size(path)
            count = path_count(path)
            if exists:
                all_paths.append(path)
            elif group.required:
                missing_required.append(path.as_posix())
            group_size += size
            group_count += count
            rows.append(
                {
                    "path": path.as_posix(),
                    "relative_path": rel(path) if exists else None,
                    "exists": exists,
                    "file_count": count,
                    "size_bytes": size,
                    "size": human(size),
                }
            )
        groups.append(
            {
                "name": group.name,
                "description": group.description,
                "required": group.required,
                "mode": group.mode,
                "exists": all(row["exists"] for row in rows) if rows else True,
                "file_count": group_count,
                "size_bytes": group_size,
                "size": human(group_size),
                "paths": rows,
            }
        )

    archive_paths = prune_nested(all_paths)
    total_size = sum(path_size(path) for path in archive_paths)
    return {
        "ok": not missing_required,
        "mode": mode,
        "note": "Dry run by default. No model load, audio generation, worker start, compilation, or ROCm/HIP probing is performed.",
        "root": ROOT.as_posix(),
        "groups": groups,
        "archive_relative_paths": [rel(path) for path in archive_paths],
        "total_size_bytes": total_size,
        "total_size": human(total_size),
        "missing_required": missing_required,
        "default_create_path": DEFAULT_CREATE_PATH.as_posix(),
    }


def tar_command(output: Path, compression: str, rel_paths: list[str]) -> list[str]:
    cmd = ["tar", "-C", ROOT.as_posix()]
    if compression == "zstd":
        cmd.append("--zstd")
        cmd.extend(["-cf", output.as_posix()])
    elif compression == "gzip":
        cmd.extend(["-czf", output.as_posix()])
    else:
        cmd.extend(["-cf", output.as_posix()])
    cmd.extend(rel_paths)
    return cmd


def auto_compression(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffixes = "".join(path.suffixes[-2:])
    if suffixes.endswith(".tar.zst") or path.suffix == ".zst":
        return "zstd"
    if suffixes.endswith(".tar.gz") or path.suffix == ".gz":
        return "gzip"
    return "none"


def can_create_archive(plan: dict[str, Any], output: Path, compression: str, overwrite: bool, min_free_after_gb: float) -> None:
    if not plan["ok"]:
        raise RuntimeError(f"missing required artifacts: {plan['missing_required']}")
    if not plan["archive_relative_paths"]:
        raise RuntimeError("no artifact paths selected")
    if output.exists() and not overwrite:
        raise RuntimeError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if compression == "zstd" and shutil.which("zstd") is None:
        raise RuntimeError("zstd is not installed; use --compression gzip or --compression none")
    if shutil.which("tar") is None:
        raise RuntimeError("tar is not installed")
    usage = shutil.disk_usage(output.parent)
    min_free_after = int(min_free_after_gb * (1024**3))
    estimated_needed = int(plan["total_size_bytes"])
    if usage.free - estimated_needed < min_free_after:
        raise RuntimeError(
            "not enough free space for a conservative archive estimate: "
            f"free={human(usage.free)} estimated_needed={human(estimated_needed)} "
            f"min_free_after={human(min_free_after)}"
        )


def create_archive(plan: dict[str, Any], output: Path, compression: str, overwrite: bool, min_free_after_gb: float) -> dict[str, Any]:
    can_create_archive(plan, output, compression, overwrite, min_free_after_gb)
    cmd = tar_command(output, compression, plan["archive_relative_paths"])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "created": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": cmd,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output": output.as_posix(),
        "output_size_bytes": output.stat().st_size if output.exists() else 0,
        "output_size": human(output.stat().st_size) if output.exists() else "0.0B",
    }


def print_text(plan: dict[str, Any], command: list[str]) -> None:
    print("BC-250 artifact bundle plan")
    print(f"mode={plan['mode']}")
    print(f"ok={plan['ok']}")
    print(f"total={plan['total_size']}")
    print(f"missing_required={len(plan['missing_required'])}")
    for group in plan["groups"]:
        print(
            f"- {group['name']}: exists={group['exists']} required={group['required']} "
            f"files={group['file_count']} size={group['size']}"
        )
    if plan["missing_required"]:
        print("missing:")
        for path in plan["missing_required"]:
            print(f"- {path}")
    print("dry_run=true")
    print("create_command:")
    print(" ".join(command))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("runtime", "runtime-evidence", "all"), default="runtime-evidence")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    parser.add_argument("--create", nargs="?", const=DEFAULT_CREATE_PATH, type=Path)
    parser.add_argument("--compression", choices=("auto", "zstd", "gzip", "none"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-free-after-gb", type=float, default=8.0)
    args = parser.parse_args()

    plan = build_plan(args.mode)
    create_path = args.create or DEFAULT_CREATE_PATH
    compression = auto_compression(create_path, args.compression)
    command = tar_command(create_path, compression, plan["archive_relative_paths"])
    plan["compression"] = compression
    plan["create_command"] = command

    if args.create:
        try:
            plan["create_result"] = create_archive(plan, args.create, compression, args.overwrite, args.min_free_after_gb)
        except Exception as exc:
            plan["create_result"] = {"created": False, "error": str(exc)}
            if args.output_json:
                args.output_json.write_text(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True) + "\n")
            print(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True) if args.json else str(exc), file=sys.stderr)
            return 1

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(plan, indent=2 if args.pretty else None, sort_keys=True))
    else:
        print_text(plan, command)
        if args.create:
            result = plan["create_result"]
            print(f"created={result.get('created')}")
            print(f"output={result.get('output')}")
            print(f"output_size={result.get('output_size')}")
    return 0 if plan["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
