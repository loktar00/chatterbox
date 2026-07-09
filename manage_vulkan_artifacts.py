#!/usr/bin/env python3
"""Inventory and optionally clean generated Vulkan probe artifacts.

This script is intentionally dry-run by default. Use both ``--delete`` and
``--yes`` to remove matched generated files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
T3 = EXPORTS / "t3_exportability"


STALE_EXPORT_DIRS = {
    "ggml_t3_real_prompt_multistep_hello_s4_p935": (
        "Superseded GGML T3 fixture; the API uses the chunk270 fixture."
    ),
    "ggml_t3_full_stack_fixture_p935": (
        "Intermediate full-stack GGML fixture; chunk270 runtime weights preserve the usable path."
    ),
    "ggml_t3_block0_fixture": "Early GGML block0 fixture superseded by full T3 runtime work.",
    "ggml_t3_block0_fixture_p935": "Early GGML block0 fixture superseded by full T3 runtime work.",
    "iree_vulkan_real_hift_stages": "HiFT stage debug matrix superseded by the real_hift_core VMFBs.",
    "iree_vulkan_f0_debug": "HiFT F0 debug probes; findings are preserved in benchmark reports.",
    "iree_vulkan_convtranspose_debug": "HiFT convtranspose debug probes; findings are preserved in reports.",
    "iree_vulkan_matrix": "Early HiFT feasibility matrix superseded by real_hift_core artifacts.",
    "subgraph_probes": "Early IREE feasibility probes superseded by validated T3/S3/HiFT artifacts.",
}


@dataclass(frozen=True)
class Candidate:
    category: str
    reason: str
    path: Path
    size: int


def sizeof(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def iter_failed_large_fused_t3() -> Iterable[Candidate]:
    """Large fused stacks proved numerically unsafe; keep JSON docs, not blobs."""
    unsafe_names = (
        "gpt2_cached_stack_l5_p128_t1",
        "gpt2_cached_stack_l6_p128_t1",
        "gpt2_cached_stack_l8_p128_t1",
    )
    for path in T3.rglob("*"):
        if not path.is_file():
            continue
        if any(name in path.name or name in path.as_posix() for name in unsafe_names):
            if path.suffix in {".json", ".md", ".txt"}:
                continue
            yield Candidate(
                "failed_large_fused_t3",
                "Fused 5/6/8-layer T3 stacks failed correctness; blobs are regenerated artifacts.",
                path,
                sizeof(path),
            )


def iter_old_t3_flag_variants() -> Iterable[Candidate]:
    """Compiler-flag matrix variants superseded by split_matmul4_split_reduction."""
    removable_variants = {
        "baseline",
        "robust_buffers",
        "generalize_matmul",
        "split_matmul_reduction_2",
        "split_matmul_reduction_4",
        "split_matmul_reduction_8",
        "vectorize_pipeline",
        "no_vector_distribution",
        "no_tile_and_fuse_matmul",
        "no_mmt4d_intrinsics",
        "enable_split_reduction",
        "no_reduction_vector_distribution",
        "fuse_multi_reduction",
        "no_aggressive_fusion",
        "no_fuse_multi_use",
        "dispatch_opt_0",
        "split_matmul8_split_reduction",
    }
    flag_root = T3 / "flag_variants"
    if not flag_root.exists():
        return
    for variant_dir in flag_root.glob("*/*"):
        if not variant_dir.is_dir() or variant_dir.name not in removable_variants:
            continue
        for path in variant_dir.rglob("*"):
            if path.is_file() and path.suffix not in {".json", ".md", ".txt"}:
                yield Candidate(
                    "old_t3_flag_variants",
                    "Superseded T3 compiler-flag matrix artifact; summaries preserve results.",
                    path,
                    sizeof(path),
                )


def iter_t3_mlir_with_vmfb() -> Iterable[Candidate]:
    """Torch/IREE MLIR is large and reproducible; VMFBs/results are smaller."""
    for path in T3.rglob("*.mlir"):
        yield Candidate(
            "t3_mlir_sources",
            "Large generated MLIR source; reproducible from probe scripts if needed.",
            path,
            sizeof(path),
        )


def iter_t3_validation_arrays() -> Iterable[Candidate]:
    """Large generated NPY fixtures that can be recreated from probe scripts."""
    patterns = (
        "*_vulkan_output_*.npy",
        "*_torch_output_*.npy",
    )
    for pattern in patterns:
        for path in T3.rglob(pattern):
            yield Candidate(
                "t3_validation_arrays",
                "Generated validation output array; JSON summaries preserve correctness results.",
                path,
                sizeof(path),
            )


def iter_failed_hift_device_lost_artifacts() -> Iterable[Candidate]:
    """HiFT shapes that compiled but triggered RADV device-lost during execution."""
    hift_root = EXPORTS / "iree_vulkan_real_hift_core"
    failed_t722_names = {
        "real_hift_core_no_fft_t722_vulkan_gfx1013.vmfb",
        "speech_feat_t722.npy",
        "source_stft_t722.npy",
        "torch_output_t722.npy",
        "iree_vulkan_output_t722.npy",
    }
    for name in failed_t722_names:
        path = hift_root / name
        if not path.exists():
            continue
        yield Candidate(
            "failed_hift_device_lost",
            "The 722-frame HiFT Vulkan core triggered VK_ERROR_DEVICE_LOST on RADV; keep only the JSON failure record.",
            path,
            sizeof(path),
        )


def iter_hift_tail_probe_artifacts() -> Iterable[Candidate]:
    """HiFT tail probe artifacts not needed by the active compact-96 path."""
    hift_root = EXPORTS / "iree_vulkan_real_hift_core"
    probe_blob_names = {
        "speech_feat_t82.npy",
        "source_stft_t82.npy",
        "torch_output_t82.npy",
        "iree_vulkan_output_t82.npy",
        "speech_feat_t96.npy",
        "source_stft_t96.npy",
        "torch_output_t96.npy",
        "iree_vulkan_output_t96.npy",
        "real_hift_core_no_fft_t82_vulkan_gfx1013.vmfb",
    }
    for name in probe_blob_names:
        path = hift_root / name
        if not path.exists():
            continue
        yield Candidate(
            "hift_tail_probe_artifacts",
            "Generated HiFT tail probe blob; compact 96-frame tail keeps only the t96 VMFB and JSON records.",
            path,
            sizeof(path),
        )


def iter_masked_cache_debug_blobs() -> Iterable[Candidate]:
    """Masked-cache probes produce large reproducible fixtures; keep VMFBs/results."""
    masked_root = T3 / "masked_cache"
    if not masked_root.exists():
        return
    for path in masked_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in {".json", ".md", ".txt", ".vmfb"}:
            continue
        yield Candidate(
            "masked_cache_debug_blobs",
            "Generated masked-cache fixture/debug blob; VMFBs and JSON reports preserve the useful results.",
            path,
            sizeof(path),
        )


def iter_non_t3_mlir_sources() -> Iterable[Candidate]:
    """Generated MLIR outside T3 is reproducible and not needed at runtime."""
    for path in EXPORTS.rglob("*.mlir"):
        if T3 in path.parents:
            continue
        yield Candidate(
            "non_t3_mlir_sources",
            "Large generated MLIR source; VMFBs and JSON reports preserve the useful runtime artifacts.",
            path,
            sizeof(path),
        )


def iter_stale_export_dirs() -> Iterable[Candidate]:
    """Old experimental export directories that are no longer part of the active path."""
    for dirname, reason in STALE_EXPORT_DIRS.items():
        export_dir = EXPORTS / dirname
        if not export_dir.exists():
            continue
        for path in export_dir.rglob("*"):
            if path.is_file():
                yield Candidate(
                    "stale_experimental_exports",
                    reason,
                    path,
                    sizeof(path),
                )


def collect(profile: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    candidates.extend(iter_failed_large_fused_t3())
    candidates.extend(iter_old_t3_flag_variants())
    candidates.extend(iter_t3_validation_arrays())
    candidates.extend(iter_failed_hift_device_lost_artifacts())
    candidates.extend(iter_hift_tail_probe_artifacts())
    if profile == "aggressive":
        candidates.extend(iter_t3_mlir_with_vmfb())
        candidates.extend(iter_masked_cache_debug_blobs())
        candidates.extend(iter_non_t3_mlir_sources())
        candidates.extend(iter_stale_export_dirs())

    seen: set[Path] = set()
    unique = []
    for candidate in candidates:
        if candidate.path in seen:
            continue
        seen.add(candidate.path)
        unique.append(candidate)
    return sorted(unique, key=lambda item: (item.category, -item.size, item.path.as_posix()))


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


def summarize(candidates: list[Candidate]) -> dict:
    by_category: dict[str, dict] = {}
    for candidate in candidates:
        item = by_category.setdefault(candidate.category, {"files": 0, "bytes": 0})
        item["files"] += 1
        item["bytes"] += candidate.size
    return {
        "files": len(candidates),
        "bytes": sum(candidate.size for candidate in candidates),
        "human": human_size(sum(candidate.size for candidate in candidates)),
        "by_category": {
            category: {
                **data,
                "human": human_size(data["bytes"]),
            }
            for category, data in sorted(by_category.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=("conservative", "aggressive"),
        default="conservative",
        help="conservative keeps MLIR; aggressive also marks generated T3 MLIR for cleanup.",
    )
    parser.add_argument("--json", type=Path, help="Write full candidate report to JSON.")
    parser.add_argument("--top", type=int, default=30, help="Number of largest files to print.")
    parser.add_argument("--delete", action="store_true", help="Delete matched files.")
    parser.add_argument("--yes", action="store_true", help="Required with --delete.")
    args = parser.parse_args()

    candidates = collect(args.profile)
    summary = summarize(candidates)
    print(f"profile={args.profile}")
    print(f"candidate_files={summary['files']}")
    print(f"candidate_bytes={summary['bytes']} ({summary['human']})")
    for category, data in summary["by_category"].items():
        print(f"  {category}: {data['files']} files, {data['human']}")

    print("\nLargest candidates:")
    for candidate in sorted(candidates, key=lambda item: item.size, reverse=True)[: args.top]:
        print(f"{human_size(candidate.size):>9}  {candidate.category}  {candidate.path}")

    if args.json:
        report = {
            "profile": args.profile,
            "summary": summary,
            "candidates": [
                {
                    "category": candidate.category,
                    "reason": candidate.reason,
                    "path": candidate.path.as_posix(),
                    "size": candidate.size,
                    "human": human_size(candidate.size),
                }
                for candidate in candidates
            ],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.json}")

    if args.delete:
        if not args.yes:
            raise SystemExit("--delete requires --yes")
        for candidate in candidates:
            candidate.path.unlink()
        for path in sorted(EXPORTS.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        print(f"\nDeleted {len(candidates)} files ({summary['human']}).")
    else:
        print("\nDry run only. Re-run with --delete --yes to remove these files.")


if __name__ == "__main__":
    main()
