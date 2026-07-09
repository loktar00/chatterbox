#!/usr/bin/env python3
"""Safe ExecuTorch export probe.

The probe is intentionally tiny and CPU-only. It never imports Chatterbox or
loads model weights. If ExecuTorch is not installed, it records that fact and
exits successfully so the runtime matrix has explicit evidence.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path("/root/chatterbox")
OUT_JSON = ROOT / "exports/benchmarks/executorch_tiny_export_probe_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/executorch_tiny_export_probe_2026-07-08.md"


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


class TinyLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# ExecuTorch Tiny Export Probe - 2026-07-08",
        "",
        f"- Status: `{report['status']}`",
        f"- ExecuTorch available: `{report['executorch_available']}`",
        f"- EXIR available: `{report['executorch_exir_available']}`",
        f"- Torch export OK: `{report['torch_export'].get('ok')}`",
        f"- ExecuTorch edge export OK: `{report['executorch_edge_export'].get('ok')}`",
        "",
        "## Decision",
        "",
    ]
    if not report["executorch_available"]:
        lines.append(
            "- ExecuTorch is not installed in the stable Chatterbox environment. Do not install it here unless using a disposable venv/container."
        )
    elif not report["executorch_edge_export"].get("ok"):
        lines.append("- ExecuTorch is present but the tiny edge export did not pass; keep it out of the active runtime path.")
    else:
        lines.append("- ExecuTorch tiny edge export passed; it can be considered for future isolated subgraph probes.")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    started = time.perf_counter()
    report: dict[str, Any] = {
        "description": "Tiny CPU-only ExecuTorch availability/export probe. No Chatterbox model load.",
        "status": "not_started",
        "torch_version": torch.__version__,
        "executorch_available": module_available("executorch"),
        "executorch_exir_available": module_available("executorch.exir"),
        "torch_export": {"ok": False},
        "executorch_edge_export": {"ok": False},
        "seconds": None,
    }

    model = TinyLinear().eval()
    example = (torch.randn(2, 4),)
    try:
        exported = torch.export.export(model, example)
        report["torch_export"] = {
            "ok": True,
            "graph_node_count": len(list(exported.graph.nodes)),
        }
    except Exception as exc:
        report["status"] = "torch_export_failed"
        report["torch_export"] = {"ok": False, "error": repr(exc)}
        report["seconds"] = time.perf_counter() - started
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        print(f"json={args.out_json}")
        print(f"markdown={args.out_md}")
        return 0

    if not report["executorch_available"] or not report["executorch_exir_available"]:
        report["status"] = "executorch_not_installed"
        report["seconds"] = time.perf_counter() - started
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
        write_markdown(report)
        print(f"status={report['status']}")
        print(f"json={args.out_json}")
        print(f"markdown={args.out_md}")
        return 0

    try:
        exir = importlib.import_module("executorch.exir")
        to_edge = getattr(exir, "to_edge")
        edge_program = to_edge(exported)
        report["executorch_edge_export"] = {
            "ok": True,
            "type": type(edge_program).__name__,
        }
        report["status"] = "ok"
    except Exception as exc:
        report["status"] = "executorch_edge_export_failed"
        report["executorch_edge_export"] = {"ok": False, "error": repr(exc)}

    report["seconds"] = time.perf_counter() - started
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"status={report['status']}")
    print(f"json={args.out_json}")
    print(f"markdown={args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
