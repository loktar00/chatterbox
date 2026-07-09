#!/usr/bin/env python3
"""Print a non-generating BC-250 Chatterbox safe-state summary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PATCH_PREFIX = "/root/chatterbox-bc250-vulkan-accel"
WATCH_PORTS = (8000, 8002, 8003, 8004, 8010, 4123, 8020)


def run(cmd: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def http_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return {"ok": True, "status": resp.status, "body": json.loads(body)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def http_status(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()
            return {"ok": True, "status": resp.status, "bytes": len(body)}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def git_info() -> dict[str, Any]:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10.0)
    commit = run(["git", "rev-parse", "--short", "HEAD"], timeout=10.0)
    status = run(["git", "status", "--short"], timeout=10.0)
    return {
        "branch": branch["stdout"].strip() if branch["returncode"] == 0 else None,
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "dirty": bool(status["stdout"].strip()) if status["returncode"] == 0 else None,
        "status": status["stdout"].strip(),
    }


def listeners() -> list[str]:
    result = run(["ss", "-ltnp"], timeout=10.0)
    lines = []
    for line in result["stdout"].splitlines():
        if any(f":{port} " in line or f":{port}\t" in line for port in WATCH_PORTS):
            lines.append(line)
    return lines


def host_ip() -> str | None:
    result = run(["hostname", "-I"], timeout=5.0)
    if result["returncode"] != 0:
        return None
    return (result["stdout"].strip().split() or [None])[0]


def verifier(allow_dirty: bool) -> dict[str, Any]:
    cmd = ["./verify_bc250_safe_stack.py", "--require-t3-validation-artifact"]
    if not allow_dirty:
        cmd.insert(1, "--require-clean-git")
    result = run(cmd, timeout=45.0)
    parsed: dict[str, Any] | None = None
    if result["stdout"]:
        try:
            parsed = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            result["json_error"] = str(exc)
    return {
        "returncode": result["returncode"],
        "ok": bool(parsed and parsed.get("ok") is True),
        "error_count": parsed.get("error_count") if parsed else None,
        "warning_count": parsed.get("warning_count") if parsed else None,
        "json_error": result.get("json_error"),
    }


def backup_artifacts(commit: str | None) -> dict[str, Any]:
    if not commit:
        return {"patch": None, "bundle": None}
    patch = Path(f"{PATCH_PREFIX}-{commit}.patch")
    bundle = Path(f"{PATCH_PREFIX}-{commit}.bundle")
    return {
        "patch": {"path": patch.as_posix(), "exists": patch.exists(), "size_bytes": patch.stat().st_size if patch.exists() else 0},
        "bundle": {
            "path": bundle.as_posix(),
            "exists": bundle.exists(),
            "size_bytes": bundle.stat().st_size if bundle.exists() else 0,
        },
    }


def build_report(allow_dirty: bool) -> dict[str, Any]:
    git = git_info()
    ip = host_ip()
    review_path = "/exports/audio_review/index.html"
    return {
        "note": "No audio generation, model load, worker start, or ROCm/HIP probing is performed.",
        "git": git,
        "verifier": verifier(allow_dirty=allow_dirty),
        "health_8000": http_json("http://127.0.0.1:8000/health"),
        "audio_review": {
            "local_url": f"http://127.0.0.1:8020{review_path}",
            "remote_url": f"http://{ip}:8020{review_path}" if ip else None,
            "status": http_status(f"http://127.0.0.1:8020{review_path}"),
        },
        "listeners": listeners(),
        "backups": backup_artifacts(git.get("commit")),
    }


def print_text(report: dict[str, Any]) -> None:
    health = report["health_8000"].get("body", {}) if report["health_8000"].get("ok") else {}
    print("BC-250 safe status")
    print(f"branch={report['git'].get('branch')} commit={report['git'].get('commit')} dirty={report['git'].get('dirty')}")
    print(
        "verifier="
        f"ok={report['verifier'].get('ok')} "
        f"errors={report['verifier'].get('error_count')} "
        f"warnings={report['verifier'].get('warning_count')}"
    )
    print(
        "safe_cpu="
        f"ok={report['health_8000'].get('ok')} "
        f"device={health.get('device')} "
        f"source={health.get('source_commit')} "
        f"dirty={health.get('source_dirty')}"
    )
    print(f"audio_review={report['audio_review'].get('remote_url')} status={report['audio_review']['status'].get('status')}")
    print("listeners:")
    for line in report["listeners"]:
        print(f"  {line}")
    print(f"patch={report['backups']['patch']['path']} exists={report['backups']['patch']['exists']}")
    print(f"bundle={report['backups']['bundle']['path']} exists={report['backups']['bundle']['exists']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(allow_dirty=args.allow_dirty)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 0 if report["verifier"].get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
