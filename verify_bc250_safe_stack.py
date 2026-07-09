#!/usr/bin/env python3
"""Verify the safe BC-250 Chatterbox Vulkan workbench state.

Default checks are non-generating: no audio synthesis, no API worker start, no
model load, and no ROCm/HIP probing. The script composes the status/preflight
tools and adds repository/runtime guard checks that are easy to run before
future optimization work or before pushing the fork branch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKER_PORT = 8003
HELPER_LIBS = (
    ROOT / "libt3_native_sampler_bridge.so",
    ROOT / "libt3_ggml_vulkan_bridge.so",
    ROOT / "libt3_ggml_vulkan_bridge_f16weights.so",
    ROOT / "libt3_ggml_vulkan_bridge_f16weights_range.so",
)
ROCM_GUARDED_SCRIPTS = (
    ROOT / "run_api_rocm.sh",
    ROOT / "verify_rocm_torch.sh",
    ROOT / "verify_native_hip_gfx1013.sh",
)


def run(cmd: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": str(exc)}


def json_command(cmd: list[str], timeout: float = 30.0) -> dict[str, Any]:
    result = run(cmd, timeout=timeout)
    parsed: dict[str, Any] | None = None
    if result["stdout"]:
        try:
            parsed = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            result["json_error"] = str(exc)
    result["json"] = parsed
    return result


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def check(name: str, ok: bool, detail: Any, severity: str = "error") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "detail": detail}


def helper_lib_checks() -> list[dict[str, Any]]:
    checks = []
    for lib in HELPER_LIBS:
        exists = lib.exists()
        checks.append(check("helper_lib_exists", exists, {"path": lib.as_posix()}))
        if not exists:
            continue
        rel_path = lib.relative_to(ROOT).as_posix()
        ignored = run(["git", "check-ignore", "-q", rel_path], timeout=5.0)
        checks.append(
            check(
                "helper_lib_ignored",
                ignored["returncode"] == 0,
                {"path": rel_path, "returncode": ignored["returncode"]},
            )
        )
        ldd = run(["ldd", lib.as_posix()], timeout=10.0)
        checks.append(
            check(
                "helper_lib_linked",
                ldd["returncode"] == 0 and "not found" not in ldd["stdout"],
                {
                    "path": lib.as_posix(),
                    "returncode": ldd["returncode"],
                    "missing": [line.strip() for line in ldd["stdout"].splitlines() if "not found" in line],
                },
            )
        )
    return checks


def rocm_guard_checks() -> list[dict[str, Any]]:
    checks = []
    for script in ROCM_GUARDED_SCRIPTS:
        text = script.read_text(errors="replace") if script.exists() else ""
        checks.append(check("rocm_script_exists", script.exists(), {"path": script.as_posix()}))
        checks.append(
            check(
                "rocm_script_guarded",
                "ALLOW_UNSAFE_BC250_ROCM" in text and "Refusing" in text,
                {"path": script.as_posix()},
            )
        )
    return checks


def source_syntax_checks() -> list[dict[str, Any]]:
    py_files = [
        "bundle_bc250_artifacts.py",
        "chatterbox_status.py",
        "preflight_vulkan_worker.py",
        "prepare_bc250_audio_review.py",
        "summarize_bc250_artifacts.py",
        "summarize_bc250_runtime_matrix.py",
        "verify_api_contract.py",
        "validate_t3_native_fast_token_buffer.py",
        "t3_ggml_vulkan_runtime.py",
        "chatterbox_api.py",
        "chatterbox_router.py",
    ]
    sh_files = [
        "build_vulkan_helpers.sh",
        "run_api.sh",
        "run_api_cpu_fast.sh",
        "run_api_rocm.sh",
        "run_api_vulkan_fast_fused_guarded.sh",
        "run_api_vulkan_fast_fused.sh",
        "run_api_vulkan_hybrid.sh",
        "run_router.sh",
        "verify_rocm_torch.sh",
        "verify_native_hip_gfx1013.sh",
    ]
    py = run([sys.executable, "-m", "py_compile", *py_files], timeout=30.0)
    sh = run(["bash", "-n", *sh_files], timeout=30.0)
    return [
        check("python_syntax", py["returncode"] == 0, {"stderr": py["stderr"]}),
        check("shell_syntax", sh["returncode"] == 0, {"stderr": sh["stderr"]}),
    ]


def runtime_matrix_summary_check() -> dict[str, Any]:
    output = Path("/tmp/chatterbox_bc250_runtime_matrix_verify.json")
    result = run(
        [
            "./summarize_bc250_runtime_matrix.py",
            "--output",
            output.as_posix(),
        ],
        timeout=15.0,
    )
    data = load_json_file(output)
    ok = (
        result["returncode"] == 0
        and bool(data)
        and nested(data, "safe_runtime", "cpu_health_ok") is True
        and isinstance(nested(data, "performance", "fast_fused_seconds"), (int, float))
        and len(data.get("components") or []) >= 8
    )
    return check(
        "runtime_matrix_summary_builds",
        ok,
        {
            "returncode": result["returncode"],
            "output": output.as_posix(),
            "component_count": len(data.get("components") or []) if data else 0,
            "decision": data.get("decision") if data else None,
            "stderr": result["stderr"],
        },
    )


def artifact_manifest_summary_check() -> dict[str, Any]:
    output = Path("/tmp/chatterbox_bc250_artifact_manifest_verify.json")
    result = run(
        [
            "./summarize_bc250_artifacts.py",
            "--output",
            output.as_posix(),
        ],
        timeout=15.0,
    )
    data = load_json_file(output)
    ok = (
        result["returncode"] == 0
        and bool(data)
        and nested(data, "t3_ggml_weights", "exists") is True
        and nested(data, "s3_iree_vulkan", "vmfb_count") >= 1
        and nested(data, "hift_iree_vulkan", "vmfb_count") >= 1
        and all(row.get("ignored") is True for row in data.get("helper_libraries", []))
    )
    return check(
        "artifact_manifest_summary_builds",
        ok,
        {
            "returncode": result["returncode"],
            "output": output.as_posix(),
            "t3_size": nested(data, "t3_ggml_weights", "size"),
            "s3_vmfb_count": nested(data, "s3_iree_vulkan", "vmfb_count"),
            "hift_vmfb_count": nested(data, "hift_iree_vulkan", "vmfb_count"),
            "stderr": result["stderr"],
        },
    )


def status_checks(status: dict[str, Any]) -> list[dict[str, Any]]:
    body = status.get("health", {}).get("8000", {}).get("body", {})
    ports = status.get("ports", {}).get("listeners", [])
    performance = status.get("performance_target", {})
    resources = status.get("resources", {})
    live_source = status.get("live_api_source", {})
    return [
        check("safe_api_alive", bool(status.get("health", {}).get("8000", {}).get("ok")), status.get("health", {}).get("8000")),
        check("safe_api_cpu_device", body.get("device") == "cpu", {"device": body.get("device")}),
        check(
            "live_safe_api_source_current",
            live_source.get("available") is True
            and live_source.get("matches_head") is True
            and live_source.get("live_dirty") is False,
            live_source,
            severity="warning",
        ),
        check(
            "only_safe_cpu_port_listening",
            all(":8000 " in line or ":8000\t" in line for line in ports) and bool(ports),
            {"listeners": ports},
        ),
        check("performance_target_met_by_saved_fast_profile", performance.get("target_met") is True, performance),
        check(
            "resource_headroom",
            (resources.get("disk_root", {}).get("free_gb") or 0) >= 8.0
            and (resources.get("memory", {}).get("MemAvailable") or 0) >= 2.0,
            resources,
            severity="warning",
        ),
    ]


def preflight_checks(preflight: dict[str, Any]) -> list[dict[str, Any]]:
    profile = preflight.get("profile", {})
    return [
        check("fast_fused_preflight_ok", preflight.get("ok") is True, {"error_count": preflight.get("error_count")}),
        check("fast_fused_preflight_no_errors", preflight.get("error_count") == 0, preflight.get("checks", [])),
        check(
            "fast_fused_threshold_met",
            ((profile.get("benchmark_threshold") or {}).get("ok") is True),
            profile.get("benchmark_threshold"),
        ),
        check(
            "preflight_did_not_start_worker",
            not any(f":{preflight.get('worker_port', DEFAULT_WORKER_PORT)} " in line for line in preflight.get("ports", {}).get("listeners", [])),
            preflight.get("ports"),
        ),
    ]


def latest_t3_validation() -> dict[str, Any]:
    paths = sorted((ROOT / "exports" / "benchmarks").glob("t3_native_fast_token_buffer_validation_*.json"))
    paths = [path for path in paths if not path.name.endswith("_benchmark.json")]
    if not paths:
        return {"exists": False, "path": None}
    path = paths[-1]
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"exists": True, "path": path.as_posix(), "error": str(exc)}
    return {
        "exists": True,
        "path": path.as_posix(),
        "ok": data.get("ok"),
        "native_fast_token_buffer": data.get("native_fast_token_buffer"),
        "all_tokens_equal_reference": data.get("all_tokens_equal_reference"),
        "no_logits_to_torch_copy": data.get("no_logits_to_torch_copy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-port", type=int, default=DEFAULT_WORKER_PORT)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--require-t3-validation-artifact", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = []
    git_status = run(["git", "status", "--short"], timeout=10.0)
    if args.require_clean_git:
        checks.append(check("git_worktree_clean", git_status["stdout"].strip() == "", git_status["stdout"]))

    checks.extend(source_syntax_checks())
    checks.append(artifact_manifest_summary_check())
    checks.append(runtime_matrix_summary_check())
    checks.extend(helper_lib_checks())
    checks.extend(rocm_guard_checks())

    api_contract_result = json_command(["./verify_api_contract.py"], timeout=30.0)
    api_contract = api_contract_result.get("json") or {}
    checks.append(
        check(
            "api_contract_command_json",
            api_contract_result["returncode"] == 0 and bool(api_contract),
            api_contract_result.get("json_error"),
        )
    )
    if api_contract:
        checks.append(
            check(
                "api_contract_3000_char_limit",
                api_contract.get("ok") is True and api_contract.get("error_count") == 0,
                api_contract.get("checks"),
            )
        )

    status_result = json_command(["./chatterbox_status.py"], timeout=15.0)
    status = status_result.get("json") or {}
    checks.append(check("status_command_json", status_result["returncode"] == 0 and bool(status), status_result.get("json_error")))
    if status:
        checks.extend(status_checks(status))

    preflight_result = json_command(
        [
            "./preflight_vulkan_worker.py",
            "--profile",
            "fast-fused-target",
            "--worker-port",
            str(args.worker_port),
            "--json",
        ],
        timeout=30.0,
    )
    preflight = preflight_result.get("json") or {}
    checks.append(
        check(
            "preflight_command_json",
            preflight_result["returncode"] == 0 and bool(preflight),
            preflight_result.get("json_error"),
        )
    )
    if preflight:
        checks.extend(preflight_checks(preflight))

    t3_validation = latest_t3_validation()
    checks.append(
        check(
            "latest_t3_native_fast_token_buffer_validation",
            (
                t3_validation.get("ok") is True
                and t3_validation.get("native_fast_token_buffer") is True
                and t3_validation.get("all_tokens_equal_reference") is True
                and t3_validation.get("no_logits_to_torch_copy") is True
            )
            or (not args.require_t3_validation_artifact and not t3_validation.get("exists")),
            t3_validation,
            severity="error" if args.require_t3_validation_artifact else "warning",
        )
    )

    error_count = sum(1 for item in checks if item["severity"] == "error" and not item["ok"])
    warning_count = sum(1 for item in checks if item["severity"] == "warning" and not item["ok"])
    report = {
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "note": "No audio generation, worker start, model load, or ROCm/HIP probing is performed.",
        "checks": checks,
    }
    text = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
