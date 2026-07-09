#!/usr/bin/env python3
"""Verify Chatterbox API request contracts without generating audio."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv/bin/python"

if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(VENV_PYTHON.as_posix(), [VENV_PYTHON.as_posix(), __file__, *sys.argv[1:]])

import chatterbox_api


def accepts(model: type[Any], **kwargs: Any) -> bool:
    try:
        model(**kwargs)
    except Exception:
        return False
    return True


def check(name: str, ok: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def main() -> int:
    limit = chatterbox_api.MAX_INPUT_CHARS
    at_limit = "x" * limit
    over_limit = "x" * (limit + 1)
    health = chatterbox_api.health()

    checks = [
        check("max_input_chars_default_3000", limit == 3000, limit),
        check("tts_request_accepts_limit", accepts(chatterbox_api.TTSRequest, text=at_limit), limit),
        check("tts_request_rejects_over_limit", not accepts(chatterbox_api.TTSRequest, text=over_limit), limit + 1),
        check("speech_request_accepts_limit", accepts(chatterbox_api.SpeechRequest, input=at_limit), limit),
        check(
            "speech_request_rejects_over_limit",
            not accepts(chatterbox_api.SpeechRequest, input=over_limit),
            limit + 1,
        ),
        check("health_reports_limit", health.get("max_input_chars") == limit, health),
        check("contract_check_does_not_load_model", chatterbox_api.MODEL is None, None),
    ]
    errors = [item for item in checks if not item["ok"]]
    report = {
        "ok": not errors,
        "error_count": len(errors),
        "note": "No model load, audio generation, worker start, or ROCm/HIP probing is performed.",
        "checks": checks,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
