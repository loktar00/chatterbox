#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

export TMPDIR="${TMPDIR:-/root/tmp}"
mkdir -p "$TMPDIR"

min_free_gb="${MIN_FREE_GB:-25}"
free_kb="$(df --output=avail -k / | tail -1 | tr -d ' ')"
free_gb="$((free_kb / 1024 / 1024))"

echo "Free space on /: ${free_gb} GiB"
echo "TMPDIR: ${TMPDIR}"

if (( free_gb < min_free_gb )); then
  echo "Refusing ROCm venv install: ${free_gb} GiB free, need at least ${min_free_gb} GiB." >&2
  echo "Add disk or free space, then rerun this script." >&2
  exit 1
fi

if [[ -d /root/chatterbox/.venv-rocm && ! -x /root/chatterbox/.venv-rocm/bin/python ]]; then
  echo "Removing incomplete /root/chatterbox/.venv-rocm"
  rm -rf /root/chatterbox/.venv-rocm
fi

/root/.local/bin/uv venv --python 3.11 /root/chatterbox/.venv-rocm

# Keep this separate from the CPU fallback venv. ROCm 6.2.4 is the newest
# backend that resolves Chatterbox's torch==2.6.0 / torchaudio==2.6.0 pins.
/root/.local/bin/uv pip install \
  --python /root/chatterbox/.venv-rocm \
  --no-cache \
  -e . \
  --torch-backend rocm6.2.4

/root/chatterbox/.venv-rocm/bin/python - <<'PY'
import torch
import torchaudio
print("torch", torch.__version__)
print("torchaudio", torchaudio.__version__)
print("torch.version.hip", torch.version.hip)
PY
