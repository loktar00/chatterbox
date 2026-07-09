#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_UNSAFE_BC250_ROCM:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to start Chatterbox ROCm API on BC-250.

ROCm compute on this BC-250 caused or coincided with a full host crash during a
minimal native HIP gfx1013 test. Leave this disabled unless you intentionally
accept that risk.
EOF
  exit 99
fi

cd /root/chatterbox
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export CHATTERBOX_DEVICE="${CHATTERBOX_DEVICE:-cuda}"
export CHATTERBOX_TORCH_THREADS="${CHATTERBOX_TORCH_THREADS:-2}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

exec /root/chatterbox/.venv-rocm/bin/python -m uvicorn chatterbox_api:app --host 0.0.0.0 --port "${PORT:-8001}"
