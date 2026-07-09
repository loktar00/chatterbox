#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

# CPU fallback with the best measured thread settings and watermark disabled.
# This is separate from run_api.sh so the stable safe API can keep its original
# defaults unless intentionally replaced.
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export CHATTERBOX_DEVICE="${CHATTERBOX_DEVICE:-cpu}"
export CHATTERBOX_TORCH_THREADS="${CHATTERBOX_TORCH_THREADS:-2}"
export CHATTERBOX_TORCH_INTEROP_THREADS="${CHATTERBOX_TORCH_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CHATTERBOX_TORCH_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CHATTERBOX_TORCH_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CHATTERBOX_TORCH_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CHATTERBOX_TORCH_THREADS}}"
export CHATTERBOX_PROGRESS="${CHATTERBOX_PROGRESS:-0}"
export CHATTERBOX_APPLY_WATERMARK="${CHATTERBOX_APPLY_WATERMARK:-0}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}"

exec /root/chatterbox/.venv/bin/python -m uvicorn chatterbox_api:app --host 0.0.0.0 --port "${PORT:-8002}"
