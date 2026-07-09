#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_UNSAFE_BC250_ROCM:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to run ROCm PyTorch smoke test on BC-250.

The previous native HIP gfx1013 kernel test was followed by a host crash. ROCm
compute on this BC-250 is now considered unsafe in this environment.

Set ALLOW_UNSAFE_BC250_ROCM=1 only if you intentionally accept the risk of a
full host crash.
EOF
  exit 99
fi

if [[ ! -x /root/chatterbox/.venv-rocm/bin/python ]]; then
  echo "Missing /root/chatterbox/.venv-rocm. Run /root/chatterbox/install_rocm_env.sh first." >&2
  exit 1
fi

if [[ ! -e /dev/kfd || ! -e /dev/dri/renderD128 ]]; then
  echo "Missing /dev/kfd or /dev/dri/renderD128. Fix LXC passthrough first." >&2
  exit 1
fi

cd /root/chatterbox

# Do not spoof gfx1013 as gfx1030 here. Community BC-250 docs warn that this
# crosses ISA families and can produce silent wrong results or instability.
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:128}"

timeout "${TORCH_VERIFY_TIMEOUT:-45}" /root/chatterbox/.venv-rocm/bin/python - <<'PY'
import torch

print("torch", torch.__version__)
print("hip", torch.version.hip)
print("cuda_is_available_for_rocm", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch cannot see a HIP device")

print("device_name", torch.cuda.get_device_name(0))

# Tiny bounded test: one allocation and one small matmul.
dev = torch.device("cuda:0")
a = torch.randn((64, 64), device=dev)
b = torch.randn((64, 64), device=dev)
c = a @ b
torch.cuda.synchronize()
print("matmul_sum", float(c.sum().cpu()))
PY
