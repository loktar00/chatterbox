#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_UNSAFE_BC250_ROCM:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Refusing to run native HIP gfx1013 smoke test on BC-250.

This exact class of test was followed by a full host crash. ROCm compute on this
BC-250 is now considered unsafe in this environment.

Set ALLOW_UNSAFE_BC250_ROCM=1 only if you intentionally accept the risk of a
full host crash.
EOF
  exit 99
fi

cd /root/chatterbox
mkdir -p build

hipcc --offload-arch=gfx1013 hip_gfx1013_smoke.cpp -O2 -o build/hip_gfx1013_smoke

timeout "${HIP_SMOKE_TIMEOUT:-30}" ./build/hip_gfx1013_smoke
