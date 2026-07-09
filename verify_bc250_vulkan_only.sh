#!/usr/bin/env bash
set -euo pipefail

echo "== DRM nodes =="
ls -la /dev/dri 2>&1 || true
echo

echo "== Vulkan devices =="
vulkaninfo --summary 2>&1 | sed -n '/Devices:/,$p'
echo

if vulkaninfo --summary 2>&1 | grep -q 'AMD BC-250 (RADV GFX1013)'; then
  echo "bc250_vulkan=ok"
else
  echo "bc250_vulkan=missing" >&2
  exit 1
fi
