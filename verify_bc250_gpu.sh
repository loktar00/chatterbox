#!/usr/bin/env bash
set -u

echo "== Device nodes =="
ls -la /dev/kfd /dev/dri 2>&1 || true
echo

echo "== Sysfs device ids =="
cat /sys/class/drm/card1/device/uevent 2>/dev/null | sed -n '1,12p' || true
printf 'kfd dev: '; cat /sys/class/kfd/kfd/dev 2>/dev/null || true
printf 'render dev: '; cat /sys/class/drm/renderD128/dev 2>/dev/null || true
echo

echo "== ROCm agent enumerator =="
timeout 10 rocm_agent_enumerator 2>&1 || true
echo

echo "== rocminfo smoke test =="
timeout 15 rocminfo 2>&1 | sed -n '1,120p' || true
echo

echo "== rocm-smi smoke test =="
timeout 15 rocm-smi 2>&1 | sed -n '1,120p' || true
echo

echo "== Vulkan summary =="
timeout 15 vulkaninfo --summary 2>&1 | sed -n '/Devices:/,$p' || true
