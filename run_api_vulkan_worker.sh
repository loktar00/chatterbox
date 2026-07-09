#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

# Pin this worker to one Vulkan physical device when multiple BC-250s are
# visible. Use `MESA_VK_DEVICE_SELECT=list vulkaninfo --summary` to discover
# selectors such as `0000:01:00.0` or `1002:13fe`.
if [[ -n "${CHATTERBOX_VK_DEVICE_SELECT:-}" ]]; then
  export MESA_VK_DEVICE_SELECT="${CHATTERBOX_VK_DEVICE_SELECT}"
  export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="${MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE:-1}"
fi

exec /root/chatterbox/run_api_vulkan_hybrid.sh
