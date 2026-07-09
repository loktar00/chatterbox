#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

# Explicit opt-in profile for the fastest measured BC-250 Vulkan path.
# This is listen-before-default: it changes the S3 step count and enables the
# request-seeded native T3 sampler, so keep the default-quality worker separate.
if [[ -n "${CHATTERBOX_VK_DEVICE_SELECT:-}" ]]; then
  export MESA_VK_DEVICE_SELECT="${CHATTERBOX_VK_DEVICE_SELECT}"
  export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="${MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE:-1}"
fi

export CHATTERBOX_APPLY_WATERMARK="${CHATTERBOX_APPLY_WATERMARK:-0}"
export CHATTERBOX_S3_TIMESTEPS="${CHATTERBOX_S3_TIMESTEPS:-1}"
export CHATTERBOX_T3_NATIVE_SAMPLER="${CHATTERBOX_T3_NATIVE_SAMPLER:-1}"
export CHATTERBOX_ALLOW_VULKAN_S3_PADDING="${CHATTERBOX_ALLOW_VULKAN_S3_PADDING:-1}"
export CHATTERBOX_REQUIRE_VULKAN_S3="${CHATTERBOX_REQUIRE_VULKAN_S3:-1}"
export CHATTERBOX_T3_MAX_GEN_LEN="${CHATTERBOX_T3_MAX_GEN_LEN:-376}"
export CHATTERBOX_DEFAULT_SEED="${CHATTERBOX_DEFAULT_SEED:-20260708}"

exec /root/chatterbox/run_api_vulkan_hybrid.sh
