#!/usr/bin/env bash
set -euo pipefail

MESA_VK_DEVICE_SELECT=list vulkaninfo --summary
