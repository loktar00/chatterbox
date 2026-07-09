#!/usr/bin/env bash
set -euo pipefail

cd /root/chatterbox

worker_port="${PORT:-8003}"

# These checks are intentionally non-generating. They verify the saved BC-250
# fast-path evidence, helper artifacts, safe CPU API, port state, and Vulkan
# device selection before the worker process is started.
./verify_bc250_safe_stack.py \
  --worker-port "${worker_port}" \
  --require-t3-validation-artifact

./preflight_vulkan_worker.py \
  --profile fast-fused-target \
  --worker-port "${worker_port}"

exec /root/chatterbox/run_api_vulkan_fast_fused.sh
