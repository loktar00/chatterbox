#!/usr/bin/env bash
set -euo pipefail

ROOT="${CHATTERBOX_ROOT:-/root/chatterbox}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/root/llama.cpp}"
LLAMA_CPP_BUILD="${LLAMA_CPP_BUILD:-${LLAMA_CPP_DIR}/build-vulkan}"
GGML_INCLUDE_DIR="${GGML_INCLUDE_DIR:-${LLAMA_CPP_DIR}/ggml/include}"
GGML_LIB_DIR="${GGML_LIB_DIR:-${LLAMA_CPP_BUILD}/bin}"
CXX="${CXX:-g++}"

cd "$ROOT"

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_file "${GGML_INCLUDE_DIR}/ggml.h"
require_file "${GGML_INCLUDE_DIR}/ggml-alloc.h"
require_file "${GGML_INCLUDE_DIR}/ggml-backend.h"
require_file "${GGML_INCLUDE_DIR}/ggml-vulkan.h"
require_file "${GGML_LIB_DIR}/libggml.so"
require_file "${GGML_LIB_DIR}/libggml-base.so"
require_file "${GGML_LIB_DIR}/libggml-vulkan.so"
require_file "${ROOT}/t3_ggml_vulkan_bridge.cpp"
require_file "${ROOT}/t3_native_sampler_bridge.cpp"

common_flags=(
  -O3
  -march=native
  -std=c++17
  -fPIC
  -shared
)

ggml_flags=(
  -I"${GGML_INCLUDE_DIR}"
  -L"${GGML_LIB_DIR}"
  -Wl,-rpath,"${GGML_LIB_DIR}"
  -lggml
  -lggml-base
  -lggml-vulkan
)

build_shared() {
  local out="$1"
  shift
  local tmp="${out}.tmp.$$"
  echo "building ${out}"
  "$CXX" "$@" -o "$tmp"
  mv "$tmp" "$out"
}

build_shared \
  "${ROOT}/libt3_native_sampler_bridge.so" \
  "${common_flags[@]}" \
  "${ROOT}/t3_native_sampler_bridge.cpp"

build_shared \
  "${ROOT}/libt3_ggml_vulkan_bridge.so" \
  "${common_flags[@]}" \
  "${ROOT}/t3_ggml_vulkan_bridge.cpp" \
  "${ggml_flags[@]}"

build_shared \
  "${ROOT}/libt3_ggml_vulkan_bridge_f16weights.so" \
  "${common_flags[@]}" \
  -DCB_T3_F16_WEIGHTS \
  "${ROOT}/t3_ggml_vulkan_bridge.cpp" \
  "${ggml_flags[@]}"

# Historical runtime probes referenced this name. It is currently the same ABI
# and compile configuration as the active F16 bridge, so keep it as a copy for
# compatibility with old benchmark artifacts and ad hoc commands.
cp -f \
  "${ROOT}/libt3_ggml_vulkan_bridge_f16weights.so" \
  "${ROOT}/libt3_ggml_vulkan_bridge_f16weights_range.so"

echo "built native Vulkan helper libraries"
