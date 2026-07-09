# BC-250 Vulkan Chatterbox Commit Notes

This fork work is container-oriented and assumes the project lives at
`/root/chatterbox`. The goal was to make Chatterbox usable on a BC-250 without
ROCm/HIP compute, which proved unsafe on this host.

## Current Performance Baseline

- RTX 5090 reference from the external server: `3.495s`
- 2x target: `6.990s`
- BC-250 default-quality Vulkan/no-watermark path: about `8.145s`
- BC-250 fast-fused/no-watermark path: `6.899565461000748s`

The T3 native fast token-buffer path is validated by:

```bash
./validate_t3_native_fast_token_buffer.py
```

Latest local T3-only validation: two hello requests, both `48` tokens, token
equality preserved, `native_sampler=true`, `native_fast_token_buffer=true`, and
`step_logits_to_torch_seconds=0.0`. This check does not generate audio, start an
API worker, or run ROCm/HIP.

The fast-fused path meets the 2x target, but it is an opt-in
listen-before-default profile. It uses:

- `CHATTERBOX_APPLY_WATERMARK=0`
- `CHATTERBOX_S3_TIMESTEPS=1`
- `CHATTERBOX_T3_NATIVE_SAMPLER=1`
- `CHATTERBOX_ALLOW_VULKAN_S3_PADDING=1`
- `CHATTERBOX_REQUIRE_VULKAN_S3=1`
- `CHATTERBOX_T3_MAX_GEN_LEN=376`
- `CHATTERBOX_DEFAULT_SEED=20260708`
- `CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS=1`
- `CHATTERBOX_VULKAN_S3_FUSED_VARIANT=split8`

## Safe Startup

If this is a fresh clone, rebuild the ignored native helper libraries first.
This compile step does not run ROCm/HIP, load the model, or generate audio:

```bash
./build_vulkan_helpers.sh
```

The script expects a Vulkan-enabled llama.cpp build at
`/root/llama.cpp/build-vulkan`. Override `LLAMA_CPP_DIR`, `LLAMA_CPP_BUILD`,
`GGML_INCLUDE_DIR`, or `GGML_LIB_DIR` if your layout differs.

Before starting a worker or pushing branch updates, run the non-generating stack
gate:

```bash
./verify_bc250_safe_stack.py --require-t3-validation-artifact
```

It verifies syntax, ignored helper library linkage, ROCm/HIP guard scripts, safe
CPU API health, fast-fused preflight, saved target-crossing benchmark metadata,
the compact runtime matrix, and the latest T3 native fast token-buffer
validation artifact.

For a short exportability/runtime decision summary, run:

```bash
./summarize_bc250_runtime_matrix.py --pretty
```

It writes ignored JSON/Markdown under `exports/benchmarks/` and summarizes what
is active, experimental, rejected, or worth trying next.
The committed static version is `BC250_RUNTIME_MATRIX.md`.

Use the preflight before starting a Vulkan worker:

```bash
./preflight_vulkan_worker.py --profile fast-fused-target --worker-port 8003
PORT=8003 ./run_api_vulkan_fast_fused.sh
```

For a multi-BC-250 setup, bind each worker with `CHATTERBOX_VK_DEVICE_SELECT`
and place the router in front of the workers:

```bash
PORT=8003 CHATTERBOX_VK_DEVICE_SELECT=0000:01:00.0 ./run_api_vulkan_fast_fused.sh
PORT=8004 CHATTERBOX_VK_DEVICE_SELECT=0000:02:00.0 ./run_api_vulkan_fast_fused.sh
CHATTERBOX_ROUTER_BACKENDS=http://127.0.0.1:8003,http://127.0.0.1:8004 ./run_router.sh
```

## Safety

ROCm/HIP is intentionally not the supported path for this host. The ROCm API
and HIP verification scripts refuse to run unless `ALLOW_UNSAFE_BC250_ROCM=1`
is explicitly set.

The stable fallback API remains CPU-only:

```bash
./run_api.sh
```

## Not Committed

Generated artifacts are ignored and should be rebuilt or copied locally:

- `exports/`
- generated `.vmfb`, `.f32`, `.npy`, `.onnx`, and `.mlir` files
- compiled helper binaries
- local virtual environments
- generated WAV files

This keeps the fork small while preserving the source, scripts, and docs needed
to reproduce the current BC-250 Vulkan path.
