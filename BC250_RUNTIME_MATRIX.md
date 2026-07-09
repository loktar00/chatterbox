# BC-250 Runtime Matrix

This is the committed, human-readable runtime/exportability summary for the
BC-250 Chatterbox fork work. It captures the current decisions without requiring
the ignored `exports/` benchmark tree to be present in a fresh clone.

For the local, evidence-backed version generated from saved artifacts, run:

```bash
./summarize_bc250_runtime_matrix.py --pretty
```

That command writes ignored JSON and Markdown under `exports/benchmarks/`. It
does not load the model, generate audio, start workers, or probe ROCm/HIP.

## Performance Snapshot

| Path | Status | Time |
| --- | --- | ---: |
| RTX 5090 reference | external comparison server | `3.495s` |
| 2x target | target for BC-250 | `6.990s` |
| BC-250 fast-fused Vulkan | target-crossing, listen-before-default | `6.899565461000748s` |
| BC-250 default-quality Vulkan, no watermark | above target | about `8.145s` |
| BC-250 CPU fallback, no watermark projection | too slow for target | about `59.964s` |

The fast-fused path meets the 2x target from saved benchmark evidence, but it is
not the default-quality path. It uses one-step S3, request-seeded native T3
sampling, padded required S3 buckets, fused split8 S3 midblocks, no watermark,
and a capped T3 generation length.

## Component Matrix

| Component | Runtime | Status | Current Decision |
| --- | --- | --- | --- |
| Safe CPU API | PyTorch CPU | Active fallback | Keep running on `:8000` while probing acceleration elsewhere. |
| CPU fast fallback | PyTorch CPU, tuned threads, no watermark | Available separate fallback | Useful as a stable baseline, but not close to target latency. |
| T3 token model | ggml + Vulkan | Active experimental path | Keep. The remaining T3 bottleneck is mostly ggml per-token wall time. |
| T3 native sampler | C++ bridge through ctypes | Active in fast paths | Keep. T3-only validation preserved token equality and avoided logits-to-Torch copy. |
| S3 flow | IREE + Vulkan | Active for exact buckets | Keep exact buckets to avoid CPU fallback. Fused split8 midblocks are opt-in. |
| HiFT vocoder | IREE + Vulkan chunked core | Active chunked path | Keep 128-frame chunks plus compact96 tail. |
| Voice encoder / conditionals | CPU, ONNX inspection possible | Not on steady-state hot path | Leave on CPU unless future profiling shows it matters. |
| ncnn | pnnx/ncnn CPU conversion only | Vulkan rejected on this host | Do not spend more BC-250 time on ncnn Vulkan here. |
| ExecuTorch | torch.export probe only | Not installed in stable env | Revisit only in a disposable venv/container. |
| Router / multi-GPU | HTTP router over one worker per GPU | Throughput path | Use for concurrent requests or chunked long text, not single-chunk latency. |

## Active Fast-Fused Profile

Use this only after the non-generating verifier passes:

```bash
./verify_bc250_safe_stack.py --require-t3-validation-artifact
./preflight_vulkan_worker.py --profile fast-fused-target --worker-port 8003
PORT=8003 ./run_api_vulkan_fast_fused_guarded.sh
```

The guarded launcher runs the non-generating verifier and preflight before it
execs the fast-fused launcher. The fast-fused launcher defaults:

- `CHATTERBOX_APPLY_WATERMARK=0`
- `CHATTERBOX_S3_TIMESTEPS=1`
- `CHATTERBOX_T3_NATIVE_SAMPLER=1`
- `CHATTERBOX_ALLOW_VULKAN_S3_PADDING=1`
- `CHATTERBOX_REQUIRE_VULKAN_S3=1`
- `CHATTERBOX_T3_MAX_GEN_LEN=376`
- `CHATTERBOX_DEFAULT_SEED=20260708`
- `CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS=1`
- `CHATTERBOX_VULKAN_S3_FUSED_VARIANT=split8`

## Audio Status

Saved audio sanity checks show the fast-fused output passes basic waveform
sanity and is effectively identical to the prior fast path. It differs from the
default-quality no-watermark output, so it remains listen-before-default until
the generated audio has been reviewed by ear.

Use `./prepare_bc250_audio_review.py` to build an ignored local HTML review page
from the saved WAV artifacts.

## Guardrails

- Do not use ROCm/HIP on this BC-250 host. The ROCm scripts require
  `ALLOW_UNSAFE_BC250_ROCM=1` and should stay guarded.
- Do not retry whole-clip HiFT `t722` Vulkan execution; it previously hit RADV
  device loss. Use chunked HiFT.
- Do not repeat ncnn Vulkan probes on this host; the tiny Vulkan smoke path was
  rejected.
- Do not install ExecuTorch into the stable Chatterbox venv. Use a disposable
  environment if revisiting it.
- Keep generated artifacts out of the fork. Rebuild helper libraries with
  `./build_vulkan_helpers.sh`; keep `exports/`, WAVs, VMFBs, and local venvs
  ignored.

## Next Useful Work

1. Listen to the fast-fused output before making it a default profile.
2. Continue S3 estimator fusion or dispatch reduction; S3 estimator cost is the
   best remaining Vulkan target.
3. Continue T3 ggml token-loop work only if it reduces per-token wall time.
4. Use router parallelism for throughput across multiple BC-250s or long-text
   chunks.
