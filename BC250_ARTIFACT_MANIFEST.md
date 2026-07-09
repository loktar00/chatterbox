# BC-250 Artifact Manifest

This file documents the generated artifacts used by the BC-250 Vulkan
Chatterbox work. These artifacts are intentionally ignored by Git so the fork
stays small.

For the local evidence-backed version, run:

```bash
./summarize_bc250_artifacts.py --pretty
```

That command writes ignored JSON and Markdown under `exports/benchmarks/`. It
only inspects files; it does not load Chatterbox, generate audio, start workers,
compile exports, or probe ROCm/HIP.

## Required Runtime Artifacts

| Category | Path | Current Local Size | Purpose |
| --- | --- | ---: | --- |
| Native helper libraries | `libt3_*.so` | about `171KB` total | ctypes bridges for T3 ggml/Vulkan and native sampler |
| T3 ggml weights | `exports/ggml_t3_real_prompt_multistep_chunk270_s4_p935/` | about `1.8GB` | weights and manifest for the active ggml/Vulkan T3 runtime |
| S3 IREE/Vulkan artifacts | `exports/s3_flow_vulkan_components/` | about `4.3GB` | exact-bucket S3 encoder/estimator VMFBs and fused midblock variants |
| HiFT IREE/Vulkan artifacts | `exports/iree_vulkan_real_hift_core/` | about `1011MB` | chunked HiFT core VMFBs, including active `t128` and compact `t96` |

The native helper libraries are rebuilt from committed C++ sources:

```bash
./build_vulkan_helpers.sh
```

The helper library outputs remain ignored by `.gitignore`.

## Saved Evidence Artifacts

| Category | Path | Purpose |
| --- | --- | --- |
| Benchmark reports | `exports/benchmarks/` | saved timings, verifier output, and projection data |
| Audio checks | `exports/audio_checks/` | waveform sanity/comparison reports |
| WAV files | `exports/**/*.wav` | local listening/debug evidence only |

These are not required in a fresh fork checkout, but they are useful when
auditing how the current decisions were reached.

## Rejected Or Caution Artifacts

- ROCm/HIP environments and outputs are not part of the supported path.
- Whole-clip HiFT `t722` Vulkan execution is rejected because it previously hit
  RADV device loss.
- ncnn Vulkan is rejected on this host after a tiny smoke path failed.
- ExecuTorch should only be revisited in a disposable venv/container.

## Verification

Before starting a Vulkan worker or pushing fork updates, run:

```bash
./verify_bc250_safe_stack.py --require-t3-validation-artifact
```

That gate verifies helper library linkage, guard scripts, safe CPU API state,
fast-fused preflight, the runtime matrix summary, and the latest T3 native
fast-token-buffer validation artifact without generating audio or touching
ROCm/HIP.
