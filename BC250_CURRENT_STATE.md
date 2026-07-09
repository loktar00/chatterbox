# BC-250 Current State

This is the stopping-point audit for the BC-250 Vulkan Chatterbox branch. It is
based on saved benchmark artifacts plus non-generating status checks.

## Summary

- Safe CPU API remains the live fallback on port `8000`.
- ROCm/HIP is not the supported path and remains guarded behind explicit unsafe
  opt-in.
- Fast-fused Vulkan is the best measured BC-250 path and meets the 2x RTX 5090
  target, but it remains listen-before-default.
- The branch is fork-ready once GitHub credentials are available.

## Performance

| Path | Status | Time |
| --- | --- | ---: |
| RTX 5090 reference | external comparison | `3.495s` |
| 2x target | BC-250 goal | `6.990s` |
| BC-250 fast-fused Vulkan | target-crossing, opt-in | `6.899565461000748s` |
| BC-250 default-quality Vulkan, no watermark | stable but above target | about `8.145s` |
| BC-250 CPU fallback, no watermark projection | stable fallback only | about `59.964s` |

Fast-fused uses one-step S3, request-seeded native T3 sampling, padded required
S3 buckets, fused split8 S3 midblocks, no watermark, and capped T3 generation
length.

## Objective Coverage

| Objective | Current State | Evidence |
| --- | --- | --- |
| Tighten CPU performance | Best measured CPU thread setting is `2/1`; CPU fast launcher disables watermark. CPU remains too slow for target latency. | `exports/cpu_thread_bench/cpu_fast_path_status_2026-07-08.json`, `run_api_cpu_fast.sh` |
| Identify exportable pieces | Runtime/exportability matrix records active, rejected, and revisit-later components. | `BC250_RUNTIME_MATRIX.md`, `summarize_bc250_runtime_matrix.py` |
| Try Vulkan subgraph runtimes | IREE/Vulkan powers S3 and HiFT subgraphs; ggml/Vulkan powers T3; ncnn Vulkan rejected here; ExecuTorch deferred to disposable env only. | `BC250_RUNTIME_MATRIX.md`, `BC250_VULKAN_PORT_PLAN.md` |
| Keep safe CPU API running | Health is OK on `:8000`; no Vulkan worker/router ports are open. | `./verify_bc250_safe_stack.py --require-t3-validation-artifact`, `ss -ltnp` |

## Known Warnings

- The currently running CPU API predates source-commit metadata in `/health`.
  This is expected until the CPU API is intentionally restarted.
- The fast-fused path passed waveform sanity and matches the prior fast path,
  but it still needs human listening before becoming the default.
- This fresh container cannot push to GitHub until credentials, `gh`, or an SSH
  key are added.

## Listen-Test Package

Prepare a local review page from existing WAV artifacts:

```bash
./prepare_bc250_audio_review.py
```

Open `exports/audio_review/index.html` in a browser or copy the referenced WAVs
to a listening machine. The script only inspects existing artifacts; it does not
load Chatterbox, generate audio, start workers, or touch ROCm/HIP.

To serve the review page for another machine:

```bash
PORT=8020 ./serve_bc250_audio_review.sh
```

Then open `http://<container-ip>:8020/exports/audio_review/index.html`.

## Safe Commands

Run the full non-generating gate:

```bash
./verify_bc250_safe_stack.py --require-t3-validation-artifact
```

The gate also inspects live processes and fails if ROCm/HIP scripts, Vulkan
workers, router workers, or non-`8000` Chatterbox API workers are running.

Dry-run ignored artifact preservation:

```bash
./bundle_bc250_artifacts.py --mode runtime-evidence
```

Start the opt-in fast-fused worker only after the gate passes:

```bash
PORT=8003 ./run_api_vulkan_fast_fused_guarded.sh
```

Do not run ROCm/HIP on this host as part of normal Chatterbox work.
