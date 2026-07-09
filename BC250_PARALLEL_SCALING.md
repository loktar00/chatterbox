# BC-250 Parallel Scaling Notes

## Short Answer

Parallel generation across two BC-250s is practical for throughput. It is not a
clean way to make one normal chunk270 request run twice as fast.

Watermark-disabled generation is acceptable for this deployment target. That is
now the reference path for the experimental Vulkan worker because it removes a
small CPU/post-processing cost without changing the core TTS model path.

The best shape is one independent Chatterbox worker per GPU:

- Worker A: BC-250 A, API port such as `8003`.
- Worker B: BC-250 B, API port such as `8004`.
- A small router sends independent requests, or long-text chunks, to the next
  available worker.

## What Scales Well

Independent requests scale well. Two GPUs should roughly double throughput if each
worker has its own BC-250 and enough CPU/RAM headroom.

Long text can also be chunked and distributed:

1. Split text into reliable chunks, currently around the known `270` character
   reliability envelope.
2. Send chunk 0 to worker A, chunk 1 to worker B, chunk 2 to whichever finishes
   first, and so on.
3. Concatenate returned WAVs in order, optionally with a small silence/crossfade.

This can reduce total wall time for long inputs, but it may change prosody at chunk
boundaries. It should be treated as a product/runtime feature, not as a model-level
single-utterance acceleration.

## What Does Not Scale Cleanly

A single chunk270 generation does not split cleanly across two GPUs:

- T3 token generation is autoregressive and serialized.
- S3 depends on the generated token sequence and currently runs as a two-step flow.
- HiFT chunks can be split internally, but preserving boundaries and context already
  requires careful windowing; cross-GPU dispatch overhead would likely erase most of
  the gain for one short request.

## Recommended Deployment Shape

Use one worker process per GPU and make each worker see only one Vulkan device if
possible. This avoids relying on fragile Vulkan device selection inside IREE/ggml.

For multiple BC-250s in one host, prefer either:

- separate containers, each passed one `/dev/dri/renderD*` device; or
- separate hosts/nodes, each running the same API and registering with a router.

If multiple BC-250s are visible inside the same container, pin each worker with
Mesa's Vulkan device-select layer. List selectors with:

```bash
./list_vulkan_devices.sh
```

This container currently reports one BC-250 selector:

```text
GPU 0: 1002:13fe "AMD BC-250 (RADV GFX1013)" integrated GPU 0000:01:00.0
```

The Mesa layer in this container supports
`MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1`, which makes the selected device
the only enumerated Vulkan physical device. Verified locally:

```bash
MESA_VK_DEVICE_SELECT=0000:01:00.0 \
MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1 \
vulkaninfo --summary
```

Worker launcher:

```bash
./preflight_vulkan_worker.py --profile default-quality --device-selector 0000:01:00.0 --worker-port 8003

PORT=8003 \
CHATTERBOX_VK_DEVICE_SELECT=0000:01:00.0 \
./run_api_vulkan_worker.sh
```

On a multi-card container, launch one worker per selector and port:

```bash
./preflight_vulkan_worker.py --profile default-quality --device-selector 0000:01:00.0 --worker-port 8003
./preflight_vulkan_worker.py --profile default-quality --device-selector 0000:02:00.0 --worker-port 8004

PORT=8003 CHATTERBOX_VK_DEVICE_SELECT=0000:01:00.0 ./run_api_vulkan_worker.sh
PORT=8004 CHATTERBOX_VK_DEVICE_SELECT=0000:02:00.0 ./run_api_vulkan_worker.sh
```

If two BC-250s are in separate containers or hosts, skip device-select inside the
worker and give the router each worker URL.

The preflight is intentionally non-generating. It does not load Chatterbox, call
ROCm/HIP tools, or touch the model pipeline. It checks:

- selected Vulkan device resolves to BC-250/RADV;
- forced device selection hides llvmpipe for that worker;
- safe CPU API is still alive;
- worker port is free;
- basic disk and memory headroom;
- saved Vulkan benchmark baselines for comparison;
- saved CPU fallback thread-tuning and chunk270 baseline.

For the experimental Vulkan launcher, no-watermark is now accepted for speed:

```bash
CHATTERBOX_APPLY_WATERMARK=0 PORT=8003 ./run_api_vulkan_hybrid.sh
```

The local API fallback paths also accept the same flag now. The safe CPU launcher
still defaults `CHATTERBOX_APPLY_WATERMARK=1`; the Vulkan hybrid launcher defaults
it to `0`.

For the measured target-crossing fast profile, use the separate fused fast
launcher:

```bash
./preflight_vulkan_worker.py --profile fast-fused-target --worker-port 8003
PORT=8003 ./run_api_vulkan_fast_fused.sh
```

That script keeps watermark off and opts into the measured fast recipe:
`CHATTERBOX_S3_TIMESTEPS=1`, `CHATTERBOX_T3_NATIVE_SAMPLER=1`,
`CHATTERBOX_ALLOW_VULKAN_S3_PADDING=1`, `CHATTERBOX_REQUIRE_VULKAN_S3=1`,
`CHATTERBOX_T3_MAX_GEN_LEN=376`, `CHATTERBOX_DEFAULT_SEED=20260708`,
`CHATTERBOX_VULKAN_S3_FUSED_MIDBLOCKS=1`, and
`CHATTERBOX_VULKAN_S3_FUSED_VARIANT=split8`. Its best saved warm request is
`6.900s` total / `6.903s` wall against the `6.990s` 2x-5090 target.
Treat it as listen-before-default, not default-equivalent.

The older `fast-target` profile remains useful as a non-fused comparison. Its
best saved warm request was `6.944s` wall.

Keep one request in flight per worker until we explicitly test concurrency. Current
benchmarking assumes sequential generation per process.

## Router

This repo now includes a small single-flight router:

- `chatterbox_router.py`
- `run_router.sh`

Example with two workers:

```bash
PORT=8010 \
CHATTERBOX_ROUTER_BACKENDS=http://127.0.0.1:8003,http://127.0.0.1:8004 \
./run_router.sh
```

For fused fast-profile workers on multiple BC-250s:

```bash
./preflight_vulkan_worker.py --profile fast-fused-target --device-selector 0000:01:00.0 --worker-port 8003
./preflight_vulkan_worker.py --profile fast-fused-target --device-selector 0000:02:00.0 --worker-port 8004

PORT=8003 CHATTERBOX_VK_DEVICE_SELECT=0000:01:00.0 ./run_api_vulkan_fast_fused.sh
PORT=8004 CHATTERBOX_VK_DEVICE_SELECT=0000:02:00.0 ./run_api_vulkan_fast_fused.sh
```

The router exposes the same client-facing endpoints used by the benchmark:

- `GET /health`
- `GET /voices`
- `POST /audio/speech`
- `POST /audio/speech/chunked`

It sends at most one `/audio/speech` request to each backend at a time and rotates
through available workers. Health and voices are proxied without generating audio.

`/audio/speech/chunked` splits the input into reliable-size chunks, sends chunks
to available workers in parallel, and stitches the returned WAV files back
together in the original order. This is the practical path for speeding up one
long text input across multiple BC-250s. It does not make a single normal
chunk270 utterance faster.

Router chunking controls:

```bash
export CHATTERBOX_ROUTER_CHUNK_CHARS=270
export CHATTERBOX_ROUTER_CHUNK_SILENCE_MS=0
```

The normal `/audio/speech` endpoint keeps its existing single-request behavior by
default. To make it automatically chunk long text, opt in:

```bash
export CHATTERBOX_ROUTER_CHUNK_LONG_TEXT=1
```

For a non-generating snapshot of ports, health, disk, memory, GPU device nodes, and
saved benchmark timings:

```bash
./chatterbox_status.py --pretty
```

Smoke validation on this container:

- Router started on `:8010` against the safe backend on `:8000`.
- `GET /health` returned `ok: true`.
- `GET /voices` returned `["default"]`.
- The router was then stopped; only the safe API on `:8000` remained listening.

## Current Single-Worker Reference

Current 32-enabled / 24-reported BC-250:

- Default quality, watermark retained: `8.395s`.
- Default quality, no watermark: `8.145s`.
- Fast listening candidate, no default-quality equivalence: `7.113s` with
  `CHATTERBOX_S3_TIMESTEPS=1` and `CHATTERBOX_APPLY_WATERMARK=0`.
- Fast target-crossing opt-in candidate: `6.944s` with
  `CHATTERBOX_S3_TIMESTEPS=1`, `CHATTERBOX_T3_NATIVE_SAMPLER=1`,
  request-scoped native sampling seed, padded Vulkan S3, and no watermark.

Two workers should improve aggregate throughput. For example, two no-watermark
workers should process two chunk270 requests in roughly the time one worker handles
one request, assuming both GPUs are stable and independently assigned.

For one long input, `/audio/speech/chunked` can lower wall time by distributing
chunks across workers. For one short chunk270 request, the expected latency is
still the single-worker latency because T3 token generation is serialized.

Current target context:

- RTX 5090 reference chunk270: `3.495s`.
- 2x target: `6.990s`.
- Current BC-250 default-quality no-watermark: about `8.145s`.
- Current fast listen-before-default no-watermark: about `7.113s`.
- Request-seeded native sampler plus padded S3 completed the guarded two-request
  trial at `8.017s`, a small `1.016x` speedup over the no-watermark baseline.
  It remains opt-in/listen-before-default because the waveform differs from the
  baseline despite passing basic audio sanity.
- Request-seeded native sampler plus one-step S3 completed the guarded
  two-request trial at `6.944s`, which is `1.99x` the 5090 reference and just
  under the `6.990s` target. It also remains opt-in/listen-before-default
  because its waveform differs from both default and the prior fast candidate.
