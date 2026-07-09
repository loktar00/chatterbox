# BC-250 Vulkan path for Chatterbox

## Current conclusion

ROCm/HIP compute is not a viable path on this host right now.

Evidence:

- `/dev/kfd` and `/dev/dri` passthrough works.
- `rocminfo` sees the BC-250 as `gfx1013`.
- `vulkaninfo` sees `AMD BC-250 (RADV GFX1013)`.
- PyTorch ROCm sees one device, but the ROCm wheel was built for:
  `gfx900 gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101`.
- It does not include `gfx1013`.
- A tiny native HIP kernel compiled for actual `gfx1013` was followed by a host
  crash/reboot.

Do not continue ROCm testing unless this host is expendable.

## What makes Chatterbox hard to move to Vulkan

Chatterbox is not just one transformer. The Turbo pipeline is:

1. Reference/audio conditioning:
   - librosa/torchaudio resampling
   - mel extraction
   - voice encoder
   - S3Tokenizer
2. T3 token model:
   - GPT2/Llama-style transformer
   - custom text/speech embeddings
   - autoregressive token generation
   - custom sampling/logits processors
3. S3Gen decoder/vocoder:
   - conformer/attention blocks
   - conditional flow/mel decoder
   - HiFiGAN-style waveform generator
4. Perth watermarking on CPU

llama.cpp Vulkan is good at GGUF transformer inference. It does not run this
whole PyTorch audio pipeline, and the T3 transformer is only one part of the
system.

## Candidate routes

### Route A: keep Chatterbox CPU, use BC-250 for other Vulkan workloads

Lowest risk. This is the current safe state.

### Route B: split acceleration

Port only the T3 token model to a Vulkan-capable transformer runtime, while
leaving S3Gen/vocoder on CPU.

Hard parts:

- T3 is not a stock text LLM. It has custom conditioning embeddings and speech
  token heads.
- The runtime must accept precomputed input embeddings or be modified to do so.
- The output is speech-token logits, not normal text tokens.
- Sampling must match Chatterbox behavior well enough to feed S3Gen.

Potential engine families:

- llama.cpp/ggml Vulkan, if the T3 GPT2-like transformer can be represented as
  a custom GGUF model and custom inference loop.
- MLC/TVM Vulkan, if T3 can be lowered as a custom transformer graph.

This is a real port, not a package install.

### Route C: compile static subgraphs to Vulkan

Use IREE/torch-mlir/ExecuTorch-style AOT compilation for pieces of the PyTorch
graph.

Potentially portable pieces:

- voice encoder
- some convolutional/vocoder submodules
- fixed-shape parts of S3Gen

Hard parts:

- dynamic autoregressive generation
- Python control flow
- Hugging Face transformer cache behavior
- custom tokenizer/resampler/mel logic
- broad PyTorch operator coverage

This is also a real port. It should start by compiling one small submodule, not
the whole TTS pipeline.

### Route D: ONNX to NCNN/Vulkan

Possible for isolated feed-forward modules, but unlikely for the complete
pipeline without model surgery.

Hard parts:

- dynamic generation loop
- unsupported PyTorch ops in converter stacks
- attention/cache semantics
- multiple model components and preprocessing steps

## Recommended next experiment

Do not run more ROCm.

## Progress on 2026-07-07

CPU tuning:

- Baseline CPU-vs-RTX-5090 benchmark is saved at
  `/root/chatterbox/exports/benchmarks/baseline_cpu_vs_5090_2026-07-07.json`.
- Local CPU is about 18x slower by request wall time and about 22x-24x slower
  by generated-audio throughput on the measured prompts.
- Thread benchmark is saved at
  `/root/chatterbox/exports/cpu_thread_bench/cpu_thread_benchmark.json`.
- Best measured CPU setting so far is:
  `CHATTERBOX_TORCH_THREADS=2` and `CHATTERBOX_TORCH_INTEROP_THREADS=1`.
- The systemd service and `run_api.sh` now default to those settings.

Subgraph export:

- Export probe script:
  `/root/chatterbox/export_subgraph_probes.py`
- Export result JSON:
  `/root/chatterbox/exports/subgraph_probes/export_probe_results.json`
- The following random-weight subgraphs exported to ONNX and passed validation:
  voice encoder forward, F0 predictor, HiFiGAN Snake, HiFiGAN ResBlock, and
  S3Gen positionwise FFN.
- The voice encoder ONNX graph still contains `LSTM`, which may be a runtime
  support problem.
- The convolutional/vocoder pieces lower to more promising Conv/MatMul/Sin
  style graphs.

IREE Vulkan:

- Installed in the CPU venv:
  `iree-base-compiler==3.11.0`, `iree-base-runtime==3.11.0`,
  `iree-turbine==3.9.0`.
- Tiny Vulkan smoke probe:
  `/root/chatterbox/iree_vulkan_probe.py`
- Vulkan matrix:
  `/root/chatterbox/iree_vulkan_subgraph_matrix.py`
- Result summary:
  `/root/chatterbox/exports/iree_vulkan_matrix/iree_vulkan_matrix_2026-07-07.md`
- IREE Vulkan on RADV/GFX1013 successfully ran these subgraphs and matched
  PyTorch within tolerance:
  ELU, Conv1d+ELU, HiFiGAN Snake, S3Gen positionwise FFN, and a HiFiGAN
  ResBlock.
- F0 predictor compiles and runs on IREE Vulkan but does not match PyTorch yet;
  this needs numerical debugging before being used.
- F0 debug now shows:
  `/root/chatterbox/exports/iree_vulkan_f0_debug/f0_debug_2026-07-07.md`
  - F0 Conv+ELU prefixes match PyTorch on Vulkan.
  - The mismatch is in the final projection shape family, including standalone
    `Linear(512, 1)`, `Linear(512, 2)`, and a 1x1 Conv rewrite.
  - Keep F0 on CPU for now.
- Tiny HiFT full decode hits a Turbine import blocker at `torch.aten.stft`.
- Tiny HiFT no-FFT core succeeds on IREE Vulkan and matches PyTorch:
  `/root/chatterbox/exports/iree_vulkan_hift_core/hift_core_2026-07-07.md`
  - This covers transposed convolutions, source fusion, ResBlocks, exp, and sin.
  - Current practical split: keep STFT/ISTFT on CPU and target the HiFT
    convolutional core for Vulkan.
- Real Chatterbox Turbo `mel2wav` weights were loaded into a standalone
  `HiFTGenerator`, using only the `mel2wav.*` subset from
  `s3gen_meanflow.safetensors`.
- Real-weight HiFT no-FFT core results:
  `/root/chatterbox/exports/iree_vulkan_real_hift_core/real_hift_core_2026-07-07.md`
  - `mel2wav` parameter count: `20,806,557`.
  - `T=1` fails Vulkan compile because the first ConvTranspose has temporal
    input length 1.
  - `T=2`, `T=4`, and `T=8` compile and run on Vulkan.
  - Reconstructed CPU-ISTFT waveform mean absolute error stays around `1e-5`.
  - IREE Vulkan core speedup over PyTorch CPU core:
    `5.23x` at `T=2`, `4.88x` at `T=4`, and `4.41x` at `T=8`.
- Split HiFT Vulkan runtime prototype:
  `/root/chatterbox/split_hift_vulkan_runtime.py`
  - Loads existing fixed-shape real-core VMFBs in-process through
    `iree.runtime`.
  - CPU handles F0/source/STFT/ISTFT.
  - Vulkan handles the real HiFT convolutional core.
  - Validation summary:
    `/root/chatterbox/exports/split_hift_vulkan/split_hift_vulkan_2026-07-07.md`
  - End-to-end split decode speedup over CPU decode for supported fixed sizes is
    about `3.0x`.
  - RNG-controlled `inference()` comparison matches CPU source exactly and
    keeps waveform max error below `1e-4` for `T=2/4/8`.
  - The IREE Python runtime emits nanobind leak diagnostics at process exit in
    short-lived validation runs; investigate before enabling it in a long-lived
    API service.
- 2026-07-08 split runtime update:
  `/root/chatterbox/exports/split_hift_vulkan/split_hift_vulkan_2026-07-08.md`
  - The split runtime now auto-discovers compiled fixed shapes.
  - Supported real-core frame sizes are now:
    `{2, 4, 8, 16, 32, 64, 128}`.
  - Same-source split decode max error remains around `7e-5` to `1.2e-4`.
  - End-to-end split decode speedup ranges from about `2.7x` on tiny chunks to
    about `4.3x` on larger chunks.
- 2026-07-08 real-core compile update:
  `/root/chatterbox/exports/iree_vulkan_real_hift_core/real_hift_core_2026-07-08.md`
  - Static IREE Vulkan core compilation works through `T=128`.
  - Reconstructed waveform mean error stays around `1e-5`.
- 2026-07-08 chunked real-mel update:
  `/root/chatterbox/exports/hift_chunking/hift_chunking_chunk270_mid_overlap_2026-07-08.json`
  - Real Chatterbox mels can be decoded by overlapping fixed `T=128` Vulkan
    HiFT windows.
  - `center=96`, `window=128` was the best measured tradeoff:
    split decode `3.907s` versus full CPU HiFT `6.791s`, with max error
    `4.655e-5`.
  - This helps the vocoder stage but only saves a few seconds on a full
    270-character request because T3 remains dominant.
- 2026-07-08 opt-in API path:
  `/root/chatterbox/chatterbox_api.py`
  - Added an experimental, disabled-by-default API path for Vulkan HiFT:
    `CHATTERBOX_EXPERIMENTAL_VULKAN_HIFT=1`.
  - `CHATTERBOX_REQUIRE_VULKAN_HIFT=1` makes failures hard instead of falling
    back to CPU.
  - Defaults are `CHATTERBOX_HIFT_WINDOW_FRAMES=128` and
    `CHATTERBOX_HIFT_CENTER_FRAMES=96`.
  - Smoke metadata:
    `/root/chatterbox/exports/split_hift_vulkan/vulkan_hift_api_smoke_2026-07-08.json`
  - Smoke test succeeded with `REQUIRE` enabled: `31.3155s` wall time for a
    `3.08s` WAV. This proves API-level Vulkan HiFT invocation works, but T3 and
    S3 flow still dominate wall time.

T3/IREE Vulkan:

- T3 profiling showed the GPT2-style token generator is the main bottleneck:
  about `42.8s` of a `60.8s` 270-character CPU request.
- IREE Vulkan initially compiled T3 pieces but produced wrong values for
  several projection and reduction shapes.
- Correctness flags found:
  ```bash
  --iree-dispatch-creation-split-matmul-reduction=4
  --iree-dispatch-creation-enable-split-reduction
  ```
- These fix isolated GPT2/T3 projection, attention, MLP, layer norm, no-cache
  block, and cached single-token block probes.
- Safe fused cached stacks are currently correct up to 4 layers:
  - `gpt2_cached_stack_l4_p128_t1`: max error `1.392e-4`.
  - `gpt2_cached_stack_l4_p384_t1`: max error `1.507e-4`.
- Fused stacks of 5, 6, and 8 layers are not safe; they produce max errors
  around `15-20`.
- All six 4-layer cached chunks covering layers 0-23 now pass the current
  `allclose_1e-4` correctness check at past length 128:
  - 0-3 max error `1.392e-04`.
  - 4-7 max error `1.953e-03`, sparse enough to still pass relative allclose.
  - 8-11 max error `1.550e-05`.
  - 12-15 max error `1.621e-05`.
  - 16-19 max error `2.742e-05`.
  - 20-23 max error `1.564e-04`.
- Cache-output probe:
  `/root/chatterbox/export_t3_cached_stack_kv_probe.py`
- Cache-output result:
  `/root/chatterbox/exports/t3_exportability/t3_cached_stack_kv_s0_l4_p128_2026-07-08.json`
  `/root/chatterbox/exports/t3_exportability/t3_cached_stack_kv_s4_l4_p128_2026-07-08.json`
  - `gpt2_cached_stack_kv_s0_l4_p128_t1` returns hidden plus eight new K/V
    cache tensors for layers 0-3.
  - Hidden max error: `1.392e-04`.
  - Cache tensor max errors ranged from `4.172e-07` to `1.335e-05`.
  - `gpt2_cached_stack_kv_s4_l4_p128_t1` returns hidden plus eight new K/V
    cache tensors for layers 4-7.
  - Hidden max error: `1.953e-03`, still passing relative `allclose_1e-4`.
  - Cache tensor max errors ranged from `2.533e-07` to `5.245e-06`.
  - All outputs passed the current `allclose_1e-4` check.
- This makes a 6-stage chunked T3 Vulkan runtime plausible from a subgraph
  correctness standpoint. The cache-output probe removes the "can the GPU
  return usable autoregressive cache tensors?" blocker. It still needs an
  efficient in-process KV/cache runtime; shelling out or copying every chunk
  would erase the small per-chunk speedup.
- Current IREE Vulkan speedup for safe T3 cached stacks is modest:
  - 4-layer p128: Vulkan `12.7ms`, CPU `14.25ms`.
  - 4-layer p384: Vulkan `13.8ms`, CPU `15.84ms`.
  - 4-layer p128 layers 20-23: Vulkan `12.7ms`.
  - 4-layer p128 layers 0-3 with cache outputs: Vulkan `11.6ms` mean.
- In-process runtime probe:
  `/root/chatterbox/benchmark_t3_kv_iree_runtime.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/t3_kv_iree_runtime_s0_l4_p128_short_2026-07-08.json`
  - Preloaded Vulkan `DeviceArray` inputs: `14.186ms` without output fetch,
    `14.765ms` with hidden plus K/V outputs fetched.
  - Host NumPy inputs: roughly `21-24ms` per 4-layer chunk.
  - Device-resident repeated calls kept RSS flat in a 300-call loop.
  - Calls that materialize host outputs still trigger IREE nanobind leak
    diagnostics at process exit; short-loop RSS growth was small, but this
    must be retested before enabling a long-lived GPU API path.
  - Design implication: a real T3 Vulkan runtime needs in-process execution
    and cache tensors kept on device as much as possible. A naive host NumPy
    cache handoff per chunk likely erases most of the current GPU gain.
  - Current cache-output chunk artifact cost is about `385 MB` MLIR plus
    `193 MB` VMFB per 4-layer chunk. With the container filesystem at `96%`,
    pause additional cache-output chunk compiles until generated artifacts are
    cleaned or the filesystem is expanded.
- Two-chunk in-process runtime probe:
  `/root/chatterbox/benchmark_t3_two_chunk_iree_runtime.py`
  - Validation:
    `/root/chatterbox/exports/t3_exportability/t3_two_chunk_iree_runtime_s0_s4_short_2026-07-08.json`
  - Timing:
    `/root/chatterbox/exports/t3_exportability/t3_two_chunk_iree_runtime_s0_s4_timing_2026-07-08.json`
  - The hidden `DeviceArray` from layers 0-3 was passed directly into the
    layers 4-7 VMFB without materializing that hidden state on host.
  - Chained correctness passed:
    layers 0-3 max error `1.392e-04`; chained layers 4-7 max error
    `5.686e-05`; both passed `allclose_1e-4`.
  - Longer timing-only result: `24.606ms` for two chained chunks without output
    fetch, `28.352ms` when fetching all outputs from both chunks.
  - This removes the "can chunk VMFBs pass hidden state on device?" blocker.
    The remaining runtime problem is efficient K/V cache append/update across
    token steps.
- Fixed-window K/V cache update probe:
  `/root/chatterbox/probe_t3_kv_cache_update_vulkan.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/cache_update/t3_kv_cache_update_vulkan_2026-07-08.json`
  - VMFB:
    `/root/chatterbox/exports/t3_exportability/cache_update/t3_kv_cache_roll_append_p128_t1_vulkan_gfx1013.vmfb`
  - VMFB size: `10,567 bytes`; compile time: `0.430s`.
  - Operation is fixed-window roll/append:
    `[1,16,128,64] past + [1,16,1,64] new -> [1,16,128,64] next`.
  - Standalone Vulkan validation was exact.
  - Four on-device cache updates fed by Vulkan T3 outputs: `0.576ms`.
  - Single T3 chunk plus four cache updates stayed in the same rough timing
    budget as the T3 chunk alone; fetching updated cache raised it to
    `16.656ms`.
  - This is useful for bucketed/rolling-window experiments. It is not exact
    growing-cache autoregressive generation.
- Two chunks plus eight cache updates:
  - Timing:
    `/root/chatterbox/exports/t3_exportability/t3_two_chunk_with_cache_update_timing_2026-07-08.json`
  - Validation:
    `/root/chatterbox/exports/t3_exportability/t3_two_chunk_with_cache_update_validated_2026-07-08.json`
  - Cache update was exact against the actual Vulkan-produced K/V tensors for
    all eight layers covered by chunks 0-3 and 4-7.
  - Two chunks without fetch: `24.378ms`.
  - Two chunks plus eight on-device cache updates without fetch: `26.425ms`.
  - Two chunks plus eight updates fetching updated cache: `32.853ms`.
  - This removes the immediate host round-trip blocker for fixed-window K/V
    cache movement. The next runtime question is exact cache growth or a
    practical bucket schedule for real T3 generation.
- Fixed-slot K/V cache update probe:
  `/root/chatterbox/probe_t3_kv_cache_slot_update_vulkan.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/cache_update/t3_kv_cache_slot_update_vulkan_2026-07-08.json`
  - VMFB:
    `/root/chatterbox/exports/t3_exportability/cache_update/t3_kv_cache_slot_update_p128_t1_vulkan_gfx1013.vmfb`
  - VMFB size: `11,291 bytes`; compile time: `0.453s`.
  - Operation is mask-selected slot replacement:
    `[1,16,128,64] cache + [1,16,1,64] new + [1,1,128,1] one-hot slot mask`.
  - Standalone Vulkan validation was exact.
  - Runtime validation fed the updater actual Vulkan T3 K/V outputs for four
    layers; all updated caches matched exactly.
  - The same VMFB updated slots `0`, `1`, `64`, and `127` exactly by changing
    only the slot mask input.
  - Four on-device slot updates: `0.843ms`.
  - Single T3 chunk plus four slot updates without fetch: `15.126ms`.
  - This is a better primitive than rolling-window update for
    preallocated/growing-cache bucket experiments. The next T3 runtime blocker
    is the attention graph: it needs a valid-position mask so unused cache
    slots do not affect logits.
- Masked attention over preallocated cache:
  `/root/chatterbox/probe_t3_masked_cache_block_vulkan.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_cache_block0_p128_valid42_2026-07-08.json`
  - Runtime-mask sweep:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_cache_block0_runtime_masks_2026-07-08.json`
  - VMFB:
    `/root/chatterbox/exports/t3_exportability/masked_cache/gpt2_masked_cache_block0_p128_valid42_t1_vulkan_gfx1013.vmfb`
  - Static cache shape: `[1,16,128,64]` key/value plus additive attention bias
    `[1,1,1,129]`.
  - PyTorch static masked block matched PyTorch dynamic-cache block for valid
    length `42`: hidden max error `1.907e-06`; new K/V exact.
  - IREE Vulkan matched PyTorch static masked block:
    hidden max error `9.537e-06`, new key `8.941e-07`, new value `4.172e-07`.
  - The same compiled VMFB passed valid lengths `0`, `1`, `42`, `64`, `127`,
    and `128` by changing only the runtime attention-bias tensor.
  - Steady IREE benchmark: `4.60ms` mean, `4.54ms` median.
  - Artifact cost: `100,787,043` byte MLIR and `50,533,112` byte VMFB.
  - This removes the one-block attention-mask blocker for preallocated cache
    buckets. Next validation target is a masked-cache 4-layer chunk, but disk
    cleanup/expansion should happen first.
- Masked 4-layer cache-output chunk:
  `/root/chatterbox/probe_t3_masked_cache_stack_kv_vulkan.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_cache_stack_s0_l4_p128_valid42_2026-07-08.json`
  - Runtime-mask sweep:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_cache_stack_s0_l4_runtime_masks_2026-07-08.json`
  - VMFB:
    `/root/chatterbox/exports/t3_exportability/masked_cache/gpt2_masked_cache_stack_s0_l4_p128_valid42_t1_vulkan_gfx1013.vmfb`
  - Static cache shape: four `[1,16,128,64]` key/value pairs plus shared
    attention bias `[1,1,1,129]`.
  - PyTorch static masked chunk matched PyTorch dynamic-cache chunk at valid
    length `42`: hidden max error `1.717e-05`; K/V max errors up to
    `8.583e-06`.
  - IREE Vulkan matched PyTorch static masked chunk: hidden max error
    `8.202e-05`; K/V max errors up to `1.144e-05`.
  - The same compiled VMFB passed runtime valid lengths `0`, `1`, `42`, `64`,
    `127`, and `128` by changing only the attention-bias tensor.
  - Valid length `0` had a sparse hidden max outlier `6.348e-03`, but mean
    hidden error was `9.573e-06`, p95 hidden error was `4.530e-06`, and the
    full output passed both `allclose_1e-4` and `allclose_1e-3`.
  - Steady IREE benchmark: `11.6ms`, essentially the same as the earlier
    cache-output 4-layer chunk.
  - Artifact cost: `403,148,734` byte MLIR and `201,702,428` byte VMFB.
  - This proves the masked preallocated-cache pattern works at the safe
    4-layer T3 chunk granularity. The next T3 work is runtime assembly:
    multiple masked chunks, fixed-slot cache updates, embeddings, speech head,
    and sampling in one measured token loop.
- Single-chunk masked-cache loop:
  `/root/chatterbox/benchmark_t3_masked_chunk_cache_loop.py`
  - 8-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_chunk_cache_loop_8step_2026-07-08.json`
  - 32-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_chunk_cache_loop_32step_2026-07-08.json`
  - Uses the existing masked 4-layer chunk VMFB and fixed-slot cache updater
    VMFB; no new transformer compile.
  - Starts with empty preallocated K/V caches, runs synthetic hidden inputs for
    multiple token steps, updates each layer's cache slot on device, and
    compares against a PyTorch dynamic-cache reference.
  - 8-step loop: all steps passed `allclose_1e-4`; no-fetch timing
    `17.392ms/step`.
  - 32-step loop: all steps passed `allclose_1e-4`; no-fetch timing
    `15.879ms/step`.
  - Sparse hidden outliers remain: max `7.324e-03` for 8 steps and
    `1.917e-02` for 32 steps. Worst-step mean and p95 errors stayed small
    (`2.459e-05` mean and `6.914e-06` p95 at the 32-step worst outlier).
  - This proves one chunk can fill and reuse preallocated cache over multiple
    token steps without host cache round-trips. Full T3 runtime still needs
    generated-token drift checks, speech head/sampling, and all six chunks.
- Single-chunk masked-cache loop plus speech head:
  `/root/chatterbox/benchmark_t3_masked_chunk_logits_loop.py`
  - 8-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_chunk_logits_loop_8step_2026-07-08.json`
  - 32-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_masked_chunk_logits_loop_32step_2026-07-08.json`
  - Uses existing VMFBs only: masked 4-layer chunk, fixed-slot cache updater,
    and speech head.
  - 8-step loop: logits passed `allclose_1e-4`, argmax matched `8/8`, top-5
    overlap was `5/5`, no-fetch timing `18.895ms/step`.
  - 32-step loop: logits passed `allclose_1e-3` but not strict `1e-4`, argmax
    matched `32/32`, top-5 overlap was `5/5`, no-fetch timing
    `16.481ms/step`.
  - Worst 32-step logits error: `4.242e-03`; worst-step logits mean error:
    `7.888e-04`; p95: `1.923e-03`.
  - This is the first assembled sub-runtime that includes cache reuse and
    speech logits on Vulkan without host cache round-trips. It still covers
    only one 4-layer T3 chunk, not the full 24-layer model or sampler.
- Full 24-layer masked-cache T3 loop:
  `/root/chatterbox/benchmark_t3_full_masked_vulkan_loop.py`
  - 4-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_4step_2026-07-08.json`
  - 8-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_8step_2026-07-08.json`
  - 32-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_32step_2026-07-08.json`
  - 128-step result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_128step_2026-07-08.json`
  - Uses all six masked 4-layer VMFBs, the fixed-slot cache updater, and the
    speech head. Hidden state stays device-resident between chunk VMFBs, and
    all 24 layer K/V caches are updated through the Vulkan slot updater.
  - 8-step loop: hidden/logits passed `allclose_1e-4`, argmax matched `8/8`,
    and no-fetch timing was `85.474ms/step`.
  - 32-step loop: logits passed `allclose_1e-3`, argmax matched `32/32`,
    cache passed `allclose_1e-4`, and no-fetch timing was `82.752ms/step`.
  - 128-step loop: logits passed `allclose_1e-3`, argmax matched `128/128`,
    cache max error was `1.808e-04`, and no-fetch timing was
    `81.924ms/step`.
  - This proves the full T3 transformer body can run on the BC-250 Vulkan path
    in an assembled in-process loop. It is still synthetic-hidden validation,
    not production generation: embeddings, CFG/sampling, real prompt
    conditioning, and generated-token drift need validation before API wiring.
- Real-prompt cache length and p1024 bucket:
  `/root/chatterbox/probe_t3_real_prompt_length_budget.py`
  - Result:
    `/root/chatterbox/exports/t3_exportability/t3_real_prompt_length_budget_2026-07-08.json`
  - Turbo default conditioning is `376` tokens, so even a hello prompt starts
    with a `385` token context.
  - The 270-character benchmark starts with a `423` token context.
  - The 270-character benchmark plus about 512 generated tokens needs roughly
    `935` cache slots, so the p128 Vulkan loop cannot cover real requests.
  - p1024 single-block and six 4-layer chunks compiled and passed Vulkan
    validation at valid length `935`.
  - p1024 fixed-slot cache updater compiled, validated exactly, and ran
    24 updates in `10.507ms`.
  - p1024 full 24-layer loop:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_p1024_32step_2026-07-08.json`
  - p1024 32-step result: logits passed `allclose_1e-3`, argmax matched
    `32/32`, cache passed `allclose_1e-4`, and no-fetch timing was
    `110.309ms/step`.
  - Final `ln_f + speech_head` Vulkan artifact:
    `/root/chatterbox/exports/t3_exportability/t3_final_norm_speech_head_vulkan_2026-07-08.json`
    - max error `5.722e-06`.
    - no-fetch timing `1.119ms`.
  - Corrected p1024 full-loop result with final head:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_full_masked_vulkan_loop_p1024_finalhead_8step_2026-07-08.json`
    - logits passed `allclose_1e-4`.
    - argmax matched `8/8`.
    - no-fetch timing `125.421ms/step`.
  - Real CPU-prefill to Vulkan follow-on bridge:
    `/root/chatterbox/benchmark_t3_real_prefill_vulkan_followon.py`
  - 270-character bridge result:
    `/root/chatterbox/exports/t3_exportability/masked_cache/t3_real_prefill_vulkan_followon_chunk270_finalhead_32step_fetchlogits_2026-07-08.json`
    - CPU computes the real prompt prefill, then Vulkan handles follow-on
      single-token T3 steps using real speech-token embeddings and copied p1024
      K/V cache.
    - logits passed `allclose_1e-4`.
    - argmax matched `32/32`.
    - cache passed `allclose_1e-4`.
    - CPU follow-on timing `106.251ms/step`.
    - Vulkan follow-on timing including logits fetch `183.461ms/step`.
  - This proves the real-prompt bridge is correct, but the current IREE/Vulkan
    bridge is slower than CPU for practical follow-on token steps.
- Measured full-runtime projection:
  `/root/chatterbox/project_t3_vulkan_runtime_scorecard.py`
  - JSON:
    `/root/chatterbox/exports/benchmarks/t3_vulkan_runtime_projection_2026-07-08.json`
  - Markdown:
    `/root/chatterbox/exports/benchmarks/t3_vulkan_runtime_projection_2026-07-08.md`
  - Inputs include CPU full API `62.809s`, RTX 5090 full API `3.495s`, CPU T3
    stage `47.089s`, CPU S3 flow `10.822s`, CPU HiFT `7.089s`, and Vulkan
    HiFT `4.249s` for the 270-character case.
  - Measured Vulkan T3 pieces: masked 4-layer chunk `11.580ms`, four cache
    slot updates `0.843ms`, speech head `0.808ms`, one-chunk Python loop with
    cache and speech head `16.481ms/step`, actual full 24-layer loop
    `81.924ms/step` at 128 steps, and p1024 real-bucket loop
    `110.309ms/step` at 32 steps.
  - Optimistic projected full 24-layer T3 token cost: `75.344ms`.
  - Python-loop projected full 24-layer T3 token cost: `98.887ms`.
  - Actual measured full-loop T3 token cost: `81.924ms`.
  - Optimistic projected full API with Vulkan T3 and Vulkan HiFT: `54.376s`,
    or `15.56x` slower than RTX 5090.
  - Python-loop projected full API with Vulkan T3 and Vulkan HiFT: `67.342s`,
    or `19.27x` slower than RTX 5090.
  - Actual measured-loop full API projection with Vulkan T3 and Vulkan HiFT:
    `58.000s`, or `16.60x` slower than RTX 5090.
  - Actual p1024 real-bucket full API projection with Vulkan T3 and Vulkan
    HiFT: `73.632s`, or `21.07x` slower than RTX 5090.
  - Corrected p1024 final-head full API projection: `81.956s`, or `23.45x`
    slower than RTX 5090.
  - Real-prefill Vulkan bridge full API projection: `121.172s`, or `34.67x`
    slower than RTX 5090.
  - Readout: the IREE Vulkan T3 loop is real and gives modest local speedup,
    but the p128 loop is too small for real prompts and the corrected p1024
    real-prefill bridge is slower than CPU. Current IREE numbers do not support
    the target of only about `2x` slower than the RTX 5090 API.
- Detailed T3 findings:
  `/root/chatterbox/exports/t3_exportability/t3_vulkan_gpu_pipeline_findings_2026-07-08.md`

Artifact cleanup:

- Helper:
  `/root/chatterbox/manage_vulkan_artifacts.py`
- Conservative dry-run report:
  `/root/chatterbox/exports/t3_exportability/artifact_cleanup_conservative_2026-07-08.json`
  - Would recover about `3.7 GB`.
  - Main candidates are failed fused 5/6/8-layer T3 blobs and superseded
    compiler-flag VMFBs. JSON summaries and docs are preserved.
- Aggressive dry-run report:
  `/root/chatterbox/exports/t3_exportability/artifact_cleanup_aggressive_2026-07-08.json`
  - Would recover about `8.0 GB`.
  - Also removes reproducible generated T3 MLIR.
- Updated dry-run reports:
  - `/root/chatterbox/exports/t3_exportability/artifact_cleanup_conservative_2026-07-08-v2.json`
  - `/root/chatterbox/exports/t3_exportability/artifact_cleanup_aggressive_2026-07-08-v2.json`
  - Conservative remains about `3.7 GB`.
  - Aggressive is now about `8.5 GB` because it includes generated
    masked-cache MLIR while preserving VMFBs and JSON reports.
- The helper is dry-run by default. Actual deletion requires both
  `--delete` and `--yes`.

Other runtime availability in this container:

- Installed/available: `iree`, `onnx`, `vulkaninfo`.
- Not currently installed or on PATH: `ncnn`, `pnnx`, `executorch`, `tvm`,
  `mlc_llm`, `onnxruntime`, `llama-cli`, `llama-server`.
- The unused `.venv-rocm` directory was removed after the ROCm/HIP crash path
  was ruled out, freeing about `18 GB`. Current root filesystem usage after the
  full T3 Vulkan artifacts is about `69%`; still avoid large alternate runtime
  installs unless they are the next focused experiment.

S3 flow/IREE Vulkan:

- Stage benchmarking after Vulkan HiFT shows S3 flow is the second-largest
  remaining CPU stage behind T3:
  - Short case: S3 flow `4.278s`.
  - 270-char case: S3 flow `10.822s`.
- Component probe script:
  `/root/chatterbox/probe_s3_flow_vulkan_components.py`
- Findings:
  `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_vulkan_component_findings_2026-07-08.md`
- Real-weight S3 flow conv/projection components export and run correctly on
  IREE Vulkan:
  - `s3_flow_downsample_t32`: max error `6.199e-06`.
  - `s3_flow_final_proj_t32`: max error `4.768e-06`.
- Real-weight S3 flow ResNet/final blocks export but are not currently usable
  on IREE Vulkan:
  - `T=32` fails Vulkan compile in `layer_norm` lowering with a SPIR-V vector
    constant legalization error.
  - `T=16` final block compiles but is numerically wrong.
- Real-weight S3 flow transformer blocks export, and IREE CPU matches PyTorch,
  but the early small-shape IREE Vulkan probes were wrong:
  - `s3_flow_mid_transformer_t16`: IREE CPU max error `1.431e-06`; IREE Vulkan
    max error `5.709e-01`.
  - Forcing Diffusers eager/bmm attention avoids the SDPA compile failure but
    does not make the small-shape Vulkan output correct.
- Isolated attention and FFN submodules are correct at early estimated S3
  shapes:
  - `s3_flow_mid_attention_t355`: max error `6.557e-06`, mean Vulkan runtime
    `2.843ms`.
  - `s3_flow_mid_ff_t355`: max error `5.245e-06`, mean Vulkan runtime
    `0.404ms`.
  - `s3_flow_down_attention_t710`: max error `9.537e-07`, mean Vulkan runtime
    `6.830ms`.
  - `s3_flow_down_ff_t710`: max error `1.097e-05`, mean Vulkan runtime
    `0.606ms`.
- Later shape capture showed the actual chunk270 estimator transformer shape is
  `[1, 1210, 256]` for down, mid, and up blocks, with attention bias shape
  `[1, 1, 1210]`:
  - Shape artifact:
    `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_actual_transformer_shapes_chunk270_synthetic_2026-07-08.json`
- Actual-shape full transformer blocks are correct on IREE Vulkan:
  - Correctness:
    `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_components_actualshape_transformer_t1210_2026-07-08.json`
  - Benchmark scorecard:
    `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_actualshape_transformer_benchmark_2026-07-08.md`
  - `s3_flow_down_transformer_bias_t1210`: max error `3.338e-06`, mean
    runtime `24.306ms`.
  - `s3_flow_mid_transformer_full_t1210`: max error `1.907e-06`, mean runtime
    `17.431ms`.
  - `s3_flow_up_transformer_t1210`: max error `6.199e-06`, mean runtime
    `15.856ms`.
- The 270-character CPU S3 profile spends about `7.443s` inside down/mid/up
  transformer blocks. Using measured actual-shape Vulkan full-block timings,
  that portion projects to about `1.995s` of device-resident compute, or about
  `3.73x` faster than the CPU transformer portion.
- Actual-shape surrounding estimator components are also correct on IREE
  Vulkan:
  - Benchmark scorecard:
    `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_actualshape_estimator_components_benchmark_2026-07-08.md`
  - ResNet/downsample/upsample/final CPU profile portion: `0.528s`.
  - Projected Vulkan time for that portion: `0.242s`.
  - Combined covered estimator CPU profile portion: `7.972s`.
  - Combined covered estimator Vulkan projection: `2.237s`, or `3.56x`
    faster than CPU for the covered estimator components.
- Current measured best 270-character warm path remains `22.583s` for
  ggml/Vulkan T3 plus IREE/Vulkan HiFT, about `6.46x` slower than the saved RTX
  5090 API baseline. If the covered actual-shape S3 estimator-component
  projection survives integration, the warm path projects to about `16.848s`,
  or about `4.82x` slower than the saved RTX 5090 baseline.
- S3 flow encoder components are also viable on IREE Vulkan:
  - Benchmark scorecard:
    `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_encoder_components_benchmark_2026-07-08.md`
  - Encoder input shape: `[1, 605, 512]`.
  - Encoder output shape: `[1, 1210, 512]`.
  - Lower Conformer layer T605 passes with max error `4.530e-06`, mean runtime
    `20.809ms`.
  - Upper Conformer layer T1210 passes with max error `1.383e-05`, mean
    runtime `70.642ms`.
  - Direct bool-mask variants hit IREE `.npy` bool input parsing failures;
    float-mask variants that convert `mask > 0.5` inside the graph pass.
  - All six lower Conformer layers and all four upper Conformer layers pass
    with distinct weights.
  - Mixed encoder projection: CPU `2.569s` to projected `0.509s`, or `5.05x`
    faster for the encoder component profile.
  - Combining covered estimator components plus the mixed encoder projection
    gives a warm-path projection of about `14.789s`, or about `4.23x` slower
    than the saved RTX 5090 baseline.
- Conclusion: keep the live API on CPU S3 flow for now, but the fixed-shape S3
  transformer path has reopened. The next S3 Vulkan work is integration and
  device-residency testing for the actual chunk270 bucket, plus separate
  ResNet/final-block normalization work.

This proves the BC-250 Vulkan path is real for small Chatterbox-derived
subgraphs. It does **not** make the whole upstream PyTorch Chatterbox pipeline
run on Vulkan yet.

## Next experiment

Do not run more ROCm.

The first CPU-only export probe has completed:

- Script: `/root/chatterbox/export_voice_encoder_onnx.py`
- Output: `/root/chatterbox/exports/voice_encoder.forward.random_weights.onnx`
- Input shape: `(1, 160, 40)`
- Output shape: `(1, 256)`
- ONNX validation: passed
- Exported op types include `LSTM`, `Gemm`, `Relu`, `ReduceL2`, and tensor shape
  ops

This is useful but not sufficient. The voice encoder graph contains ONNX `LSTM`,
which many Vulkan inference runtimes do not accelerate directly.

The IREE path proved correctness but not enough speed. The first speed-positive
result is the llama.cpp/ggml Vulkan pivot:

- Build: `/root/llama.cpp/build-vulkan`
- Benchmark source: `/root/chatterbox/bench_ggml_t3_vulkan.cpp`
- Benchmark binary: `/root/chatterbox/bench_ggml_t3_vulkan`
- Results:
  `/root/chatterbox/exports/benchmarks/ggml_t3_vulkan_primitives_2026-07-08.md`

Key speedups from that run:

- `qkv_1024_to_3072`, f32: `5.66x` faster than ggml CPU.
- `mlp_up_1024_to_4096`, f32: `6.32x` faster than ggml CPU.
- `mlp_down_4096_to_1024`, f32: `5.84x` faster than ggml CPU.
- `attention_ctx935_16h64`, f32: `2.94x` faster than ggml CPU.
- `dense_layer_no_kv`, f32: `8.46x` faster than ggml CPU.

This is not an API speedup yet, but it is the strongest evidence so far that a
real GPU T3 pipeline can be faster than CPU on this BC-250. The next safe
experiment is therefore a ggml-backed T3 prototype, not another ROCm attempt:

1. Export or map Chatterbox T3 GPT-2 weights into ggml-compatible tensors.
   First block fixture is done:
   `/root/chatterbox/exports/ggml_t3_block0_fixture/manifest.json`.
2. Build a one-layer cached GPT-2 block with real weights and compare against
   PyTorch for a single token. This is done for block 0:
   `/root/chatterbox/exports/benchmarks/ggml_t3_block0_correctness_2026-07-08.md`.
   Result: ggml Vulkan `0.346675 ms`, ggml CPU `2.02373 ms`, `5.84x`
   speedup, Vulkan max abs error `1.33514e-05`.
3. Expand the same probe to longer cache length, especially the realistic
   `p1024/valid935` bucket. This is done for block 0:
   `/root/chatterbox/exports/benchmarks/ggml_t3_block0_correctness_p935_2026-07-08.md`.
   Result: ggml Vulkan `0.609847 ms`, ggml CPU `3.6207 ms`, `5.94x`
   speedup, Vulkan max abs error `3.52859e-05`.
4. Export/map all 24 layers with a device-resident KV cache and compare
   generated token logits/top-k against PyTorch. First full-stack offline
   fixture is done:
   `/root/chatterbox/exports/benchmarks/ggml_t3_full_stack_correctness_p935_2026-07-08.md`.
   Result: ggml Vulkan `12.9533 ms`, ggml CPU `90.0656 ms`, `6.95x`
   speedup, Vulkan logits max abs error `2.92063e-06`, argmax match
   `2031/2031`, top-10 overlap `10/10`.
5. Only after correctness holds, connect it as an opt-in API path.

The next T3 Vulkan work is no longer basic feasibility. It is runtime
engineering:

1. Convert the full-stack fixture into a multi-step loop. This is done for
   real prompt trajectories:
   `/root/chatterbox/exports/benchmarks/ggml_t3_real_prompt_multistep_chunk270_s4_p935_2026-07-08.md`
   and
   `/root/chatterbox/exports/benchmarks/ggml_t3_real_prompt_multistep_hello_s4_p935_2026-07-08.md`.
2. Feed real speech-token embeddings and compare logits/top-k for multiple
   consecutive steps. This is done for four steps in both fixtures. Chunk270
   result: Vulkan `16.1886 ms/step`, CPU `101.09 ms/step`, `4/4` argmax
   matches, `10/10` top-10 overlap for every step.
3. Keep all 24 layers' KV cache device-resident.
   This is done in the offline C++ resident-cache probe:
   `/root/chatterbox/exports/benchmarks/ggml_t3_resident_cache_chunk270_s4_p935_2026-07-08.md`.
4. Update one cache slot per generated token on device. This is done with
   `ggml_set_rows`; focused cache update result:
   `/root/chatterbox/exports/benchmarks/ggml_cache_set_rows_p935_2026-07-08.md`.
   Full T3 resident-cache result: Vulkan `15.5914 ms/step`, steady-state about
   `14.00 ms/step`, `4/4` argmax matches, `10/10` top-10 overlap.
5. Wire the ggml T3 path into `chatterbox_api.py` behind an explicit
   experimental flag. The first bridge is now implemented:
   - C ABI: `/root/chatterbox/t3_ggml_vulkan_bridge.cpp`
   - Shared library: `/root/chatterbox/libt3_ggml_vulkan_bridge.so`
   - Python wrapper: `/root/chatterbox/t3_ggml_vulkan_runtime.py`
   - API flag: `CHATTERBOX_EXPERIMENTAL_VULKAN_T3=1`
   - Hard-fail flag: `CHATTERBOX_REQUIRE_VULKAN_T3=1`
   - ctypes validation:
     `/root/chatterbox/exports/benchmarks/ggml_t3_ctypes_bridge_chunk270_s4_p935_2026-07-08.md`

   Result: the Python-facing bridge validates against the same real-prompt
   chunk270 fixture with `4/4` correct steps, `10/10` top-10 overlap on every
   step, and `15.6982 ms/step` mean runtime. The default API remains CPU-only;
   this path is not enabled unless the explicit flag is set.

6. Run an API-shape opt-in WAV smoke test. This is now done in an isolated
   process, not the live systemd service:
   `/root/chatterbox/benchmark_t3_ggml_vulkan_api_path.py`.

   Bounded deterministic smoke:
   `/root/chatterbox/exports/benchmarks/ggml_t3_api_path_smoke_32tok_2026-07-08.md`.
   Result: CPU T3 `12.881s`, Vulkan T3 `3.517s`, `3.66x` speedup, tokens
   matched exactly with `top_k=1`.

   Original 270-character benchmark shape with API-default sampling:
   `/root/chatterbox/exports/benchmarks/ggml_t3_api_path_chunk270_default_2026-07-08.md`.
   Result: Vulkan T3 `7.422s`, S3 + watermark `17.858s`, warm pipeline
   estimate `25.280s`, audio `14.200s`, generated speech tokens `352`.

   Scorecard:
   `/root/chatterbox/exports/benchmarks/performance_vulkan_t3_scorecard_2026-07-08.md`.
   Against the saved local CPU API baseline for chunk270 (`62.809s`), this is
   `2.48x` faster by warm request wall time and about `2.37x` faster by
   generated-audio throughput. It is still about `7.23x` slower than the RTX
   5090 by wall time, so the 2x-slower-than-5090 target is not met yet.

7. Combine the ggml Vulkan T3 bridge with the existing split IREE/Vulkan HiFT
   path. This is now done in an isolated process:
   `/root/chatterbox/exports/benchmarks/ggml_t3_hift_vulkan_api_path_chunk270_default_r2_2026-07-08.md`.

   Result for the same 270-character case:
   - Vulkan T3: `7.384s`
   - S3 flow: `10.504s`
   - Source/F0: `0.130s`
   - Vulkan HiFT decode: `3.984s`
   - Watermark: `0.578s`
   - Warm pipeline estimate excluding load: `22.583s`
   - Audio: `14.200s`

   Against the local CPU API baseline, this is `2.78x` faster by warm wall time
   and about `2.65x` faster by generated-audio throughput. It is still about
   `6.46x` slower than the RTX 5090 by wall time. The IREE runtime emitted
   nanobind leak diagnostics at shutdown; treat split HiFT as a service-risk
   until persistent-process behavior is retested.

8. Profile and isolate the remaining S3 flow bottleneck. This is now done:
   `/root/chatterbox/exports/benchmarks/s3_flow_chunk270_profile_transformers_2026-07-08.md`
   and
   `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_vulkan_component_findings_2026-07-08.md`.

   Real chunk270 S3 flow timing:
   - S3 flow total: `10.228s`
   - Flow encoder: `2.196s`
   - Decoder wrapper: `8.020s`
   - Transformer self-attention across calls: about `5.734s`
   - Transformer FFN across calls: about `1.634s`

   New isolated Vulkan probes show:
   - S3 attention alone is correct on IREE Vulkan.
   - S3 FFN alone is correct on IREE Vulkan.
   - S3 LayerNorm alone is wrong on IREE Vulkan for `[1, 16, 256]`:
     max abs error about `0.623`.
   - The same LayerNorm MLIR compiled to IREE CPU matches PyTorch with max abs
     error about `7.15e-07`.
   - Manual mean/variance LayerNorm is also wrong on IREE Vulkan.

   Interpretation: S3 flow is blocked by Vulkan correctness for
   LayerNorm/reduction-style normalization, not by raw attention or FFN kernels.

   Real-shape attention/FFN probes at the actual chunk270 sequence lengths are
   also complete:
   `/root/chatterbox/exports/s3_flow_vulkan_components/s3_flow_realshape_attention_ff_benchmark_2026-07-08.md`.

   Results:
   - `s3_flow_mid_attention_t355`: correct, `2.843 ms`
   - `s3_flow_mid_ff_t355`: correct, `0.404 ms`
   - `s3_flow_down_attention_t710`: correct, `6.830 ms`
   - `s3_flow_down_ff_t710`: correct, `0.606 ms`
   - CPU profile spends about `5.734s` in S3 transformer attention and `1.634s`
     in S3 transformer FFN.
   - Optimistic device-resident Vulkan attention+FFN projection is about
     `0.397s`, excluding LayerNorm, transfer, and Python overhead.

   This means the S3 flow upside is real if normalization can be fixed, but a
   CPU/GPU round-trip around every LayerNorm is unlikely to preserve the full
   gain.

9. Next bridge work: move beyond isolated-process testing. Either start a
   separate experimental service with `CHATTERBOX_EXPERIMENTAL_VULKAN_T3=1`
   and `CHATTERBOX_T3_MAX_GEN_LEN` set conservatively, or continue reducing the
   remaining S3 flow/vocoder CPU time before exposing the path as a network
   endpoint.

Experimental service scaffolding is ready but intentionally not enabled:

- Launcher: `/root/chatterbox/run_api_vulkan_t3.sh`
- Unit: `/etc/systemd/system/chatterbox-api-vulkan-t3.service`
- Port: `8002`
- State after setup: `disabled` and `inactive`
- Safe CPU API state after setup: `enabled` and `active` on port `8000`

Start this only deliberately for network testing:

```bash
systemctl start chatterbox-api-vulkan-t3.service
curl -s http://127.0.0.1:8002/health
```

An ONNX/IREE S3 or voice encoder experiment can still be useful later, but it
is not the main speed path now. If we want to keep pushing non-T3 acceleration,
the next safe experiment is another CPU-only export/compile feasibility test:

1. Isolate a small, fixed-shape module, such as a HiFiGAN
   block.
2. Try exporting that module to ONNX or torch-export.
3. Try compiling that exported graph to a Vulkan-capable runtime.
4. Only run the compiled Vulkan result after the export path is proven and the
   test is small.

This will tell us whether a partial Vulkan port is technically tractable without
touching ROCm/HIP again.

## Current service state

The safe CPU Turbo API is running:

- `chatterbox-api.service`: enabled and active
- Health endpoint: `http://127.0.0.1:8000/health`
- Test output: `/root/chatterbox/exports/cpu_turbo_test.wav`

The unsafe ROCm service is disabled and inactive:

- `chatterbox-api-rocm.service`: disabled and inactive

Dangerous ROCm scripts require explicit override:

```bash
ALLOW_UNSAFE_BC250_ROCM=1 /root/chatterbox/verify_native_hip_gfx1013.sh
ALLOW_UNSAFE_BC250_ROCM=1 /root/chatterbox/verify_rocm_torch.sh
ALLOW_UNSAFE_BC250_ROCM=1 /root/chatterbox/run_api_rocm.sh
```
