#!/usr/bin/env python3
"""Profile S3 flow submodules on the real chunk270 API-shape path."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "benchmarks"


class TimingCollector:
    def __init__(self, modules: dict[str, torch.nn.Module]) -> None:
        self.modules = modules
        self.handles: list[Any] = []
        self.stack: list[dict[str, Any]] = []
        self.stats = defaultdict(lambda: {"calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0})

    def __enter__(self) -> "TimingCollector":
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _pre_hook(self, name: str):
        def hook(_module, _inputs) -> None:
            self.stack.append({"name": name, "start": time.perf_counter(), "child_s": 0.0})

        return hook

    def _post_hook(self, name: str):
        def hook(_module, _inputs, _output) -> None:
            ended = time.perf_counter()
            frame = self.stack.pop()
            if frame["name"] != name:
                raise RuntimeError(f"timing hook stack mismatch: expected {frame['name']} got {name}")
            inclusive = ended - frame["start"]
            exclusive = max(0.0, inclusive - frame["child_s"])
            stat = self.stats[name]
            stat["calls"] += 1
            stat["inclusive_s"] += inclusive
            stat["exclusive_s"] += exclusive
            if self.stack:
                self.stack[-1]["child_s"] += inclusive

        return hook

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for name, stat in self.stats.items():
            calls = int(stat["calls"])
            rows.append(
                {
                    "name": name,
                    "calls": calls,
                    "inclusive_seconds": stat["inclusive_s"],
                    "exclusive_seconds": stat["exclusive_s"],
                    "mean_inclusive_ms": stat["inclusive_s"] * 1000.0 / calls if calls else 0.0,
                    "mean_exclusive_ms": stat["exclusive_s"] * 1000.0 / calls if calls else 0.0,
                }
            )
        return sorted(rows, key=lambda row: row["inclusive_seconds"], reverse=True)


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def selected_modules(model: ChatterboxTurboTTS) -> dict[str, torch.nn.Module]:
    flow = model.s3gen.flow
    decoder = flow.decoder
    estimator = decoder.estimator
    modules: dict[str, torch.nn.Module] = {
        "flow.input_embedding": flow.input_embedding,
        "flow.encoder": flow.encoder,
        "flow.encoder_proj": flow.encoder_proj,
        "flow.spk_embed_affine_layer": flow.spk_embed_affine_layer,
        "decoder": decoder,
        "decoder.estimator": estimator,
        "estimator.time_embeddings": estimator.time_embeddings,
        "estimator.time_mlp": estimator.time_mlp,
        "estimator.time_embed_mixer": estimator.time_embed_mixer,
        "estimator.final_block": estimator.final_block,
        "estimator.final_proj": estimator.final_proj,
    }
    for index, block in enumerate(estimator.down_blocks):
        modules[f"estimator.down.{index}.resnet"] = block[0]
        modules[f"estimator.down.{index}.transformers"] = block[1]
        for block_index, transformer in enumerate(block[1]):
            prefix = f"estimator.down.{index}.transformer.{block_index}"
            modules[prefix] = transformer
            modules[f"{prefix}.norm1"] = transformer.norm1
            modules[f"{prefix}.attn1"] = transformer.attn1
            modules[f"{prefix}.norm3"] = transformer.norm3
            modules[f"{prefix}.ff"] = transformer.ff
        modules[f"estimator.down.{index}.downsample"] = block[2]
    for index, block in enumerate(estimator.mid_blocks):
        modules[f"estimator.mid.{index}.resnet"] = block[0]
        modules[f"estimator.mid.{index}.transformers"] = block[1]
        for block_index, transformer in enumerate(block[1]):
            prefix = f"estimator.mid.{index}.transformer.{block_index}"
            modules[prefix] = transformer
            modules[f"{prefix}.norm1"] = transformer.norm1
            modules[f"{prefix}.attn1"] = transformer.attn1
            modules[f"{prefix}.norm3"] = transformer.norm3
            modules[f"{prefix}.ff"] = transformer.ff
    for index, block in enumerate(estimator.up_blocks):
        modules[f"estimator.up.{index}.resnet"] = block[0]
        modules[f"estimator.up.{index}.transformers"] = block[1]
        for block_index, transformer in enumerate(block[1]):
            prefix = f"estimator.up.{index}.transformer.{block_index}"
            modules[prefix] = transformer
            modules[f"{prefix}.norm1"] = transformer.norm1
            modules[f"{prefix}.attn1"] = transformer.attn1
            modules[f"{prefix}.norm3"] = transformer.norm3
            modules[f"{prefix}.ff"] = transformer.ff
        modules[f"estimator.up.{index}.upsample"] = block[2]
    return {name: module for name, module in modules.items() if module is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default="Hello world, this is a short S3 flow profile.")
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--out-prefix", default="s3_flow_chunk270_profile_2026-07-08")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop_threads)
    torch.manual_seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.out_prefix}.json"
    md_path = OUT_DIR / f"{args.out_prefix}.md"

    load_start = time.perf_counter()
    model = ChatterboxTurboTTS.from_pretrained("cpu")
    load_seconds = time.perf_counter() - load_start

    selected_text = select_text(args.case, args.text)
    normalized = punc_norm(selected_text)
    text_tokens = model.tokenizer(normalized, return_tensors="pt", padding=True, truncation=True)
    text_tokens = text_tokens.input_ids.to(model.device)

    runtime_start = time.perf_counter()
    runtime = T3GGMLVulkanRuntime()
    runtime_load_seconds = time.perf_counter() - runtime_start

    t3_start = time.perf_counter()
    speech_tokens = inference_turbo_vulkan(
        model.t3,
        runtime,
        t3_cond=model.conds.t3,
        text_tokens=text_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_gen_len=args.max_gen_len,
    )
    t3_seconds = time.perf_counter() - t3_start

    speech_tokens = speech_tokens[speech_tokens < 6561].to(model.device)
    silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(model.device)
    speech_tokens = torch.cat([speech_tokens, silence])

    modules = selected_modules(model)
    flow_start = time.perf_counter()
    with TimingCollector(modules) as collector:
        mels = model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)
    flow_seconds = time.perf_counter() - flow_start

    rows = collector.summary()
    top_rows = rows[:25]
    result = {
        "description": "S3 flow submodule timing on the real chunk270 API-shape path.",
        "case": args.case,
        "normalized_chars": len(normalized),
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        },
        "load_seconds": load_seconds,
        "t3_runtime_load_seconds": runtime_load_seconds,
        "vulkan_t3_seconds": t3_seconds,
        "speech_token_count_with_silence": int(speech_tokens.numel()),
        "mel_shape": list(mels.shape),
        "mel_frames": int(mels.shape[-1]),
        "flow_seconds": flow_seconds,
        "timings": rows,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# S3 Flow Chunk270 Profile",
        "",
        f"- Case: `{args.case}`",
        f"- Normalized chars: `{len(normalized)}`",
        f"- Speech tokens with silence: `{int(speech_tokens.numel())}`",
        f"- Mel shape: `{list(mels.shape)}`",
        f"- Vulkan T3: `{t3_seconds:.3f}s`",
        f"- S3 flow total: `{flow_seconds:.3f}s`",
        "",
        "| Module | Calls | Inclusive s | Exclusive s | Mean incl ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in top_rows:
        lines.append(
            f"| `{row['name']}` | {row['calls']} | {row['inclusive_seconds']:.3f} | "
            f"{row['exclusive_seconds']:.3f} | {row['mean_inclusive_ms']:.3f} |"
        )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"vulkan_t3_seconds={t3_seconds:.3f}")
    print(f"flow_seconds={flow_seconds:.3f}")
    for row in top_rows[:10]:
        print(
            f"{row['name']}: calls={row['calls']} inclusive={row['inclusive_seconds']:.3f}s "
            f"exclusive={row['exclusive_seconds']:.3f}s"
        )
    print(f"wrote={json_path}")
    print(f"wrote={md_path}")


if __name__ == "__main__":
    main()
