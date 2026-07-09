#!/usr/bin/env python3
"""Capture actual S3 estimator inputs from the chunk270 API-shaped path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_tts_servers import CHUNK_270, SHORT_TEXT
from chatterbox.models.s3gen.const import S3GEN_SIL
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from t3_ggml_vulkan_runtime import T3GGMLVulkanRuntime, inference_turbo_vulkan


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports" / "s3_flow_vulkan_components" / "real_estimator_inputs"


def select_text(case: str, custom_text: str) -> str:
    return {
        "custom": custom_text,
        "short": SHORT_TEXT,
        "chunk270": CHUNK_270,
    }[case]


def np_float(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.float32).numpy()


def np_long(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().to(torch.long).numpy()


def array_summary(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("custom", "short", "chunk270"), default="chunk270")
    parser.add_argument("--text", default="Hello world, this is a short S3 capture.")
    parser.add_argument("--max-gen-len", type=int, default=420)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR / "chunk270_vulkan_t3_cpu_s3_2026-07-08",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CHATTERBOX_PROGRESS", "0")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop_threads)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    np.save(args.output_dir / "speech_tokens.npy", np_long(speech_tokens))

    estimator = model.s3gen.flow.decoder.estimator
    original_forward = estimator.forward
    active_record: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    def down_resnet_pre_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        if active_record is None:
            return
        active_record["packed_x"] = np_float(inputs[0])
        active_record["mask"] = np_float(inputs[1])
        active_record["time_emb"] = np_float(inputs[2])

    def transformer_pre_hook(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if active_record is None or "attention_bias" in active_record:
            return
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(inputs) > 1:
            attention_mask = inputs[1]
        if attention_mask is not None:
            active_record["attention_bias"] = np_float(attention_mask)

    def wrapped_forward(
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal active_record
        record: dict[str, Any] = {
            "index": len(records),
            "external_shapes": {
                "x": list(x.shape),
                "mask": list(mask.shape),
                "mu": list(mu.shape),
                "t": list(t.shape),
                "spks": list(spks.shape) if spks is not None else None,
                "cond": list(cond.shape) if cond is not None else None,
                "r": list(r.shape) if r is not None else None,
            },
            "t_value": np_float(t).reshape(-1).tolist(),
            "r_value": np_float(r).reshape(-1).tolist() if r is not None else None,
        }
        active_record = record
        started = time.perf_counter()
        try:
            output = original_forward(x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        finally:
            active_record = None
        record["cpu_estimator_seconds"] = time.perf_counter() - started
        record["cpu_output"] = np_float(output)
        records.append(record)
        return output

    down_handle = estimator.down_blocks[0][0].register_forward_pre_hook(down_resnet_pre_hook)
    transformer_handle = estimator.down_blocks[0][1][0].register_forward_pre_hook(
        transformer_pre_hook,
        with_kwargs=True,
    )
    estimator.forward = wrapped_forward

    flow_start = time.perf_counter()
    try:
        mels = model.s3gen.flow_inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
            n_cfm_timesteps=2,
            finalize=True,
        ).to(dtype=model.s3gen.dtype)
    finally:
        estimator.forward = original_forward
        down_handle.remove()
        transformer_handle.remove()
    flow_seconds = time.perf_counter() - flow_start

    call_summaries = []
    call_dirs = []
    for record in records:
        index = int(record["index"])
        call_dir = args.output_dir / f"call_{index}"
        call_dir.mkdir(parents=True, exist_ok=True)

        if "attention_bias" not in record:
            mask = record["mask"]
            record["attention_bias"] = np.where(mask > 0.5, 0.0, -1.0e10).astype(np.float32)
            record["attention_bias_fallback"] = True
        else:
            record["attention_bias_fallback"] = False

        for key in ("packed_x", "mask", "time_emb", "attention_bias", "cpu_output"):
            np.save(call_dir / f"{key}.npy", record[key])

        summary = {
            "index": index,
            "directory": call_dir.as_posix(),
            "external_shapes": record["external_shapes"],
            "t_value": record["t_value"],
            "r_value": record["r_value"],
            "cpu_estimator_seconds": record["cpu_estimator_seconds"],
            "attention_bias_fallback": record["attention_bias_fallback"],
            "packed_x": array_summary(record["packed_x"]),
            "mask": array_summary(record["mask"]),
            "time_emb": array_summary(record["time_emb"]),
            "attention_bias": array_summary(record["attention_bias"]),
            "cpu_output": array_summary(record["cpu_output"]),
            "mask_all_ones": bool(np.allclose(record["mask"], 1.0)),
            "attention_bias_all_zero": bool(np.allclose(record["attention_bias"], 0.0)),
        }
        (call_dir / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")
        call_summaries.append(summary)
        call_dirs.append(call_dir.as_posix())

    metadata = {
        "description": "Actual estimator-call tensors captured from chunk270 API-shaped S3 flow.",
        "case": args.case,
        "normalized_chars": len(normalized),
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_gen_len": args.max_gen_len,
        },
        "load_seconds": load_seconds,
        "t3_runtime_load_seconds": runtime_load_seconds,
        "vulkan_t3_seconds": t3_seconds,
        "flow_seconds": flow_seconds,
        "speech_token_count_with_silence": int(speech_tokens.numel()),
        "mel_shape": list(mels.shape),
        "estimator_calls": len(records),
        "call_dirs": call_dirs,
        "calls": call_summaries,
        "notes": [
            "T3 tokens are generated with the existing ggml/Vulkan bridge.",
            "S3 flow and estimator outputs are CPU reference outputs.",
            "Captured tensors are the packed estimator input, mask, time embedding, attention bias, and estimator output.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"output_dir={args.output_dir}")
    print(f"vulkan_t3_seconds={t3_seconds:.3f}")
    print(f"flow_seconds={flow_seconds:.3f}")
    print(f"speech_token_count_with_silence={int(speech_tokens.numel())}")
    print(f"mel_shape={list(mels.shape)}")
    print(f"estimator_calls={len(records)}")
    for summary in call_summaries:
        print(
            f"call_{summary['index']}: "
            f"packed_x={summary['packed_x']['shape']} "
            f"mask_all_ones={summary['mask_all_ones']} "
            f"attention_bias_all_zero={summary['attention_bias_all_zero']} "
            f"cpu_estimator_seconds={summary['cpu_estimator_seconds']:.3f}"
        )


if __name__ == "__main__":
    main()
