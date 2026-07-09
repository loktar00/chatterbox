#!/usr/bin/env python3
"""Validate exact T3 sampler variants without running Chatterbox or Vulkan."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path("/root/chatterbox")
OUT_JSON = ROOT / "exports/benchmarks/t3_seen_mask_sampler_validation_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/t3_seen_mask_sampler_validation_2026-07-08.md"


def filter_scores(
    speech_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    scores = speech_logits.clone()
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    if top_k > 0:
        top_k = min(top_k, scores.size(-1))
        top_values, top_indices = torch.topk(scores, top_k)
        top_k_threshold = top_values[..., -1, None]
        indices_to_remove = scores < top_k_threshold
        if top_p < 1.0 and scores.size(0) == 1:
            sorted_logits, sorted_order = torch.sort(top_values, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[..., -1:] = 0
            top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
            top_indices_to_remove = top_indices_to_remove.scatter(
                1,
                sorted_order,
                sorted_indices_to_remove,
            )
            scores = torch.full_like(scores, -float("inf"))
            scores = scores.scatter(1, top_indices, top_values.masked_fill(top_indices_to_remove, -float("inf")))
        else:
            scores = scores.masked_fill(indices_to_remove, -float("inf"))
    if top_p < 1.0 and not (top_k > 0 and scores.size(0) == 1):
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -1:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float("inf"))
    return scores


def current_scores(
    input_ids: torch.LongTensor,
    speech_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    scores = filter_scores(speech_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if repetition_penalty != 1.0:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        scores = scores.scatter(1, input_ids, score)
    return scores


def seen_mask_scores(
    seen_token_mask: torch.Tensor,
    speech_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    scores = filter_scores(speech_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if repetition_penalty != 1.0:
        penalized = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        scores = torch.where(seen_token_mask, penalized, scores)
    return scores


def unique_id_scores(
    input_ids: torch.LongTensor,
    speech_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.Tensor:
    scores = filter_scores(speech_logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if repetition_penalty != 1.0:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        unique_ids = torch.unique(input_ids, sorted=False).unsqueeze(0)
        score = torch.gather(scores, 1, unique_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        scores = scores.scatter(1, unique_ids, score)
    return scores


def sample_from_scores(scores: torch.Tensor) -> torch.LongTensor | None:
    if torch.all(scores == -float("inf")):
        return None
    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def equal_scores(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left, right))


def case_run(seen_len: int, trials: int, vocab: int, params: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(20260708 + seen_len)
    base_input = torch.randint(0, vocab, (1, seen_len), generator=generator)
    if seen_len > 4:
        # Force duplicates, matching real autoregressive speech tokens.
        base_input[:, ::7] = base_input[:, 0:1]
    seen_mask = torch.zeros((1, vocab), dtype=torch.bool)
    seen_mask.scatter_(1, base_input, True)
    unique_ids = torch.unique(base_input, sorted=False)

    score_equal = {"seen_mask": True, "unique_ids": True}
    token_equal = {"seen_mask": True, "unique_ids": True}
    first_mismatch: dict[str, Any] = {}
    logits_bank = [torch.randn((1, vocab), generator=generator) for _ in range(trials)]
    seeds = [int(10_000 + i) for i in range(trials)]

    for index, logits in enumerate(logits_bank):
        current = current_scores(base_input, logits, **params)
        seen = seen_mask_scores(seen_mask, logits, **params)
        unique = unique_id_scores(base_input, logits, **params)
        for name, candidate in (("seen_mask", seen), ("unique_ids", unique)):
            if score_equal[name] and not equal_scores(current, candidate):
                finite = torch.isfinite(current) | torch.isfinite(candidate)
                max_abs = torch.max(torch.abs(current[finite] - candidate[finite])).item() if torch.any(finite) else 0.0
                score_equal[name] = False
                first_mismatch.setdefault(name, {"trial": index, "score_max_abs": max_abs})
            torch.manual_seed(seeds[index])
            token_current = sample_from_scores(current)
            torch.manual_seed(seeds[index])
            token_candidate = sample_from_scores(candidate)
            same_token = (
                token_current is None
                and token_candidate is None
                or token_current is not None
                and token_candidate is not None
                and torch.equal(token_current, token_candidate)
            )
            if token_equal[name] and not same_token:
                token_equal[name] = False
                first_mismatch.setdefault(name, {}).update(
                    {
                        "token_trial": index,
                        "current": None if token_current is None else int(token_current.item()),
                        "candidate": None if token_candidate is None else int(token_candidate.item()),
                    }
                )

    timings: dict[str, float] = {}
    for name, fn in (
        ("current", lambda logits: current_scores(base_input, logits, **params)),
        ("seen_mask", lambda logits: seen_mask_scores(seen_mask, logits, **params)),
        ("unique_ids", lambda logits: unique_id_scores(base_input, logits, **params)),
    ):
        started = time.perf_counter()
        for index, logits in enumerate(logits_bank):
            scores = fn(logits)
            torch.manual_seed(seeds[index])
            sample_from_scores(scores)
        timings[name] = time.perf_counter() - started

    return {
        "seen_len": seen_len,
        "unique_len": int(unique_ids.numel()),
        "trials": trials,
        "score_equal": score_equal,
        "token_equal": token_equal,
        "first_mismatch": first_mismatch,
        "timings": timings,
        "speedup_vs_current": {
            "seen_mask": timings["current"] / timings["seen_mask"] if timings["seen_mask"] > 0 else None,
            "unique_ids": timings["current"] / timings["unique_ids"] if timings["unique_ids"] > 0 else None,
        },
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# T3 Seen-Mask Sampler Validation - 2026-07-08",
        "",
        f"- Overall exact: `{report['overall_exact']}`",
        f"- Recommended runtime change: `{report['recommended_runtime_change']}`",
        "",
        "| Seen Len | Unique Len | Variant | Scores Equal | Tokens Equal | Time | Speedup |",
        "| ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for case in report["cases"]:
        for variant in ("seen_mask", "unique_ids"):
            lines.append(
                f"| {case['seen_len']} | {case['unique_len']} | {variant} | "
                f"{case['score_equal'][variant]} | {case['token_equal'][variant]} | "
                f"`{case['timings'][variant]:.4f}s` | "
                f"`{case['speedup_vs_current'][variant]:.3f}x` |"
            )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--vocab", type=int, default=8192)
    args = parser.parse_args()

    torch.set_num_threads(1)
    params = {
        "temperature": 0.8,
        "top_k": 1000,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    }
    cases = [case_run(seen_len, args.trials, args.vocab, params) for seen_len in (1, 64, 358)]
    exact = all(
        case["score_equal"]["seen_mask"]
        and case["token_equal"]["seen_mask"]
        and case["score_equal"]["unique_ids"]
        and case["token_equal"]["unique_ids"]
        for case in cases
    )
    seen_mask_speedups = [case["speedup_vs_current"]["seen_mask"] or 0.0 for case in cases]
    unique_speedups = [case["speedup_vs_current"]["unique_ids"] or 0.0 for case in cases]
    best_seen = min(seen_mask_speedups)
    best_unique = min(unique_speedups)
    recommended = "none"
    if exact and best_seen >= 1.10:
        recommended = "consider_seen_mask_guarded"
    elif exact and best_unique >= 1.10:
        recommended = "consider_unique_ids_guarded"

    decision = (
        "No runtime change. The variants must be exact and consistently at least 1.10x faster before wiring a new guarded path."
    )
    if recommended != "none":
        decision = (
            f"{recommended} passed the CPU-only threshold. Wire only behind an opt-in env flag and validate token equality on the real T3 loop before benchmarking audio."
        )
    report = {
        "description": "CPU-only validation of exact T3 repetition-penalty sampler variants.",
        "params": params,
        "torch_threads": torch.get_num_threads(),
        "overall_exact": exact,
        "recommended_runtime_change": recommended,
        "decision": decision,
        "cases": cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"recommended_runtime_change={recommended}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
