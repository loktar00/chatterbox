#!/usr/bin/env python3
"""Profile T3 sampler candidates without Chatterbox model or Vulkan execution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path("/root/chatterbox")
OUT_JSON = ROOT / "exports/benchmarks/t3_sampler_candidate_profile_2026-07-08.json"
OUT_MD = ROOT / "exports/benchmarks/t3_sampler_candidate_profile_2026-07-08.md"


PARAMS = {
    "temperature": 0.8,
    "top_k": 1000,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
}


def make_case(seen_len: int, vocab: int, trials: int, seed: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed + seen_len)
    input_ids = torch.randint(0, vocab, (1, seen_len), generator=gen)
    if seen_len > 4:
        input_ids[:, ::7] = input_ids[:, 0:1]
    seen_mask = torch.zeros((1, vocab), dtype=torch.bool)
    seen_mask.scatter_(1, input_ids, True)
    logits_bank = [torch.randn((1, vocab), generator=gen) for _ in range(trials)]
    return {
        "seen_len": seen_len,
        "unique_len": int(torch.unique(input_ids).numel()),
        "input_ids": input_ids,
        "seen_mask": seen_mask,
        "logits_bank": logits_bank,
        "seeds": [seed * 10 + i for i in range(trials)],
    }


def current_full_vocab(input_ids: torch.LongTensor, logits: torch.Tensor, timings: dict[str, float]) -> torch.LongTensor | None:
    temperature = PARAMS["temperature"]
    top_k = PARAMS["top_k"]
    top_p = PARAMS["top_p"]
    repetition_penalty = PARAMS["repetition_penalty"]

    started = time.perf_counter()
    scores = logits
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    timings["temperature"] += time.perf_counter() - started

    started = time.perf_counter()
    top_k = min(top_k, scores.size(-1))
    top_values, top_indices = torch.topk(scores, top_k)
    top_k_threshold = top_values[..., -1, None]
    _indices_to_remove = scores < top_k_threshold
    timings["topk"] += time.perf_counter() - started

    started = time.perf_counter()
    sorted_logits, sorted_order = torch.sort(top_values, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -1:] = 0
    top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    top_indices_to_remove = top_indices_to_remove.scatter(1, sorted_order, sorted_indices_to_remove)
    timings["top_p"] += time.perf_counter() - started

    started = time.perf_counter()
    scores = torch.full_like(scores, -float("inf"))
    scores = scores.scatter(1, top_indices, top_values.masked_fill(top_indices_to_remove, -float("inf")))
    timings["full_vocab_scatter"] += time.perf_counter() - started

    started = time.perf_counter()
    score = torch.gather(scores, 1, input_ids)
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
    scores = scores.scatter(1, input_ids, score)
    timings["repetition"] += time.perf_counter() - started

    started = time.perf_counter()
    if torch.all(scores == -float("inf")):
        timings["softmax_multinomial"] += time.perf_counter() - started
        return None
    probs = F.softmax(scores, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    timings["softmax_multinomial"] += time.perf_counter() - started
    return token


def compact_candidate(
    input_ids: torch.LongTensor,
    seen_mask: torch.Tensor,
    logits: torch.Tensor,
    timings: dict[str, float],
) -> torch.LongTensor | None:
    temperature = PARAMS["temperature"]
    top_k = PARAMS["top_k"]
    top_p = PARAMS["top_p"]
    repetition_penalty = PARAMS["repetition_penalty"]

    started = time.perf_counter()
    scores = logits
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    timings["temperature"] += time.perf_counter() - started

    started = time.perf_counter()
    top_k = min(top_k, scores.size(-1))
    top_values, top_indices = torch.topk(scores, top_k)
    timings["topk"] += time.perf_counter() - started

    started = time.perf_counter()
    sorted_logits, sorted_order = torch.sort(top_values, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -1:] = 0
    top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    top_indices_to_remove = top_indices_to_remove.scatter(1, sorted_order, sorted_indices_to_remove)
    top_values = top_values.masked_fill(top_indices_to_remove, -float("inf"))
    timings["top_p"] += time.perf_counter() - started

    started = time.perf_counter()
    seen_top = seen_mask.gather(1, top_indices)
    penalized = torch.where(top_values < 0, top_values * repetition_penalty, top_values / repetition_penalty)
    top_values = torch.where(seen_top, penalized, top_values)
    timings["repetition"] += time.perf_counter() - started

    started = time.perf_counter()
    finite = torch.isfinite(top_values)
    if not torch.any(finite):
        timings["softmax_multinomial"] += time.perf_counter() - started
        return None
    candidate_values = top_values[finite].unsqueeze(0)
    candidate_indices = top_indices[finite].unsqueeze(0)
    probs = F.softmax(candidate_values, dim=-1)
    sampled_pos = torch.multinomial(probs, num_samples=1)
    token = candidate_indices.gather(1, sampled_pos)
    timings["softmax_multinomial"] += time.perf_counter() - started
    return token


def compact_candidate_no_finite_pack(
    input_ids: torch.LongTensor,
    seen_mask: torch.Tensor,
    logits: torch.Tensor,
    timings: dict[str, float],
) -> torch.LongTensor | None:
    temperature = PARAMS["temperature"]
    top_k = PARAMS["top_k"]
    top_p = PARAMS["top_p"]
    repetition_penalty = PARAMS["repetition_penalty"]

    started = time.perf_counter()
    scores = logits
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    timings["temperature"] += time.perf_counter() - started

    started = time.perf_counter()
    top_k = min(top_k, scores.size(-1))
    top_values, top_indices = torch.topk(scores, top_k)
    timings["topk"] += time.perf_counter() - started

    started = time.perf_counter()
    sorted_logits, sorted_order = torch.sort(top_values, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    sorted_indices_to_remove[..., -1:] = 0
    top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    top_indices_to_remove = top_indices_to_remove.scatter(1, sorted_order, sorted_indices_to_remove)
    top_values = top_values.masked_fill(top_indices_to_remove, -float("inf"))
    timings["top_p"] += time.perf_counter() - started

    started = time.perf_counter()
    seen_top = seen_mask.gather(1, top_indices)
    penalized = torch.where(top_values < 0, top_values * repetition_penalty, top_values / repetition_penalty)
    top_values = torch.where(seen_top, penalized, top_values)
    timings["repetition"] += time.perf_counter() - started

    started = time.perf_counter()
    if torch.all(top_values == -float("inf")):
        timings["softmax_multinomial"] += time.perf_counter() - started
        return None
    probs = F.softmax(top_values, dim=-1)
    sampled_pos = torch.multinomial(probs, num_samples=1)
    token = top_indices.gather(1, sampled_pos)
    timings["softmax_multinomial"] += time.perf_counter() - started
    return token


def empty_timings() -> dict[str, float]:
    return {
        "temperature": 0.0,
        "topk": 0.0,
        "top_p": 0.0,
        "full_vocab_scatter": 0.0,
        "repetition": 0.0,
        "softmax_multinomial": 0.0,
    }


def run_variant(case: dict[str, Any], variant: str) -> dict[str, Any]:
    timings = empty_timings()
    started = time.perf_counter()
    tokens: list[int | None] = []
    for index, logits in enumerate(case["logits_bank"]):
        torch.manual_seed(case["seeds"][index])
        if variant == "current_full_vocab":
            token = current_full_vocab(case["input_ids"], logits, timings)
        elif variant == "compact_finite_pack":
            token = compact_candidate(case["input_ids"], case["seen_mask"], logits, timings)
        elif variant == "compact_no_finite_pack":
            token = compact_candidate_no_finite_pack(case["input_ids"], case["seen_mask"], logits, timings)
        else:
            raise ValueError(variant)
        tokens.append(None if token is None else int(token.item()))
    total = time.perf_counter() - started
    return {"total_seconds": total, "timings": timings, "tokens": tokens}


def compare_distributions(case: dict[str, Any], samples: int, seed: int) -> dict[str, Any]:
    """Compare current and compact probability vectors for one representative logit row."""

    del samples, seed
    logits = case["logits_bank"][0]

    current_scores = logits.clone()
    if PARAMS["temperature"] > 0 and PARAMS["temperature"] != 1.0:
        current_scores = current_scores / PARAMS["temperature"]
    top_values, top_indices = torch.topk(current_scores, min(PARAMS["top_k"], current_scores.size(-1)))
    sorted_logits, sorted_order = torch.sort(top_values, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - PARAMS["top_p"])
    sorted_indices_to_remove[..., -1:] = 0
    top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
    top_indices_to_remove = top_indices_to_remove.scatter(1, sorted_order, sorted_indices_to_remove)

    current_scores = torch.full_like(current_scores, -float("inf"))
    current_scores = current_scores.scatter(
        1,
        top_indices,
        top_values.masked_fill(top_indices_to_remove, -float("inf")),
    )
    score = torch.gather(current_scores, 1, case["input_ids"])
    score = torch.where(score < 0, score * PARAMS["repetition_penalty"], score / PARAMS["repetition_penalty"])
    current_scores = current_scores.scatter(1, case["input_ids"], score)

    compact_values = top_values.masked_fill(top_indices_to_remove, -float("inf"))
    seen_top = case["seen_mask"].gather(1, top_indices)
    penalized = torch.where(
        compact_values < 0,
        compact_values * PARAMS["repetition_penalty"],
        compact_values / PARAMS["repetition_penalty"],
    )
    compact_values = torch.where(seen_top, penalized, compact_values)
    compact_scores = torch.full_like(current_scores, -float("inf"))
    compact_scores = compact_scores.scatter(1, top_indices, compact_values)

    current_support = torch.isfinite(current_scores)
    compact_support = torch.isfinite(compact_scores)
    current_probs = F.softmax(current_scores, dim=-1)
    compact_probs = F.softmax(compact_scores, dim=-1)
    diff = torch.abs(current_probs - compact_probs)
    return {
        "support_equal": bool(torch.equal(current_support, compact_support)),
        "prob_max_abs": float(torch.max(diff).item()),
        "prob_l1": float(torch.sum(diff).item()),
        "finite_support_count": int(torch.sum(current_support).item()),
    }


def profile_case(seen_len: int, vocab: int, trials: int, seed: int) -> dict[str, Any]:
    case = make_case(seen_len, vocab, trials, seed)
    variants = {
        name: run_variant(case, name)
        for name in ("current_full_vocab", "compact_finite_pack", "compact_no_finite_pack")
    }
    current_total = variants["current_full_vocab"]["total_seconds"]
    for name, result in variants.items():
        result["speedup_vs_current"] = current_total / result["total_seconds"] if result["total_seconds"] > 0 else None
        del result["tokens"]
    return {
        "seen_len": case["seen_len"],
        "unique_len": case["unique_len"],
        "trials": trials,
        "vocab": vocab,
        "variants": variants,
        "distribution_smoke": compare_distributions(case, min(128, trials), seed + 99),
    }


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# T3 Sampler Candidate Profile - 2026-07-08",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Runtime wiring: `{report['runtime_wiring']}`",
        "",
        "| Seen Len | Variant | Total | Speedup | TopK | TopP | Scatter | Repetition | Softmax+Sample |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        for name, item in case["variants"].items():
            timings = item["timings"]
            lines.append(
                f"| {case['seen_len']} | {name} | `{item['total_seconds']:.4f}s` | "
                f"`{item['speedup_vs_current']:.3f}x` | `{timings['topk']:.4f}s` | "
                f"`{timings['top_p']:.4f}s` | `{timings['full_vocab_scatter']:.4f}s` | "
                f"`{timings['repetition']:.4f}s` | `{timings['softmax_multinomial']:.4f}s` |"
            )
    lines.extend(
        [
            "",
            "## Probability Check",
            "",
            "| Seen Len | Support Equal | Max Abs Prob Diff | L1 Prob Diff | Finite Support |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for case in report["cases"]:
        dist = case["distribution_smoke"]
        lines.append(
            f"| {case['seen_len']} | {dist['support_equal']} | "
            f"`{dist['prob_max_abs']:.3g}` | `{dist['prob_l1']:.3g}` | "
            f"{dist['finite_support_count']} |"
        )
    lines.extend(["", "## Decision", "", report["decision"], ""])
    OUT_MD.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--vocab", type=int, default=8192)
    args = parser.parse_args()

    torch.set_num_threads(1)
    cases = [profile_case(seen_len, args.vocab, args.trials, 20260708) for seen_len in (1, 64, 358)]
    compact_speedups = [
        case["variants"]["compact_no_finite_pack"]["speedup_vs_current"] or 0.0
        for case in cases
    ]
    min_compact_speedup = min(compact_speedups)
    recommendation = "no_runtime_change"
    runtime_wiring = False
    decision = (
        "No runtime change. The compact distribution-equivalent sampler does not clear the 1.20x minimum speedup threshold across cases, and it is not token-identical."
    )
    if min_compact_speedup >= 1.20:
        recommendation = "consider_listen_before_default_guarded_sampler"
        runtime_wiring = False
        decision = (
            "The compact sampler is fast enough to justify a guarded listen-before-default experiment, but it must still be validated on real generated audio before default use."
        )
    report = {
        "description": "CPU-only T3 sampler candidate profile; no Chatterbox model load and no Vulkan execution.",
        "params": PARAMS,
        "torch_threads": torch.get_num_threads(),
        "recommendation": recommendation,
        "runtime_wiring": runtime_wiring,
        "decision": decision,
        "cases": cases,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"recommendation={recommendation}")
    print(f"min_compact_speedup={min_compact_speedup:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
