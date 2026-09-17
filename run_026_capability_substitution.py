"""Run 026: same-actor capability-substitution attack.

Public/burned calibration only. The exact frozen SmolLM2-360M actor is kept
constant while processing changes around it.

Arms:
- one-surface semantic scoring
- dual-surface evidence fusion
- uncertainty-triggered second-surface compute
- one-pass recursive scratchpad
- two-pass recursive refinement
- multi-trajectory candidate coverage on hard cases
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from offline_reasoning_battery import ChoiceScorer, build_battery, sha256_json
from offline_reasoning_content_probe import surface_task_and_choices

RUN_VERSION = "helix-capability-substitution-attack-001.0"


def log_softmax(values: list[float]) -> list[float]:
    m = max(values)
    z = m + math.log(sum(math.exp(v - m) for v in values))
    return [v - z for v in values]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1]


class AttackScorer(ChoiceScorer):
    def __init__(self, model_dir: str):
        self.score_calls = 0
        self.score_prompt_tokens = 0
        self.score_choice_tokens = 0
        self.generate_calls = 0
        self.generate_input_tokens = 0
        self.generate_output_tokens = 0
        super().__init__(model_dir)
        self.reset_counters()

    def reset_counters(self) -> None:
        self.score_calls = 0
        self.score_prompt_tokens = 0
        self.score_choice_tokens = 0
        self.generate_calls = 0
        self.generate_input_tokens = 0
        self.generate_output_tokens = 0

    def counters(self) -> dict[str, Any]:
        return {
            "score_forward_batches": self.score_calls,
            "score_prompt_tokens": self.score_prompt_tokens,
            "score_choice_tokens": self.score_choice_tokens,
            "generate_calls": self.generate_calls,
            "generate_input_tokens": self.generate_input_tokens,
            "generate_output_tokens": self.generate_output_tokens,
        }

    @torch.inference_mode()
    def score_choices(self, prompt: str, choices):
        choices = list(choices)
        if hasattr(self, "tokenizer"):
            self.score_calls += 1
            self.score_prompt_tokens += len(
                self.tokenizer(prompt, add_special_tokens=True).input_ids
            )
            self.score_choice_tokens += sum(
                len(self.tokenizer(" " + str(c), add_special_tokens=False).input_ids)
                for c in choices
            )
        return super().score_choices(prompt, choices)

    @torch.inference_mode()
    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        sample: bool,
    ) -> str:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded.get("attention_mask")
        if attention is not None:
            attention = attention.to(self.device)
        torch.manual_seed(seed)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": sample,
        }
        if sample:
            kwargs.update({"temperature": 0.8, "top_p": 0.95})
        output = self.model.generate(**kwargs)
        completion = output[0, input_ids.shape[1]:]
        self.generate_calls += 1
        self.generate_input_tokens += int(input_ids.shape[1])
        self.generate_output_tokens += int(completion.shape[0])
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def score_prompt(scorer: AttackScorer, prompt: str) -> dict[str, Any]:
    task_prompt, choices = surface_task_and_choices(prompt)
    raw = scorer.score_choices(task_prompt, choices)
    prior = scorer.score_choices("Answer value:", choices)
    pmi = [raw[i] - prior[i] for i in range(len(choices))]
    normalized = log_softmax(pmi)
    return {
        "prediction": argmax(normalized),
        "scores": normalized,
        "margin": margin(normalized),
    }


def scratchpad_prompt(prompt: str, prior: str | None = None) -> str:
    task_prompt, _choices = surface_task_and_choices(prompt)
    stem = task_prompt.removesuffix("\nAnswer value:")
    if prior is None:
        return (
            stem
            + "\nWork through the structure carefully. Do not use A/B/C/D labels."
            + "\nReasoning:"
        )
    return (
        stem
        + "\nFirst reasoning attempt:\n"
        + prior
        + "\nCheck the attempt, repair any mistake, and derive a cleaner result."
        + "\nRevised reasoning:"
    )


def score_with_reasoning(
    scorer: AttackScorer,
    prompt: str,
    reasoning: str,
) -> dict[str, Any]:
    task_prompt, choices = surface_task_and_choices(prompt)
    stem = task_prompt.removesuffix("\nAnswer value:")
    augmented = (
        stem
        + "\nIntermediate reasoning:\n"
        + reasoning
        + "\nAnswer value:"
    )
    raw = scorer.score_choices(augmented, choices)
    prior = scorer.score_choices("Answer value:", choices)
    pmi = [raw[i] - prior[i] for i in range(len(choices))]
    normalized = log_softmax(pmi)
    return {
        "prediction": argmax(normalized),
        "scores": normalized,
        "margin": margin(normalized),
    }


def fuse(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    scores = [a["scores"][i] + b["scores"][i] for i in range(4)]
    return {
        "prediction": argmax(scores),
        "scores": scores,
        "margin": margin(scores),
    }


def summarize(cases: list[dict[str, Any]], predictions: dict[str, int]) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    correct = 0
    for case in cases:
        ok = predictions[case["case_id"]] == case["correct_index"]
        correct += int(ok)
        by_family[case["family"]].append(ok)
        by_difficulty[case["difficulty"]].append(ok)
    return {
        "cases": len(cases),
        "correct": correct,
        "accuracy": correct / len(cases),
        "by_family": {
            key: sum(values) / len(values)
            for key, values in sorted(by_family.items())
        },
        "by_difficulty": {
            key: sum(values) / len(values)
            for key, values in sorted(by_difficulty.items())
        },
    }


def evaluate(model_dir: str, seed: int, cases_per_family: int) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    scorer = AttackScorer(model_dir)
    all_started = time.time()

    primary_surface = {
        case["case_id"]: sorted(case["surfaces"])[0] for case in cases
    }
    secondary_surface = {
        case["case_id"]: sorted(case["surfaces"])[1] for case in cases
    }

    scorer.reset_counters()
    started = time.time()
    primary: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = primary_surface[case["case_id"]]
        primary[case["case_id"]] = score_prompt(scorer, case["surfaces"][name])
    direct_resources = scorer.counters()
    direct_resources["wall_seconds"] = time.time() - started
    direct_pred = {cid: row["prediction"] for cid, row in primary.items()}

    scorer.reset_counters()
    started = time.time()
    secondary: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = secondary_surface[case["case_id"]]
        secondary[case["case_id"]] = score_prompt(scorer, case["surfaces"][name])
    secondary_resources = scorer.counters()
    secondary_resources["wall_seconds"] = time.time() - started

    fused = {
        case["case_id"]: fuse(
            primary[case["case_id"]],
            secondary[case["case_id"]],
        )
        for case in cases
    }
    fused_pred = {cid: row["prediction"] for cid, row in fused.items()}
    fused_resources = {
        key: direct_resources.get(key, 0) + secondary_resources.get(key, 0)
        for key in set(direct_resources) | set(secondary_resources)
    }

    uncertainty_order = sorted(
        cases,
        key=lambda case: (
            primary[case["case_id"]]["margin"],
            case["case_id"],
        ),
    )
    adaptive_curve = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        extra = int(round(len(cases) * fraction))
        upgraded = {case["case_id"] for case in uncertainty_order[:extra]}
        predictions = {
            case["case_id"]: (
                fused[case["case_id"]]["prediction"]
                if case["case_id"] in upgraded
                else primary[case["case_id"]]["prediction"]
            )
            for case in cases
        }
        stats = summarize(cases, predictions)
        adaptive_curve.append({
            "second_surface_fraction": fraction,
            "second_surface_cases": extra,
            **stats,
            "estimated_score_forward_batches": (
                direct_resources["score_forward_batches"]
                + fraction * secondary_resources["score_forward_batches"]
            ),
            "selection_signal": "lowest primary-surface margin only; labels unused",
        })

    scorer.reset_counters()
    started = time.time()
    recursive1_pred: dict[str, int] = {}
    recursive1_rows = []
    for i, case in enumerate(cases):
        name = primary_surface[case["case_id"]]
        prompt = case["surfaces"][name]
        reasoning = scorer.generate_text(
            scratchpad_prompt(prompt),
            max_new_tokens=32,
            seed=seed + i,
            sample=False,
        )
        scored = score_with_reasoning(scorer, prompt, reasoning)
        recursive1_pred[case["case_id"]] = scored["prediction"]
        recursive1_rows.append({
            "case_id": case["case_id"],
            "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            "reasoning_tokens": len(
                scorer.tokenizer(reasoning, add_special_tokens=False).input_ids
            ),
            "prediction": scored["prediction"],
            "margin": scored["margin"],
        })
    recursive1_resources = scorer.counters()
    recursive1_resources["wall_seconds"] = time.time() - started

    scorer.reset_counters()
    started = time.time()
    recursive2_pred: dict[str, int] = {}
    recursive2_rows = []
    for i, case in enumerate(cases):
        name = primary_surface[case["case_id"]]
        prompt = case["surfaces"][name]
        first = scorer.generate_text(
            scratchpad_prompt(prompt),
            max_new_tokens=24,
            seed=seed + 1000 + i,
            sample=False,
        )
        revised = scorer.generate_text(
            scratchpad_prompt(prompt, first),
            max_new_tokens=32,
            seed=seed + 2000 + i,
            sample=False,
        )
        scored = score_with_reasoning(scorer, prompt, revised)
        recursive2_pred[case["case_id"]] = scored["prediction"]
        recursive2_rows.append({
            "case_id": case["case_id"],
            "first_sha256": hashlib.sha256(first.encode()).hexdigest(),
            "revised_sha256": hashlib.sha256(revised.encode()).hexdigest(),
            "prediction": scored["prediction"],
            "margin": scored["margin"],
        })
    recursive2_resources = scorer.counters()
    recursive2_resources["wall_seconds"] = time.time() - started

    hard_cases = [case for case in cases if case["difficulty"] == "hard"]
    scorer.reset_counters()
    started = time.time()
    coverage_rows = []
    selected_pred: dict[str, int] = {}
    oracle_hits = 0
    for i, case in enumerate(hard_cases):
        name = primary_surface[case["case_id"]]
        prompt = case["surfaces"][name]
        candidates = []
        for sample_index in range(3):
            reasoning = scorer.generate_text(
                scratchpad_prompt(prompt),
                max_new_tokens=28,
                seed=seed + 3000 + i * 10 + sample_index,
                sample=True,
            )
            scored = score_with_reasoning(scorer, prompt, reasoning)
            candidates.append({
                "prediction": scored["prediction"],
                "scores": scored["scores"],
                "margin": scored["margin"],
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
            })
        correct = case["correct_index"]
        oracle = any(row["prediction"] == correct for row in candidates)
        oracle_hits += int(oracle)
        votes = Counter(row["prediction"] for row in candidates)
        best_count = max(votes.values())
        tied = {choice for choice, count in votes.items() if count == best_count}
        if len(tied) == 1:
            selected = next(iter(tied))
        else:
            total_scores = {
                choice: sum(row["scores"][choice] for row in candidates)
                for choice in tied
            }
            selected = max(total_scores, key=total_scores.get)
        selected_pred[case["case_id"]] = selected
        coverage_rows.append({
            "case_id": case["case_id"],
            "candidate_predictions": [row["prediction"] for row in candidates],
            "unique_predictions": len(set(row["prediction"] for row in candidates)),
            "oracle_contains_correct": oracle,
            "selected_prediction": selected,
            "selected_correct": selected == correct,
        })
    coverage_resources = scorer.counters()
    coverage_resources["wall_seconds"] = time.time() - started

    public_spec = [
        {
            "case_id": c["case_id"],
            "family": c["family"],
            "difficulty": c["difficulty"],
            "correct_index": c["correct_index"],
            "surface_sha256": {
                k: hashlib.sha256(v.encode()).hexdigest()
                for k, v in c["surfaces"].items()
            },
        }
        for c in cases
    ]

    return {
        "schema_version": RUN_VERSION,
        "status": "public-burned-calibration",
        "seed": seed,
        "cases_per_family": cases_per_family,
        "source_battery_sha256": sha256_json(public_spec),
        "same_actor_required": True,
        "arms": {
            "direct-content": {
                "summary": summarize(cases, direct_pred),
                "resources": direct_resources,
                "primary_surface": primary_surface,
            },
            "dual-surface-fusion": {
                "summary": summarize(cases, fused_pred),
                "resources": fused_resources,
                "fusion": "sum of surface-native PMI log-probabilities",
            },
            "adaptive-fusion": {
                "curve": adaptive_curve,
                "full_scoring_wall_seconds": (
                    direct_resources["wall_seconds"]
                    + secondary_resources["wall_seconds"]
                ),
            },
            "recursive-scratchpad-1": {
                "summary": summarize(cases, recursive1_pred),
                "resources": recursive1_resources,
                "cases": recursive1_rows,
            },
            "recursive-scratchpad-2": {
                "summary": summarize(cases, recursive2_pred),
                "resources": recursive2_resources,
                "cases": recursive2_rows,
            },
            "candidate-coverage-hard": {
                "cases": len(hard_cases),
                "oracle_correct": oracle_hits,
                "oracle_coverage": oracle_hits / len(hard_cases),
                "selected_summary": summarize(hard_cases, selected_pred),
                "resources": coverage_resources,
                "rows": coverage_rows,
                "oracle_role": "posthoc diagnostic only",
            },
        },
        "total_elapsed_seconds": time.time() - all_started,
        "claim_boundary": (
            "Public/burned procedural calibration only. Same frozen actor. "
            "This run measures processing deltas and bottlenecks; it is not protected "
            "promotion evidence, does not establish frontier equivalence, and cannot "
            "certify a capacity wall."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-026-attack.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(args.model_dir, args.seed, args.cases_per_family)
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "source_battery_sha256": result["source_battery_sha256"],
        "arms": {
            name: (
                value.get("summary")
                or value.get("selected_summary")
                or value.get("curve")
            )
            for name, value in result["arms"].items()
        },
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
