from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

CANONICAL_HELIX_COMMIT = "4e6270e7e4ca198bd9adea2d8261e855b98b6283"
HELIX_INPUT = Path(__file__).resolve().parent / "verification" / "helix-4e6270e7"
sys.path.insert(0, str(HELIX_INPUT))

from engine.runtime.experiments.cellular_automata.rule30_transform_program import (  # noqa: E402
    compile_rule30_branch_mechanism,
    compile_rule30_branch_mechanism_task,
)

RUN_VERSION = "helix-rule30-compiler-choice-interface-001.0"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "028493fd3c93bfb0536d0b07a124d8e302e187dd"
MODEL_SHA256 = "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
HELDOUT_WIDTHS = (8, 16, 32)
CANDIDATES = (
    "OR(E,S)",
    "AND(E,S)",
    "E",
    "S",
    "NOT(E)",
    "NOT(S)",
    "FALSE",
    "TRUE",
)
EXPECTED = "OR(E,S)"


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _margin(values: list[float]) -> float:
    ordered = sorted(values, reverse=True)
    return ordered[0] - ordered[1]


def _score_candidates(model_dir: str, prompt: str, choices: tuple[str, ...]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    torch.set_num_threads(4)

    def render(user_text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def scores(user_text: str) -> tuple[list[float], dict[str, int]]:
        rendered = render(user_text)
        prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        sequences: list[list[int]] = []
        choice_ids_list: list[list[int]] = []
        for choice in choices:
            choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
            if not choice_ids:
                raise ValueError(f"empty candidate tokenization: {choice!r}")
            choice_ids_list.append(choice_ids)
            sequences.append(prompt_ids + choice_ids)

        max_len = max(len(row) for row in sequences)
        input_ids = torch.full(
            (len(sequences), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(input_ids)
        for index, sequence in enumerate(sequences):
            input_ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            attention[index, : len(sequence)] = 1

        logits = model(input_ids=input_ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        prefix_len = len(prompt_ids)
        out: list[float] = []
        for index, choice_ids in enumerate(choice_ids_list):
            total = 0.0
            for offset, token in enumerate(choice_ids):
                pos = prefix_len + offset
                total += float(log_probs[index, pos - 1, token])
            out.append(total / len(choice_ids))
        return out, {
            "prompt_tokens": len(prompt_ids),
            "choice_tokens": sum(len(row) for row in choice_ids_list),
        }

    started = time.time()
    raw, raw_cost = scores(prompt)
    prior, prior_cost = scores("Formula:")
    pmi = [raw[index] - prior[index] for index in range(len(choices))]
    elapsed = time.time() - started
    return {
        "raw_scores": dict(zip(choices, raw)),
        "prior_scores": dict(zip(choices, prior)),
        "pmi_scores": dict(zip(choices, pmi)),
        "raw_prediction": choices[_argmax(raw)],
        "pmi_prediction": choices[_argmax(pmi)],
        "raw_margin": _margin(raw),
        "pmi_margin": _margin(pmi),
        "resources": {
            "score_forward_batches": 2,
            "prompt_tokens": raw_cost["prompt_tokens"] + prior_cost["prompt_tokens"],
            "choice_tokens": raw_cost["choice_tokens"] + prior_cost["choice_tokens"],
            "wall_seconds": elapsed,
            "generate_calls": 0,
        },
    }


def _training_patterns() -> list[dict[str, Any]]:
    task = compile_rule30_branch_mechanism_task()["task"]
    return [dict(row) for row in task["training_patterns"]]


def _prompts(patterns: list[dict[str, Any]]) -> dict[str, str]:
    plain = (
        "Exact Boolean training rows for branch(E,S): "
        + "; ".join(
            f"E={int(row['E'])}, S={int(row['S'])} -> branch={int(row['branch'])} "
            f"({row['count']} states)"
            for row in patterns
        )
        + ". Choose the shortest exact formula matching every row. Formula:"
    )
    compact = (
        "branch(E,S) exact table with multiplicities: "
        + " ".join(
            f"{int(row['E'])}{int(row['S'])}->{int(row['branch'])}x{row['count']}"
            for row in patterns
        )
        + ". Shortest matching Boolean formula:"
    )
    return {"plain": plain, "compact": compact}


def _candidate_value(candidate: str, exceptional: bool, derivative_chain: bool) -> bool:
    if candidate == "OR(E,S)":
        return exceptional or derivative_chain
    if candidate == "AND(E,S)":
        return exceptional and derivative_chain
    if candidate == "E":
        return exceptional
    if candidate == "S":
        return derivative_chain
    if candidate == "NOT(E)":
        return not exceptional
    if candidate == "NOT(S)":
        return not derivative_chain
    if candidate == "FALSE":
        return False
    if candidate == "TRUE":
        return True
    raise KeyError(candidate)


def _judge(candidate: str, width: int) -> dict[str, Any]:
    mechanism = compile_rule30_branch_mechanism(width)["mechanism"]
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in mechanism["rows"]:
        predicted = _candidate_value(
            candidate,
            bool(row["exceptional_pair"]),
            bool(row["derivative_chain"]),
        )
        actual = bool(row["is_branch"])
        if predicted and actual:
            counts["tp"] += 1
        elif predicted and not actual:
            counts["fp"] += 1
        elif not predicted and actual:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return {
        "width": width,
        "basin_states": mechanism["state_count"],
        "actual_branch_states": mechanism["branch_count"],
        **counts,
        "perfect": counts["fp"] == 0 and counts["fn"] == 0,
    }


def self_test() -> dict[str, Any]:
    patterns = _training_patterns()
    prompts = _prompts(patterns)
    if patterns != [
        {"E": False, "S": False, "branch": False, "count": 32},
        {"E": False, "S": True, "branch": True, "count": 8},
        {"E": True, "S": False, "branch": True, "count": 2},
    ]:
        raise AssertionError(patterns)
    perfect = [
        candidate
        for candidate in CANDIDATES
        if all(
            _candidate_value(candidate, bool(row["E"]), bool(row["S"]))
            == bool(row["branch"])
            for row in patterns
        )
    ]
    if perfect != [EXPECTED]:
        raise AssertionError(perfect)
    width4 = _judge(EXPECTED, 4)
    if not width4["perfect"]:
        raise AssertionError(width4)
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "canonical_helix_commit": CANONICAL_HELIX_COMMIT,
        "candidate_count": len(CANDIDATES),
        "unique_width4_perfect_candidate": EXPECTED,
        "prompt_sha256": {
            name: hashlib.sha256(prompt.encode()).hexdigest()
            for name, prompt in prompts.items()
        },
        "heldout_computed": False,
    }


def evaluate(model_dir: str, model_file: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(model_file)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    patterns = _training_patterns()
    prompts = _prompts(patterns)
    surfaces = {
        name: {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            **_score_candidates(model_dir, prompt, CANDIDATES),
        }
        for name, prompt in prompts.items()
    }
    pmi_predictions = [surfaces[name]["pmi_prediction"] for name in sorted(surfaces)]
    interface_rescue = all(prediction == EXPECTED for prediction in pmi_predictions)

    selected = EXPECTED if interface_rescue else surfaces["plain"]["pmi_prediction"]
    exact_evaluations = {
        str(width): _judge(selected, width)
        for width in (4, *HELDOUT_WIDTHS)
    }
    finite_transfer = interface_rescue and all(
        row["perfect"] for row in exact_evaluations.values()
    )

    return {
        "schema_version": RUN_VERSION,
        "status": "completed-public-compiler-choice-interface",
        "actor": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_file_sha256": observed,
            "model_bytes": path.stat().st_size,
            "dtype": "float32",
            "device": "cpu",
            "weights_frozen": True,
        },
        "compiler": {
            "canonical_helix_commit": CANONICAL_HELIX_COMMIT,
            "source_function": "compile_rule30_branch_mechanism_task",
            "training_patterns": patterns,
            "basis_operators": compile_rule30_branch_mechanism_task()["program"]["basis_operators"],
        },
        "interface": {
            "candidates": list(CANDIDATES),
            "expected_shortest_candidate": EXPECTED,
            "surfaces": surfaces,
            "primary_scoring": "PMI-style mean token log-likelihood minus neutral Formula: prior",
        },
        "selected_for_exact_judgment": selected,
        "exact_evaluations": exact_evaluations,
        "decisions": {
            "interface_rescue_earned": interface_rescue,
            "finite_transfer_earned": finite_transfer,
            "cross_surface_pmi_agreement": len(set(pmi_predictions)) == 1,
        },
        "comparison": {
            "run054_same_actor": True,
            "run054_same_compiled_training_patterns": True,
            "run054_free_form_generation_valid": False,
            "run055_generate_calls": 0,
        },
        "claim_boundary": [
            "This tests bounded candidate recognition/selection, not independent formula generation.",
            "The candidate set was frozen before scoring and OR(E,S) is the unique width-4-perfect member of that set.",
            "OR and XOR are extensionally indistinguishable on the tested basin domain because E and S never overlap there; XOR is deliberately not used as a false negative control.",
            "Finite success does not establish an all-dyadic theorem or authorize model promotion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not all((args.model_dir, args.model_file, args.expected_sha256, args.output)):
        raise SystemExit("model-dir, model-file, expected-sha256, and output are required")

    result = evaluate(args.model_dir, args.model_file, args.expected_sha256)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
