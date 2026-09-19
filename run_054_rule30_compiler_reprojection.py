from __future__ import annotations

import argparse
import hashlib
import json
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

RUN_VERSION = "helix-rule30-compiler-reprojection-001.0"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_REVISION = "028493fd3c93bfb0536d0b07a124d8e302e187dd"
EXPECTED_PROMPT_SHA256 = "9e6f37cd3f411f888186f0f772e46e29fa71b17de8494d0c4ae7c0bc1f2cc070"
EXPECTED_PROMPT_BYTES = 669
HELDOUT_WIDTHS = (8, 16, 32)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("actor output contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("actor JSON must be an object")
    return value


def parse_formula_node(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 8:
        raise ValueError("formula nesting exceeds limit")
    if not isinstance(value, dict):
        raise ValueError("formula node must be an object")
    if set(value) != {"op", "args"}:
        raise ValueError("formula node must contain exactly op and args")
    op = str(value["op"]).strip().lower()
    args = value["args"]
    if not isinstance(args, list):
        raise ValueError("formula args must be a list")
    if op == "var":
        if len(args) != 1 or args[0] not in {"E", "S"}:
            raise ValueError("var args must be exactly ['E'] or ['S']")
        return {"op": "var", "args": [args[0]]}
    if op == "not":
        if len(args) != 1:
            raise ValueError("not requires one expression")
        return {"op": "not", "args": [parse_formula_node(args[0], depth + 1)]}
    if op in {"and", "or"}:
        if len(args) != 2:
            raise ValueError(f"{op} requires two expressions")
        return {
            "op": op,
            "args": [
                parse_formula_node(args[0], depth + 1),
                parse_formula_node(args[1], depth + 1),
            ],
        }
    raise ValueError(f"unknown formula operator: {op}")


def parse_actor_formula(text: str) -> dict[str, Any]:
    value = extract_json_object(text)
    if set(value) != {"formula"}:
        raise ValueError("actor output must contain exactly the top-level formula key")
    return parse_formula_node(value["formula"])


def formula_size(node: dict[str, Any]) -> int:
    op = node["op"]
    if op == "var":
        return 1
    return 1 + sum(formula_size(child) for child in node["args"])


def evaluate_formula(node: dict[str, Any], *, exceptional: bool, derivative_chain: bool) -> bool:
    op = node["op"]
    if op == "var":
        return exceptional if node["args"][0] == "E" else derivative_chain
    if op == "not":
        return not evaluate_formula(node["args"][0], exceptional=exceptional, derivative_chain=derivative_chain)
    left = evaluate_formula(node["args"][0], exceptional=exceptional, derivative_chain=derivative_chain)
    right = evaluate_formula(node["args"][1], exceptional=exceptional, derivative_chain=derivative_chain)
    return left and right if op == "and" else left or right


def actor_generate(model_dir: str, prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    config_limit = int(getattr(model.config, "max_position_embeddings", 8192) or 8192)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", config_limit) or config_limit)
    context_limit = min(config_limit, tokenizer_limit) if tokenizer_limit < 1_000_000 else config_limit
    if prompt_tokens + max_new_tokens > context_limit:
        raise RuntimeError(
            f"prompt plus generation exceeds context: {prompt_tokens}+{max_new_tokens}>{context_limit}"
        )

    started = time.time()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - started
    new_tokens = generated[0, prompt_tokens:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(new_tokens.shape[-1]),
        "generation_seconds": elapsed,
        "context_limit": context_limit,
    }


def judge_width(formula: dict[str, Any], width: int) -> dict[str, Any]:
    mechanism = compile_rule30_branch_mechanism(width)["mechanism"]
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for row in mechanism["rows"]:
        predicted = evaluate_formula(
            formula,
            exceptional=bool(row["exceptional_pair"]),
            derivative_chain=bool(row["derivative_chain"]),
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


def minimum_training_formula_size() -> int:
    training = compile_rule30_branch_mechanism_task()["task"]["training_patterns"]
    target = tuple(bool(row["branch"]) for row in training)
    assignments = [(bool(row["E"]), bool(row["S"])) for row in training]

    truths_by_size: dict[int, set[tuple[bool, ...]]] = {
        1: {
            tuple(e for e, _s in assignments),
            tuple(s for _e, s in assignments),
        }
    }
    if target in truths_by_size[1]:
        return 1

    for size in range(2, 10):
        current: set[tuple[bool, ...]] = set()
        for prior in truths_by_size.get(size - 1, set()):
            current.add(tuple(not bit for bit in prior))
        for left_size in range(1, size - 1):
            right_size = size - 1 - left_size
            for left in truths_by_size.get(left_size, set()):
                for right in truths_by_size.get(right_size, set()):
                    current.add(tuple(a and b for a, b in zip(left, right)))
                    current.add(tuple(a or b for a, b in zip(left, right)))
        truths_by_size[size] = current
        if target in current:
            return size
    raise AssertionError("training target absent from bounded grammar")


def self_test() -> dict[str, Any]:
    compiled = compile_rule30_branch_mechanism_task()
    task = compiled["task"]
    if task["actor_prompt_sha256"] != EXPECTED_PROMPT_SHA256:
        raise AssertionError(task["actor_prompt_sha256"])
    if task["actor_prompt_bytes"] != EXPECTED_PROMPT_BYTES:
        raise AssertionError(task["actor_prompt_bytes"])
    synthetic = json.dumps({
        "formula": {
            "op": "or",
            "args": [
                {"op": "var", "args": ["E"]},
                {"op": "var", "args": ["S"]},
            ],
        }
    })
    formula = parse_actor_formula(synthetic)
    width4 = judge_width(formula, 4)
    if not width4["perfect"]:
        raise AssertionError(width4)
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "canonical_helix_commit": CANONICAL_HELIX_COMMIT,
        "compiler_prompt_sha256": task["actor_prompt_sha256"],
        "compiler_prompt_bytes": task["actor_prompt_bytes"],
        "compiler_basis_operators": compiled["program"]["basis_operators"],
        "synthetic_formula_size": formula_size(formula),
        "width4_perfect": True,
        "heldout_computed": False,
    }


def evaluate(
    model_dir: str,
    model_file: str,
    expected_sha256: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    model_path = Path(model_file)
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    compiled = compile_rule30_branch_mechanism_task()
    task = compiled["task"]
    prompt = task["actor_prompt"]
    if task["actor_prompt_sha256"] != EXPECTED_PROMPT_SHA256:
        raise AssertionError("compiler prompt hash changed after preregistration")
    if task["actor_prompt_bytes"] != EXPECTED_PROMPT_BYTES:
        raise AssertionError("compiler prompt byte count changed after preregistration")

    actor_text, generation = actor_generate(model_dir, prompt, max_new_tokens)
    actor_output_sha256 = hashlib.sha256(actor_text.encode()).hexdigest()

    formula = None
    parse_error = None
    try:
        formula = parse_actor_formula(actor_text)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    evaluations: dict[str, Any] = {}
    minimum_size = minimum_training_formula_size()
    if formula is not None:
        evaluations["4"] = judge_width(formula, 4)
        # Hidden exact widths are computed only after actor output is frozen above.
        for width in HELDOUT_WIDTHS:
            evaluations[str(width)] = judge_width(formula, width)

    structured = formula is not None
    train_perfect = bool(evaluations.get("4", {}).get("perfect"))
    heldout_perfect = structured and all(
        evaluations.get(str(width), {}).get("perfect", False)
        for width in HELDOUT_WIDTHS
    )
    actor_size = formula_size(formula) if formula is not None else None
    minimum_tied = structured and train_perfect and actor_size == minimum_size

    return {
        "schema_version": RUN_VERSION,
        "status": "completed-public-compiler-reprojection",
        "actor": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_file_sha256": observed,
            "model_bytes": model_path.stat().st_size,
            "dtype": "float32",
            "device": "cpu",
            "weights_frozen": True,
        },
        "compiler": {
            "canonical_helix_commit": CANONICAL_HELIX_COMMIT,
            "basis_operators": compiled["program"]["basis_operators"],
            "operators": compiled["program"]["operators"],
            "task_prompt_sha256": task["actor_prompt_sha256"],
            "task_prompt_bytes": task["actor_prompt_bytes"],
            "training_patterns": task["training_patterns"],
            "heldout_widths": list(HELDOUT_WIDTHS),
        },
        "comparison": {
            "run053_same_actor": True,
            "run053_prompt_bytes": 2054,
            "run053_feature_count": 70,
            "run053_actor_valid_json": False,
            "run054_prompt_bytes": task["actor_prompt_bytes"],
            "run054_coordinate_count": 2,
        },
        "actor_output_sha256": actor_output_sha256,
        "actor_parse_error": parse_error,
        "actor_formula": formula,
        "actor_formula_size": actor_size,
        "minimum_training_formula_size": minimum_size,
        "generation": generation,
        "evaluations": evaluations,
        "decisions": {
            "structured_generation_rescue_earned": structured,
            "finite_transfer_rescue_earned": bool(train_perfect and heldout_perfect),
            "minimum_size_mechanism_recovery_earned": bool(minimum_tied and heldout_perfect),
        },
        "claim_boundary": [
            "The E/S coordinates were selected from prior exact Run 053 control evidence; this is a representation-access test, not independent coordinate discovery.",
            "Widths 8, 16, and 32 were computed only after the actor output was frozen.",
            "A successful finite-transfer result does not establish the all-dyadic theorem.",
            "No model promotion follows.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return

    required = (args.model_dir, args.model_file, args.expected_sha256, args.output)
    if not all(required):
        raise SystemExit("model-dir, model-file, expected-sha256, and output are required")
    result = evaluate(
        args.model_dir,
        args.model_file,
        args.expected_sha256,
        args.max_new_tokens,
    )
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
