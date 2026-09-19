from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_055_rule30_compiler_choice_interface import (
    CANDIDATES,
    EXPECTED,
    evaluate as evaluate_360_contract,
    self_test as contract_self_test,
)

RUN_VERSION = "helix-rule30-compiler-choice-scale-control-001.0"
MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_REVISION = "d1bb90bcfbe0f211109880f4da18da66f229c4f6"
MODEL_SHA256 = "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"


def self_test() -> dict:
    inherited = contract_self_test()
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "inherited_contract_status": inherited["status"],
        "canonical_helix_commit": inherited["canonical_helix_commit"],
        "candidate_count": inherited["candidate_count"],
        "expected_shortest_candidate": EXPECTED,
        "candidates": list(CANDIDATES),
        "experimental_change": "actor scale only: 360M-Instruct -> 1.7B-Instruct",
        "heldout_computed": False,
    }


def evaluate(model_dir: str, model_file: str, expected_sha256: str) -> dict:
    result = evaluate_360_contract(model_dir, model_file, expected_sha256)
    path = Path(model_file)
    result["schema_version"] = RUN_VERSION
    result["status"] = "completed-public-compiler-choice-scale-control"
    result["actor"] = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_file_sha256": expected_sha256,
        "model_bytes": path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
        "weights_frozen": True,
    }
    result["comparison"] = {
        "comparator_run": "055-rule30-compiler-choice-interface",
        "comparator_actor": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "experimental_change": "same-family instruct actor scale only",
        "same_compiler_input": True,
        "same_prompt_surfaces": True,
        "same_candidates": True,
        "same_scoring": True,
        "comparator_pmi_prediction_plain": "NOT(E)",
        "comparator_pmi_prediction_compact": "NOT(E)",
        "comparator_interface_rescue": False,
    }
    result["decisions"]["scale_rescue_earned"] = bool(
        result["decisions"]["interface_rescue_earned"]
    )
    result["claim_boundary"] = [
        "Public same-family scale control only.",
        "The exact Run 055 compiler input, prompts, candidates, and PMI scoring are reused unchanged.",
        "Success would localize this bounded task difference to actor scale/capability more strongly; it would not establish a universal parameter threshold.",
        "Finite transfer does not establish an all-dyadic theorem or authorize model promotion.",
    ]
    return result


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

    path = Path(args.model_file)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(args.model_dir, args.model_file, args.expected_sha256)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
