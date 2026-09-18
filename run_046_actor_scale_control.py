"""Run 046: paired actor-scale control for exact representation operators.

This reuses the exact Run 039 evaluator and task seeds while changing only the
frozen SmolLM2 base actor from 360M to 1.7B. The retained 360M Run 039 receipt is
the comparator; this runner executes only the 1.7B side.

Model bytes are remote execution dependencies, not repository artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_039_exact_representation_operators import evaluate, self_test


RUN_VERSION = "helix-actor-scale-control-001.0"
BASELINE_RUN = "039-exact-representation-operators"
DEFAULT_SEEDS = "20261019,20261020,20261021"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def wrapper_self_test() -> dict[str, Any]:
    substrate = self_test(seed=20261019, cases_per_family=4)
    return {
        "schema_version": RUN_VERSION,
        "paired_baseline_run": BASELINE_RUN,
        "evaluator": "run_039_exact_representation_operators.evaluate",
        "substrate_self_test": substrate,
        "actor_scale_is_only_intended_experimental_change": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--actor-model", default="HuggingFaceTB/SmolLM2-1.7B")
    parser.add_argument("--actor-revision", required=False)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--output", default="run-046-actor-scale-control.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(wrapper_self_test(), indent=2, sort_keys=True))
        return

    required = {
        "model-dir": args.model_dir,
        "model-file": args.model_file,
        "expected-sha256": args.expected_sha256,
        "actor-revision": args.actor_revision,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(missing))

    model_path = Path(args.model_file)
    observed = sha256_file(model_path)
    if observed != args.expected_sha256:
        raise SystemExit(
            f"model digest mismatch: expected {args.expected_sha256}, observed {observed}"
        )

    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = evaluate(args.model_dir, seeds, args.cases_per_family)
    result["schema_version"] = RUN_VERSION
    result["scale_control"] = {
        "paired_baseline_run": BASELINE_RUN,
        "experimental_change": "actor scale within SmolLM2 base family",
        "baseline_actor": "HuggingFaceTB/SmolLM2-360M",
        "scale_probe_actor": args.actor_model,
        "same_run_039_evaluator": True,
        "baseline_rerun": False,
    }
    result["actor"] = {
        "model": args.actor_model,
        "revision": args.actor_revision,
        "model_file_sha256": observed,
        "model_bytes": model_path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
        "weights_frozen": True,
    }
    result["claim_boundary"] = (
        "Public paired scale control only. The retained Run 039 360M receipt is the "
        "comparator and is not rerun. Candidate coverage is not selected-answer "
        "accuracy. This run cannot authorize model promotion or frontier equivalence."
    )

    output = Path(args.output)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "actor": result["actor"],
                "scale_control": result["scale_control"],
                "arms": result["arms"],
                "coverage": result["coverage"],
                "representation_frontier": result["representation_frontier"],
                "resources": result["resources"],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
