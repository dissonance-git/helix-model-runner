"""Run an alternate same-scale actor on the Run 021 semantic-content battery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from offline_reasoning_content_probe import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--actor-revision", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="actor-control.json")
    args = parser.parse_args()

    path = Path(args.model_file)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(args.model_dir, args.seed, args.cases_per_family)
    result["actor"] = {
        "model": args.actor_model,
        "revision": args.actor_revision,
        "model_file_sha256": observed,
        "model_bytes": path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    result["comparison_role"] = (
        "same-parameter-scale actor control on the identical frozen public battery; "
        "not a promotion candidate by implication"
    )
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "source_battery_sha256": result["source_battery_sha256"],
        "actor": result["actor"],
        "family_results": result["family_results"],
        "overall_diagnostic": result["overall_diagnostic"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
