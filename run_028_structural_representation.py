"""Run 028: expand the frozen actor's reasoning coordinates structurally.

Each new surface is compiled only from learner-visible task text. The compiler
never receives the correct index or hidden generator operation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from offline_reasoning_battery import ChoiceScorer, build_battery, sha256_json
from offline_reasoning_content_probe import choose_content, surface_task_and_choices

RUN_VERSION = "helix-structural-representation-expansion-001.0"


def render(stem: str, choices: list[str]) -> str:
    block = "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(choices))
    return f"{stem}\nChoices:\n{block}\nAnswer:"


def task_stem(prompt: str) -> str:
    stem, _choices = surface_task_and_choices(prompt)
    suffix = "\nAnswer value:"
    if not stem.endswith(suffix):
        raise ValueError("unexpected task stem")
    return stem[:-len(suffix)]


def grid_cells(compact: str) -> str:
    rows = compact.split("/")
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"malformed compact grid: {compact}")
    cells = []
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            cells.append(f"r{r}c{c}={value}")
    return "{" + ", ".join(cells) + "}"


def relation_surface(surfaces: dict[str, str]) -> str:
    stem = task_stem(surfaces["compact"])
    prefix = "N=north S=south E=east W=west. "
    if not stem.startswith(prefix):
        raise ValueError("relation compact prefix changed")
    body = stem[len(prefix):]
    parts = body.split("; ")
    facts, query = parts[:-1], parts[-1]
    vectors = {"N": "(0,+1)", "S": "(0,-1)", "E": "(+1,0)", "W": "(-1,0)"}
    equations = []
    for fact in facts:
        left, direction, right = fact.split()
        equations.append(f"pos({left})-pos({right})={vectors[direction]}")
    if " ? " not in query:
        raise ValueError("relation query changed")
    left, right = query.split(" ? ", 1)
    _prose_stem, choices = surface_task_and_choices(surfaces["prose"])
    return render(
        "Use integer 2D coordinates for the same spatial facts. "
        + "; ".join(equations)
        + f". Add the displacement equations along the chain. "
          f"What compass direction is pos({left})-pos({right})?",
        choices,
    )


def abstract_surface(surfaces: dict[str, str]) -> str:
    stem = task_stem(surfaces["compact"])
    if not stem.startswith("Infer T: ") or " ; Q=" not in stem:
        raise ValueError("abstract compact shape changed")
    demos_text, query = stem[len("Infer T: "):].split(" ; Q=", 1)
    if not query.endswith(" ; T(Q)=?"):
        raise ValueError("abstract query shape changed")
    query = query[:-len(" ; T(Q)=?")]
    demos = []
    for item in demos_text.split(" ; "):
        left, right = item.split("->", 1)
        demos.append(
            f"input {grid_cells(left)} becomes output {grid_cells(right)}"
        )
    _compact_stem, choices = surface_task_and_choices(surfaces["compact"])
    structural_choices = [grid_cells(choice) for choice in choices]
    return render(
        "The same grid task is written as coordinate-indexed cell values. "
        "Infer the unchanged transformation from: "
        + "; ".join(demos)
        + f". Apply it to query {grid_cells(query)}.",
        structural_choices,
    )


def bit_tuples(a: str, b: str, c: str | None = None) -> str:
    names = "pqrst"
    rows = []
    for name, av, bv, idx in zip(names, a, b, range(5), strict=True):
        out = "?" if c is None else c[idx]
        rows.append(f"{name}:({av},{bv})->{out}")
    return " | ".join(rows)


def matrix_surface(surfaces: dict[str, str]) -> str:
    stem = task_stem(surfaces["bits"])
    marker = "Infer one fixed set operation from the examples: "
    if not stem.startswith(marker):
        raise ValueError("matrix bits prefix changed")
    parts = stem[len(marker):].split(" ; ")
    examples, query = parts[:-1], parts[-1]
    rendered = []
    for item in examples:
        match = re.fullmatch(r"([01]{5}) \? ([01]{5}) = ([01]{5})", item)
        if not match:
            raise ValueError(f"matrix example changed: {item}")
        rendered.append(bit_tuples(*match.groups()))
    match = re.fullmatch(r"([01]{5}) \? ([01]{5}) =", query)
    if not match:
        raise ValueError(f"matrix query changed: {query}")
    _bits_stem, choices = surface_task_and_choices(surfaces["bits"])
    names = "pqrst"
    structural_choices = [
        " ".join(f"{name}={bit}" for name, bit in zip(names, choice, strict=True))
        for choice in choices
    ]
    return render(
        "The same unknown set operation is written independently at each element position. "
        "Infer one fixed rule from the example tuples: "
        + "; ".join(rendered)
        + ". Query tuples: "
        + bit_tuples(match.group(1), match.group(2), None),
        structural_choices,
    )


def planning_surface(surfaces: dict[str, str]) -> str:
    stem = task_stem(surfaces["grid"])
    match = re.search(
        r"Grid rows top-to-bottom: ([SG#./]+)\. Choose the (\d+)-move route\.$",
        stem,
    )
    if not match:
        raise ValueError("planning grid shape changed")
    rows = match.group(1).split("/")
    moves = int(match.group(2))
    n = len(rows)
    start = goal = None
    blocked = set()
    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            if char == "S":
                start = (x, y)
            elif char == "G":
                goal = (x, y)
            elif char == "#":
                blocked.add((x, y))
    if start is None or goal is None:
        raise ValueError("planning grid lacks S/G")
    directions = {"U": (0, -1), "R": (1, 0), "D": (0, 1), "L": (-1, 0)}
    edges = []
    for y in range(n):
        for x in range(n):
            if (x, y) in blocked:
                continue
            for action, (dx, dy) in directions.items():
                q = (x + dx, y + dy)
                if 0 <= q[0] < n and 0 <= q[1] < n and q not in blocked:
                    edges.append(f"({x},{y})-{action}->({q[0]},{q[1]})")
    _grid_stem, choices = surface_task_and_choices(surfaces["grid"])
    return render(
        f"The same board is compiled into its complete legal-move graph. "
        f"Start={start}; goal={goal}; choose exactly {moves} moves. "
        "Legal directed edges: "
        + "; ".join(edges),
        choices,
    )


def stack_surface(surfaces: dict[str, str]) -> str:
    stem = task_stem(surfaces["compact"])
    match = re.fullmatch(
        r"pairs ka/ti mo/re su/va; LIFO; prefix=(.*); completion=\?",
        stem,
    )
    if not match:
        raise ValueError("stack compact shape changed")
    token_map = {
        "ka": "(", "ti": ")",
        "mo": "[", "re": "]",
        "su": "{", "va": "}",
    }

    def translate(text: str) -> str:
        tokens = [tok for tok in text.split() if tok]
        return "".join(token_map[tok] for tok in tokens)

    prefix = translate(match.group(1))
    _compact_stem, choices = surface_task_and_choices(surfaces["compact"])
    structural_choices = [translate(choice) for choice in choices]
    return render(
        "The same explicitly given token pairs are translated into ordinary bracket notation: "
        "ka/ti=(), mo/re=[], su/va={}. "
        f"Prefix={prefix}. Choose the completion that makes the whole bracket string balanced.",
        structural_choices,
    )


TRANSFORMS = {
    "relational-composition": relation_surface,
    "abstract-transformation": abstract_surface,
    "relational-matrix": matrix_surface,
    "grounded-planning": planning_surface,
    "stack-language": stack_surface,
}


def structural_surface(family: str, surfaces: dict[str, str]) -> str:
    # Deliberately receives no correct index or hidden generator state.
    return TRANSFORMS[family](surfaces)


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    by_seed: dict[int, list[bool]] = defaultdict(list)
    hits = 0
    for row in rows:
        ok = row[key] == row["correct_index"]
        hits += int(ok)
        by_family[row["family"]].append(ok)
        by_difficulty[row["difficulty"]].append(ok)
        by_seed[row["seed"]].append(ok)
    return {
        "cases": len(rows),
        "correct": hits,
        "accuracy": hits / len(rows),
        "by_family": {k: sum(v)/len(v) for k,v in sorted(by_family.items())},
        "by_difficulty": {k: sum(v)/len(v) for k,v in sorted(by_difficulty.items())},
        "by_seed": {str(k): sum(v)/len(v) for k,v in sorted(by_seed.items())},
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    scorer = ChoiceScorer(model_dir)
    started = time.time()
    rows = []
    public_specs = []

    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        public_specs.append({
            "seed": seed,
            "cases": [
                {
                    "case_id": c["case_id"],
                    "family": c["family"],
                    "difficulty": c["difficulty"],
                    "correct_index": c["correct_index"],
                    "surface_sha256": {
                        name: hashlib.sha256(prompt.encode()).hexdigest()
                        for name, prompt in c["surfaces"].items()
                    },
                }
                for c in cases
            ],
        })
        for case in cases:
            names = sorted(case["surfaces"])
            primary_name, secondary_name = names
            structural = structural_surface(case["family"], dict(case["surfaces"]))
            scored_primary = choose_content(scorer, case["surfaces"][primary_name])
            scored_secondary = choose_content(scorer, case["surfaces"][secondary_name])
            scored_structural = choose_content(scorer, structural)
            rows.append({
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": case["correct_index"],
                "primary_surface": primary_name,
                "secondary_surface": secondary_name,
                "primary_prediction": scored_primary["pmi_prediction"],
                "secondary_prediction": scored_secondary["pmi_prediction"],
                "structural_prediction": scored_structural["pmi_prediction"],
                "primary_margin": scored_primary["pmi_margin"],
                "secondary_margin": scored_secondary["pmi_margin"],
                "structural_margin": scored_structural["pmi_margin"],
                "structural_prompt_sha256": hashlib.sha256(structural.encode()).hexdigest(),
            })

    existing_oracle = sum(
        (r["primary_prediction"] == r["correct_index"])
        or (r["secondary_prediction"] == r["correct_index"])
        for r in rows
    )
    three_oracle = sum(
        (r["primary_prediction"] == r["correct_index"])
        or (r["secondary_prediction"] == r["correct_index"])
        or (r["structural_prediction"] == r["correct_index"])
        for r in rows
    )
    structural_only = [
        r for r in rows
        if r["structural_prediction"] == r["correct_index"]
        and r["primary_prediction"] != r["correct_index"]
        and r["secondary_prediction"] != r["correct_index"]
    ]

    incremental_by_family: dict[str, int] = defaultdict(int)
    for row in structural_only:
        incremental_by_family[row["family"]] += 1

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-structural-representation-calibration",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "source_battery_sha256": sha256_json(public_specs),
        "construction_boundary": (
            "Structural surfaces are deterministic functions of learner-visible task text. "
            "Construction functions receive family plus visible surfaces only, never correct_index "
            "or latent generator operation."
        ),
        "arms": {
            "existing-primary-surface": summarize(rows, "primary_prediction"),
            "existing-secondary-surface": summarize(rows, "secondary_prediction"),
            "new-structural-surface": summarize(rows, "structural_prediction"),
        },
        "coverage": {
            "existing_two_surface_oracle_correct": existing_oracle,
            "existing_two_surface_oracle": existing_oracle / len(rows),
            "three_surface_oracle_correct": three_oracle,
            "three_surface_oracle": three_oracle / len(rows),
            "structural_only_correct": len(structural_only),
            "structural_only_fraction": len(structural_only) / len(rows),
            "structural_only_by_family": dict(sorted(incremental_by_family.items())),
        },
        "rows": rows,
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "Fresh public procedural calibration only. Oracle coverage is posthoc diagnostic. "
            "No selector receives test labels. This run cannot authorize promotion, frontier "
            "equivalence, recursive improvement, or a capacity-wall claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seeds", default="20260921,20260922,20260923")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-028-structural.json")
    args = parser.parse_args()

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(
        args.model_dir,
        [int(v) for v in args.seeds.split(",") if v.strip()],
        args.cases_per_family,
    )
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
        "arms": result["arms"],
        "coverage": result["coverage"],
        "elapsed_seconds": result["elapsed_seconds"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
