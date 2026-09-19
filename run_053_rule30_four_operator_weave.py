"""Run 053: Rule 30 four-operator weave.

A frozen open-weight proposal actor sees only the exact width-4 zero-tail-basin
branch labels and a typed DUALIZE -> FACTOR -> LIFT -> PROJECT research route.
Only after actor output is frozen are widths 8, 16, and 32 evaluated exactly.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

TRAIN_WIDTH = 4
HELDOUT_WIDTHS = (8, 16, 32)
MAX_CANDIDATES = 3
MAX_CLAUSES = 4
MAX_LITERALS_PER_CLAUSE = 6
MODEL_FREE_MAX_CLAUSE_LITERALS = 3
MODEL_FREE_MAX_DNF_CLAUSES = 2
RUN_VERSION = "helix-rule30-four-operator-weave-001.0"

PROGRAM = [
    {
        "operator": "DUALIZE",
        "source": "forward-rule30-history",
        "target": "reverse-zero-tail-ancestry",
        "meaning": "Reverse the question: characterize exact predecessors that can reach the all-zero tail.",
    },
    {
        "operator": "FACTOR",
        "source": "reverse-zero-tail-ancestry",
        "target": "branch-core-plus-forced-exterior",
        "meaning": "Separate states with two legal zero-basin predecessors from states with one forced predecessor.",
    },
    {
        "operator": "LIFT",
        "source": "branch-core-plus-forced-exterior",
        "target": "bounded-algebraic-feature-space",
        "meaning": "Expose periods, complements, cyclic derivatives, and exact word relations as Boolean features.",
    },
    {
        "operator": "PROJECT",
        "source": "bounded-algebraic-feature-space",
        "target": "small-boolean-dnf-classifier",
        "meaning": "Compress the branch predicate into a short DNF over the declared feature names.",
    },
]


def shift(word: int, n: int, q: int) -> int:
    q %= n
    mask = (1 << n) - 1
    return word if q == 0 else ((word << q) | (word >> (n - q))) & mask


def word_period(word: int, n: int) -> int:
    for p in range(1, n + 1):
        if n % p == 0 and shift(word, n, p) == word:
            return p
    return n


def forcing(a: int, b: int, c: int, d: int, n: int) -> int:
    return (~(a ^ b) & ((1 << n) - 1)) & (c | d)


def derivative(word: int, n: int) -> int:
    return word ^ shift(word, n, 1)


def inverse_derivative(target: int, n: int) -> tuple[int, ...]:
    if target.bit_count() & 1:
        return ()
    mask = (1 << n) - 1
    bits = [(target >> i) & 1 for i in range(n)]
    preimage_bits = [0] * n
    for i in range(1, n):
        preimage_bits[i] = preimage_bits[i - 1] ^ bits[i]
    preimage = sum(bit << i for i, bit in enumerate(preimage_bits))
    if derivative(preimage, n) != target:
        raise AssertionError("cyclic derivative reconstruction failed")
    return preimage, preimage ^ mask


def zero_tail_basin(n: int) -> tuple[dict[tuple[int, ...], int], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    zero = (0, 0, 0, 0)
    queue = deque([zero])
    depth = {zero: 0}
    edges: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    while queue:
        b, c, d, e = state = queue.popleft()
        for a in inverse_derivative(forcing(b, c, d, e, n), n):
            predecessor = (a, b, c, d)
            edges.append((predecessor, state))
            if predecessor not in depth:
                depth[predecessor] = depth[state] + 1
                queue.append(predecessor)
    return depth, edges


def state_features(state: tuple[int, int, int, int], n: int) -> dict[str, bool]:
    a, b, c, d = state
    mask = (1 << n) - 1
    h = forcing(a, b, c, d, n)
    words = {"a": a, "b": b, "c": c, "d": d, "h": h}
    features: dict[str, bool] = {}
    for name, word in words.items():
        period = word_period(word, n)
        for bound in (1, 2, 4):
            features[f"{name}.period<={bound}"] = period <= bound
        features[f"{name}.zero"] = word == 0
        features[f"{name}.ones"] = word == mask
        features[f"{name}.even-weight"] = word.bit_count() % 2 == 0
        if n % 2 == 0:
            features[f"{name}.half-repeat"] = shift(word, n, n // 2) == word
            features[f"{name}.half-complement"] = shift(word, n, n // 2) == (word ^ mask)
    names = tuple(words)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            features[f"{left}={right}"] = words[left] == words[right]
            features[f"{left}=~{right}"] = (words[left] ^ words[right]) == mask
    for name, word in words.items():
        dword = derivative(word, n)
        for right, right_word in words.items():
            features[f"D({name})={right}"] = dword == right_word
    return features


@dataclass(frozen=True)
class Basin:
    width: int
    states: tuple[tuple[int, int, int, int], ...]
    branch: frozenset[tuple[int, int, int, int]]


def build_basin(n: int) -> Basin:
    depth, edges = zero_tail_basin(n)
    outdegree = Counter(source for source, _target in edges)
    branch = frozenset(state for state, degree in outdegree.items() if degree > 1)
    return Basin(n, tuple(sorted(depth)), branch)


def width4_training() -> tuple[Basin, list[str], list[dict[str, Any]]]:
    basin = build_basin(TRAIN_WIDTH)
    feature_rows = [(state, state_features(state, TRAIN_WIDTH)) for state in basin.states]
    feature_names = sorted(feature_rows[0][1])
    variable = [
        name for name in feature_names
        if len({features[name] for _state, features in feature_rows}) > 1
    ]
    rows = []
    for state, features in feature_rows:
        rows.append({
            "state": [format(word, f"0{TRAIN_WIDTH}b") for word in state],
            "label": "BRANCH" if state in basin.branch else "NONBRANCH",
            "true_features": [name for name in variable if features[name]],
        })
    return basin, variable, rows


def feature_alias(name: str) -> str:
    """Compact reversible notation for the declared feature language."""
    suffixes = {
        ".period<=1": "p1",
        ".period<=2": "p2",
        ".period<=4": "p4",
        ".zero": "z",
        ".ones": "o",
        ".even-weight": "e",
        ".half-repeat": "hr",
        ".half-complement": "hc",
    }
    for suffix, short in suffixes.items():
        if name.endswith(suffix):
            return name[:-len(suffix)] + short
    if name.startswith("D(") and ")=" in name:
        left, right = name[2:].split(")=", 1)
        return f"D{left}={right}"
    if "=~" in name:
        return name.replace("=~", "~")
    return name


def feature_aliases(variable_features: list[str]) -> dict[str, str]:
    aliases = {feature_alias(name): name for name in variable_features}
    if len(aliases) != len(variable_features):
        raise AssertionError("feature alias collision")
    return aliases


def build_prompt(variable_features: list[str], rows: list[dict[str, Any]]) -> str:
    # Exact transpose of the same width-4 labeled feature matrix with compact,
    # reversible semantic aliases. No feature truth value or label is removed.
    branch_rows = [row for row in rows if row["label"] == "BRANCH"]
    nonbranch_rows = [row for row in rows if row["label"] == "NONBRANCH"]
    aliases = feature_aliases(variable_features)
    full_to_alias = {full: alias for alias, full in aliases.items()}
    matrix = {}
    for name in variable_features:
        alias = full_to_alias[name]
        matrix[alias] = [
            [i for i, row in enumerate(branch_rows) if name in row["true_features"]],
            [i for i, row in enumerate(nonbranch_rows) if name in row["true_features"]],
        ]
    payload = {
        "task": "Find 1-3 short DNF formulas selecting every B row and no N row. Only width4 is shown; hidden widths 8/16/32 are tested unchanged.",
        "route": "DUALIZE>FACTOR>LIFT>PROJECT",
        "legend": "xp1/p2/p4=period(x)<=1/2/4;xz/xo/xe=zero/ones/even-weight;xhr/xhc=half-repeat/half-complement;x=y equality;x~y complement;Dx=y cyclic derivative(x)=y",
        "rows": {"B": len(branch_rows), "N": len(nonbranch_rows)},
        "truth": matrix,
        "truth_format": "alias:[B_true_row_ids,N_true_row_ids]; !alias is complement within each group",
        "output": {"candidates": [{"id": "x", "dnf": [["alias", "!alias"]]}]},
        "rules": "DNF=OR of AND clauses. Use aliases exactly. No row/state IDs in formulas. Prefer fewest literals. JSON only. No all-width claim.",
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("actor output contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("actor JSON must be an object")
    return value


def parse_candidates(text: str, alias_to_full: dict[str, str]) -> list[dict[str, Any]]:
    value = extract_json_object(text)
    raw_candidates = value.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("actor output requires non-empty candidates list")
    candidates: list[dict[str, Any]] = []
    for raw in raw_candidates[:MAX_CANDIDATES]:
        if not isinstance(raw, dict):
            raise ValueError("candidate must be an object")
        candidate_id = str(raw.get("id") or "").strip()
        dnf = raw.get("dnf")
        if not candidate_id or not isinstance(dnf, list) or not dnf or len(dnf) > MAX_CLAUSES:
            raise ValueError("candidate id/dnf invalid")
        clauses: list[list[str]] = []
        alias_clauses: list[list[str]] = []
        for clause in dnf:
            if not isinstance(clause, list) or not clause or len(clause) > MAX_LITERALS_PER_CLAUSE:
                raise ValueError("candidate clause invalid")
            literals: list[str] = []
            alias_literals: list[str] = []
            seen: set[str] = set()
            polarity: dict[str, bool] = {}
            for literal in clause:
                token = str(literal or "").strip()
                negated = token.startswith("!")
                alias = token[1:] if negated else token
                if alias not in alias_to_full:
                    raise ValueError(f"unknown feature alias: {token}")
                full = alias_to_full[alias]
                if alias in polarity and polarity[alias] != negated:
                    raise ValueError(f"self-contradictory clause literal: {alias}")
                normalized = ("!" if negated else "") + full
                if normalized in seen:
                    continue
                polarity[alias] = negated
                seen.add(normalized)
                alias_literals.append(token)
                literals.append(normalized)
            alias_clauses.append(alias_literals)
            clauses.append(literals)
        candidates.append({
            "id": candidate_id,
            "actor_alias_dnf": alias_clauses,
            "dnf": clauses,
            "mechanism": str(raw.get("mechanism") or "").strip(),
            "source": "actor",
        })
    return candidates


def classify(features: dict[str, bool], dnf: list[list[str]]) -> bool:
    for clause in dnf:
        if all((not features[literal[1:]]) if literal.startswith("!") else features[literal] for literal in clause):
            return True
    return False


def evaluate_many(candidates: list[dict[str, Any]], basin: Basin) -> dict[str, dict[str, Any]]:
    counts = {candidate["id"]: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for candidate in candidates}
    for state in basin.states:
        features = state_features(state, basin.width)
        actual = state in basin.branch
        for candidate in candidates:
            predicted = classify(features, candidate["dnf"])
            bucket = counts[candidate["id"]]
            if predicted and actual:
                bucket["tp"] += 1
            elif predicted and not actual:
                bucket["fp"] += 1
            elif not predicted and actual:
                bucket["fn"] += 1
            else:
                bucket["tn"] += 1
    out = {}
    for candidate in candidates:
        row = counts[candidate["id"]]
        out[candidate["id"]] = {
            "width": basin.width,
            "basin_states": len(basin.states),
            "actual_branch_states": len(basin.branch),
            **row,
            "perfect": row["fp"] == 0 and row["fn"] == 0,
        }
    return out


def model_free_width4_minimal(variable_features: list[str], basin: Basin) -> list[dict[str, Any]]:
    rows = [(state, state in basin.branch, state_features(state, basin.width)) for state in basin.states]
    positives = [i for i, (_state, label, _features) in enumerate(rows) if label]
    negatives = [i for i, (_state, label, _features) in enumerate(rows) if not label]
    positive_mask = sum(1 << i for i in positives)
    negative_mask = sum(1 << i for i in negatives)
    all_mask = (1 << len(rows)) - 1
    literals: list[tuple[str, int]] = []
    for name in variable_features:
        true_mask = sum(1 << i for i, (_state, _label, features) in enumerate(rows) if features[name])
        literals.append((name, true_mask))
        literals.append(("!" + name, all_mask ^ true_mask))
    pure_by_coverage: dict[int, tuple[str, ...]] = {}
    for size in range(1, MODEL_FREE_MAX_CLAUSE_LITERALS + 1):
        for indexes in itertools.combinations(range(len(literals)), size):
            clause_mask = all_mask
            polarity: dict[str, bool] = {}
            names: list[str] = []
            contradictory = False
            for index in indexes:
                literal, literal_mask = literals[index]
                negated = literal.startswith("!")
                base = literal[1:] if negated else literal
                if base in polarity and polarity[base] != negated:
                    contradictory = True
                    break
                polarity[base] = negated
                names.append(literal)
                clause_mask &= literal_mask
            if contradictory:
                continue
            positive_coverage = clause_mask & positive_mask
            if not positive_coverage or (clause_mask & negative_mask):
                continue
            names_tuple = tuple(names)
            prior = pure_by_coverage.get(positive_coverage)
            if prior is None or (len(names_tuple), names_tuple) < (len(prior), prior):
                pure_by_coverage[positive_coverage] = names_tuple
    coverages = sorted(pure_by_coverage)
    solutions: list[tuple[int, int, tuple[tuple[str, ...], ...]]] = []
    for clauses_count in range(1, MODEL_FREE_MAX_DNF_CLAUSES + 1):
        for selected in itertools.combinations(coverages, clauses_count):
            union = 0
            clauses: list[tuple[str, ...]] = []
            for coverage in selected:
                union |= coverage
                clauses.append(pure_by_coverage[coverage])
            if union != positive_mask:
                continue
            literal_count = sum(len(clause) for clause in clauses)
            solutions.append((literal_count, clauses_count, tuple(sorted(clauses))))
        if solutions:
            break
    if not solutions:
        return []
    best_literals = min(row[0] for row in solutions)
    best = sorted({row[2] for row in solutions if row[0] == best_literals})
    return [
        {
            "id": f"model-free-minimal-{i+1}",
            "dnf": [list(clause) for clause in clauses],
            "mechanism": "exact width-4 minimum under the bounded clause search",
            "source": "model-free-exact-search",
        }
        for i, clauses in enumerate(best)
    ]


def actor_generate(model_dir: str, prompt: str, max_new_tokens: int) -> tuple[str, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    config_limit = int(getattr(model.config, "max_position_embeddings", 8192) or 8192)
    tokenizer_limit = int(getattr(tokenizer, "model_max_length", config_limit) or config_limit)
    context_limit = min(config_limit, tokenizer_limit) if tokenizer_limit < 1_000_000 else config_limit
    if prompt_tokens + max_new_tokens > context_limit:
        raise RuntimeError(f"prompt plus generation exceeds context: {prompt_tokens}+{max_new_tokens}>{context_limit}")
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


def self_test() -> dict[str, Any]:
    basin, variable, rows = width4_training()
    if len(basin.states) != 42 or len(basin.branch) != 10:
        raise AssertionError((len(basin.states), len(basin.branch)))
    prompt = build_prompt(variable, rows)
    aliases = feature_aliases(variable)
    synthetic_alias = sorted(aliases)[0]
    synthetic = json.dumps({
        "candidates": [{"id": "smoke", "dnf": [[synthetic_alias]], "mechanism": "smoke"}]
    })
    parsed = parse_candidates(synthetic, aliases)
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "width4_states": len(basin.states),
        "width4_branch_states": len(basin.branch),
        "variable_features": len(variable),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_bytes": len(prompt.encode()),
        "parser_candidates": len(parsed),
        "heldout_computed": False,
    }


def evaluate(model_dir: str, model_file: str, expected_sha256: str, max_new_tokens: int) -> dict[str, Any]:
    model_path = Path(model_file)
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    train_basin, variable_features, train_rows = width4_training()
    aliases = feature_aliases(variable_features)
    prompt = build_prompt(variable_features, train_rows)
    actor_text, generation = actor_generate(model_dir, prompt, max_new_tokens)
    actor_output_sha256 = hashlib.sha256(actor_text.encode()).hexdigest()
    parse_error = None
    actor_candidates: list[dict[str, Any]] = []
    try:
        actor_candidates = parse_candidates(actor_text, aliases)
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    heldout_basins = {n: build_basin(n) for n in HELDOUT_WIDTHS}
    model_free = model_free_width4_minimal(variable_features, train_basin)
    all_candidates = actor_candidates + model_free
    evaluations: dict[str, dict[str, Any]] = {candidate["id"]: {} for candidate in all_candidates}
    if all_candidates:
        train_eval = evaluate_many(all_candidates, train_basin)
        for candidate in all_candidates:
            evaluations[candidate["id"]][str(TRAIN_WIDTH)] = train_eval[candidate["id"]]
        for n, basin in heldout_basins.items():
            width_eval = evaluate_many(all_candidates, basin)
            for candidate in all_candidates:
                evaluations[candidate["id"]][str(n)] = width_eval[candidate["id"]]

    def enriched(candidate: dict[str, Any]) -> dict[str, Any]:
        evaluation = evaluations[candidate["id"]]
        literals = sum(len(clause) for clause in candidate["dnf"])
        clauses = len(candidate["dnf"])
        train_perfect = evaluation.get(str(TRAIN_WIDTH), {}).get("perfect", False)
        heldout_all_perfect = all(evaluation.get(str(n), {}).get("perfect", False) for n in HELDOUT_WIDTHS)
        return {
            **candidate,
            "clauses": clauses,
            "literals": literals,
            "train_perfect": train_perfect,
            "heldout_all_perfect": heldout_all_perfect,
            "evaluations": evaluation,
        }

    actor_rows = [enriched(candidate) for candidate in actor_candidates]
    baseline_rows = [enriched(candidate) for candidate in model_free]
    actor_finite_transfer = any(row["train_perfect"] and row["heldout_all_perfect"] for row in actor_rows)
    actor_compact_transfer = any(
        row["train_perfect"] and row["heldout_all_perfect"] and row["clauses"] <= 2 and row["literals"] <= 4
        for row in actor_rows
    )
    baseline_transfer = any(row["train_perfect"] and row["heldout_all_perfect"] for row in baseline_rows)

    return {
        "schema_version": RUN_VERSION,
        "status": "completed-public-four-operator-weave",
        "actor": {
            "model": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "revision": "028493fd3c93bfb0536d0b07a124d8e302e187dd",
            "model_file_sha256": observed,
            "model_bytes": model_path.stat().st_size,
            "dtype": "float32",
            "device": "cpu",
            "weights_frozen": True,
        },
        "transform_program": PROGRAM,
        "blinding": {
            "actor_visible_widths": [TRAIN_WIDTH],
            "heldout_widths": list(HELDOUT_WIDTHS),
            "heldout_basin_computed_after_actor_response": True,
            "model_free_search_run_after_actor_response": True,
            "feature_pruning": "constant-across-width4 only; label-blind",
        },
        "training": {
            "width": TRAIN_WIDTH,
            "basin_states": len(train_basin.states),
            "branch_states": len(train_basin.branch),
            "variable_features": len(variable_features),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_bytes": len(prompt.encode()),
            "feature_alias_count": len(aliases),
            "feature_alias_map_sha256": hashlib.sha256(
                json.dumps(aliases, sort_keys=True).encode()
            ).hexdigest(),
        },
        "feature_aliases": aliases,
        "actor_output_sha256": actor_output_sha256,
        "actor_parse_error": parse_error,
        "actor_candidates": actor_rows,
        "model_free_minimal_candidates": baseline_rows,
        "decisions": {
            "actor_finite_transfer_earned": actor_finite_transfer,
            "actor_compact_rediscovery_earned": actor_compact_transfer,
            "model_free_task_solvability_confirmed": baseline_transfer,
        },
        "generation": generation,
        "claim_boundary": [
            "The open-weight actor is a proposal source, not the verifier.",
            "Only width 4 labels are visible to the actor.",
            "Widths 8, 16, and 32 are complete exact finite held-out checks computed after actor output is frozen.",
            "Model-free bounded search is a control and does not count as actor success.",
            "A held-out-perfect classifier is a finite candidate invariant, not an all-dyadic theorem.",
            "No Rule 30 center-column nonperiodicity or prize claim follows.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", default="run-053-rule30-four-operator-weave.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")
    result = evaluate(args.model_dir, args.model_file, args.expected_sha256, args.max_new_tokens)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "status": result["status"],
        "training": result["training"],
        "actor_parse_error": result["actor_parse_error"],
        "actor_candidates": [
            {
                "id": row["id"],
                "dnf": row["dnf"],
                "clauses": row["clauses"],
                "literals": row["literals"],
                "train_perfect": row["train_perfect"],
                "heldout_all_perfect": row["heldout_all_perfect"],
                "evaluations": row["evaluations"],
            }
            for row in result["actor_candidates"]
        ],
        "model_free_minimal_candidates": [
            {
                "id": row["id"],
                "dnf": row["dnf"],
                "clauses": row["clauses"],
                "literals": row["literals"],
                "train_perfect": row["train_perfect"],
                "heldout_all_perfect": row["heldout_all_perfect"],
            }
            for row in result["model_free_minimal_candidates"]
        ],
        "decisions": result["decisions"],
        "generation": result["generation"],
        "actor_output_sha256": result["actor_output_sha256"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
