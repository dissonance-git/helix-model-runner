
"""Fresh offline reasoning battery for the frozen Helix Model reference substrate.

The tasks are procedurally generated and donor-inspired rather than copied benchmark
items. Results are public calibration evidence, not protected promotion evidence.

Donor ideas:
- EleutherAI lm-evaluation-harness: teacher-forced multiple-choice log-likelihood
  and mutual-information style label-prior normalization.
- ARC / ARC-GEN / ConceptARC: infer a transformation from demonstrations and
  apply it to a fresh grid.
- bAbI / CLUTRR: multi-hop relational composition.
- DeepMind PGM: infer a latent relation and transfer it to a held-out row.
- BabyAI: compact grounded planning.
- BIG-Bench Dyck-style tasks: stack / formal-language reasoning.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BATTERY_VERSION = "helix-offline-reasoning-001.0"
LABELS = ("A", "B", "C", "D")
DONORS = [
    {
        "name": "lm-evaluation-harness",
        "project": "EleutherAI/lm-evaluation-harness",
        "mechanism": "teacher-forced multiple-choice log-likelihood plus prior-normalized comparison",
    },
    {
        "name": "ARC / ARC-GEN / ConceptARC",
        "project": "fchollet/ARC-AGI; google/ARC-GEN; victorvikram/ConceptARC",
        "mechanism": "few-demonstration abstract transformation induction over fresh grids",
    },
    {
        "name": "bAbI / CLUTRR",
        "project": "facebookarchive/bAbI-tasks; facebookresearch/clutrr",
        "mechanism": "multi-hop relational composition under surface variation",
    },
    {
        "name": "PGM",
        "project": "google-deepmind/abstract-reasoning-matrices",
        "mechanism": "infer a row relation and transfer it to a held-out row",
    },
    {
        "name": "BabyAI",
        "project": "mila-iqia/babyai",
        "mechanism": "bounded grounded planning in a compact grid world",
    },
    {
        "name": "BIG-Bench",
        "project": "google/BIG-bench",
        "mechanism": "formal-language / stack reasoning with fresh symbol bindings",
    },
]


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _stable_rng(seed: int, *parts: Any) -> random.Random:
    material = "|".join([str(seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _balanced_options(rng: random.Random, correct: str, distractors: Iterable[str], correct_slot: int) -> tuple[list[str], int]:
    wrong = []
    for item in distractors:
        if item != correct and item not in wrong:
            wrong.append(item)
    if len(wrong) < 3:
        raise ValueError("need at least three unique distractors")
    rng.shuffle(wrong)
    options = wrong[:3]
    options.insert(correct_slot, correct)
    return options, correct_slot


def _format_mcq(stem: str, options: list[str], *, compact: bool) -> str:
    if compact:
        rendered = " | ".join(f"{LABELS[i]}={v}" for i, v in enumerate(options))
        return f"{stem}\n{rendered}\nAnswer:"
    rendered = "\n".join(f"{LABELS[i]}. {v}" for i, v in enumerate(options))
    return f"{stem}\nChoices:\n{rendered}\nAnswer:"


DIRS = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
}
RELATION_NAME = {
    (-1, -1): "southwest",
    (-1, 0): "west",
    (-1, 1): "northwest",
    (0, -1): "south",
    (0, 1): "north",
    (1, -1): "southeast",
    (1, 0): "east",
    (1, 1): "northeast",
}
REL_NAMES = tuple(RELATION_NAME.values())
NAMES = ("Ari", "Bex", "Cato", "Dara", "Eno", "Fia", "Galen", "Hira")


def _axis_sign(v: int) -> int:
    return (v > 0) - (v < 0)


def make_relation_case(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _stable_rng(seed, "relation", index)
    hops = 2 if difficulty == "easy" else 4
    while True:
        moves = [rng.choice(tuple(DIRS)) for _ in range(hops)]
        dx = sum(DIRS[m][0] for m in moves)
        dy = sum(DIRS[m][1] for m in moves)
        rel = RELATION_NAME.get((_axis_sign(dx), _axis_sign(dy)))
        if rel:
            break
    names = list(NAMES[: hops + 1])
    rng.shuffle(names)
    facts = [f"{names[i]} is {moves[i]} of {names[i+1]}." for i in range(hops)]
    compact_facts = [f"{names[i]} {moves[i][0].upper()} {names[i+1]}" for i in range(hops)]
    distractors = [x for x in REL_NAMES if x != rel]
    correct_slot = index % 4
    options, slot = _balanced_options(rng, rel, distractors, correct_slot)
    return {
        "family": "relational-composition",
        "difficulty": difficulty,
        "case_id": f"rel-{index:02d}",
        "options": options,
        "correct_index": slot,
        "surfaces": {
            "prose": _format_mcq(
                "Use the spatial facts transitively. "
                + " ".join(facts)
                + f" Where is {names[0]} relative to {names[-1]}?",
                options,
                compact=False,
            ),
            "compact": _format_mcq(
                "N=north S=south E=east W=west. "
                + "; ".join(compact_facts)
                + f"; {names[0]} ? {names[-1]}",
                options,
                compact=True,
            ),
        },
    }


def rot90(g: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in zip(*g[::-1]))


def flip_h(g: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(reversed(row)) for row in g)


def flip_v(g: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(reversed(g))


def transpose(g: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in zip(*g))


GRID_TRANSFORMS: dict[str, Callable[[tuple[tuple[int, ...], ...]], tuple[tuple[int, ...], ...]]] = {
    "rotate-90": rot90,
    "flip-horizontal": flip_h,
    "flip-vertical": flip_v,
    "transpose": transpose,
}


def _random_grid(rng: random.Random, n: int = 3) -> tuple[tuple[int, ...], ...]:
    while True:
        g = tuple(tuple(rng.randrange(3) for _ in range(n)) for _ in range(n))
        outs = {fn(g) for fn in GRID_TRANSFORMS.values()}
        if len(outs) == len(GRID_TRANSFORMS):
            return g


def _grid_compact(g: tuple[tuple[int, ...], ...]) -> str:
    return "/".join("".join(map(str, row)) for row in g)


def _grid_prose(g: tuple[tuple[int, ...], ...]) -> str:
    return "[" + "; ".join(" ".join(map(str, row)) for row in g) + "]"


def make_grid_case(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _stable_rng(seed, "grid", index)
    transform_names = list(GRID_TRANSFORMS)
    target_name = transform_names[index % len(transform_names)]
    fn = GRID_TRANSFORMS[target_name]
    demos = 1 if difficulty == "easy" else 2
    demo_pairs = []
    for _ in range(demos):
        g = _random_grid(rng)
        demo_pairs.append((g, fn(g)))
    query = _random_grid(rng)
    outputs = {name: tfn(query) for name, tfn in GRID_TRANSFORMS.items()}
    correct = _grid_compact(outputs[target_name])
    distractors = [_grid_compact(v) for name, v in outputs.items() if name != target_name]
    correct_slot = index % 4
    options, slot = _balanced_options(rng, correct, distractors, correct_slot)

    prose_demos = " ".join(
        f"Example {i+1}: input {_grid_prose(a)} becomes {_grid_prose(b)}."
        for i, (a, b) in enumerate(demo_pairs)
    )
    compact_demos = " ; ".join(
        f"{_grid_compact(a)}->{_grid_compact(b)}" for a, b in demo_pairs
    )
    prose_options = [
        _grid_prose(tuple(tuple(int(c) for c in row) for row in option.split("/")))
        for option in options
    ]
    return {
        "family": "abstract-transformation",
        "difficulty": difficulty,
        "case_id": f"grid-{index:02d}",
        "options": options,
        "correct_index": slot,
        "surfaces": {
            "prose": _format_mcq(
                f"Infer the same grid transformation from the examples. {prose_demos} "
                f"Apply it to {_grid_prose(query)}.",
                prose_options,
                compact=False,
            ),
            "compact": _format_mcq(
                f"Infer T: {compact_demos} ; Q={_grid_compact(query)} ; T(Q)=?",
                options,
                compact=True,
            ),
        },
    }


BIT_OPS = {
    "union": lambda a, b: a | b,
    "intersection": lambda a, b: a & b,
    "xor": lambda a, b: a ^ b,
    "left-minus-right": lambda a, b: a & (~b & 0b11111),
}


def _bits(v: int) -> str:
    return f"{v:05b}"


def _set(v: int) -> str:
    names = "pqrst"
    members = [names[i] for i in range(5) if v & (1 << (4 - i))]
    return "{" + ",".join(members) + "}"


def _unique_bit_problem(rng: random.Random) -> tuple[int, int, dict[str, int]]:
    while True:
        a = rng.randrange(1, 32)
        b = rng.randrange(1, 32)
        outputs = {name: fn(a, b) for name, fn in BIT_OPS.items()}
        if len(set(outputs.values())) == 4:
            return a, b, outputs


def make_matrix_case(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _stable_rng(seed, "matrix", index)
    op_names = list(BIT_OPS)
    target = op_names[index % len(op_names)]
    demo_count = 1 if difficulty == "easy" else 2
    demos = []
    for _ in range(demo_count):
        a, b, outputs = _unique_bit_problem(rng)
        demos.append((a, b, outputs[target]))
    qa, qb, qouts = _unique_bit_problem(rng)
    correct_value = qouts[target]
    correct_slot = index % 4
    bit_options, slot = _balanced_options(
        rng,
        _bits(correct_value),
        [_bits(v) for name, v in qouts.items() if name != target],
        correct_slot,
    )
    option_values = [int(v, 2) for v in bit_options]
    set_options = [_set(v) for v in option_values]

    bit_demos = " ; ".join(f"{_bits(a)} ? {_bits(b)} = {_bits(c)}" for a, b, c in demos)
    set_demos = " ".join(f"{_set(a)} with {_set(b)} gives {_set(c)}." for a, b, c in demos)
    return {
        "family": "relational-matrix",
        "difficulty": difficulty,
        "case_id": f"matrix-{index:02d}",
        "options": bit_options,
        "correct_index": slot,
        "surfaces": {
            "bits": _format_mcq(
                f"Infer one fixed set operation from the examples: {bit_demos} ; "
                f"{_bits(qa)} ? {_bits(qb)} =",
                bit_options,
                compact=True,
            ),
            "sets": _format_mcq(
                f"Infer the same fixed operation from the examples. {set_demos} "
                f"Apply it to {_set(qa)} and {_set(qb)}.",
                set_options,
                compact=False,
            ),
        },
    }


MOVES = {
    "U": (0, -1),
    "R": (1, 0),
    "D": (0, 1),
    "L": (-1, 0),
}


def _neighbors(pos: tuple[int, int], blocked: set[tuple[int, int]], n: int) -> Iterable[tuple[tuple[int, int], str]]:
    x, y = pos
    for action, (dx, dy) in MOVES.items():
        q = (x + dx, y + dy)
        if 0 <= q[0] < n and 0 <= q[1] < n and q not in blocked:
            yield q, action


def _all_shortest(start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]], n: int) -> list[str]:
    q = deque([(start, "")])
    seen_depth = {start: 0}
    solutions: list[str] = []
    best = None
    while q:
        pos, path = q.popleft()
        if best is not None and len(path) > best:
            break
        if pos == goal:
            best = len(path)
            solutions.append(path)
            continue
        for nxt, action in _neighbors(pos, blocked, n):
            depth = len(path) + 1
            old = seen_depth.get(nxt)
            if old is None or depth <= old:
                seen_depth[nxt] = depth
                q.append((nxt, path + action))
    return solutions


def _follow(start: tuple[int, int], path: str, blocked: set[tuple[int, int]], n: int) -> tuple[int, int] | None:
    x, y = start
    for a in path:
        dx, dy = MOVES[a]
        x, y = x + dx, y + dy
        if not (0 <= x < n and 0 <= y < n) or (x, y) in blocked:
            return None
    return x, y


def _planning_world(rng: random.Random, difficulty: str) -> tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], str]:
    n = 4
    target_len = 3 if difficulty == "easy" else 5
    cells = [(x, y) for y in range(n) for x in range(n)]
    for _ in range(5000):
        start, goal = rng.sample(cells, 2)
        available = [c for c in cells if c not in {start, goal}]
        block_count = 2 if difficulty == "easy" else 4
        blocked = set(rng.sample(available, block_count))
        sols = _all_shortest(start, goal, blocked, n)
        if len(sols) == 1 and len(sols[0]) == target_len:
            return n, start, goal, blocked, sols[0]
    raise RuntimeError("could not generate planning world")


def _ascii_grid(n: int, start: tuple[int, int], goal: tuple[int, int], blocked: set[tuple[int, int]]) -> str:
    rows = []
    for y in range(n):
        row = []
        for x in range(n):
            p = (x, y)
            row.append("S" if p == start else "G" if p == goal else "#" if p in blocked else ".")
        rows.append("".join(row))
    return "/".join(rows)


def make_plan_case(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _stable_rng(seed, "plan", index)
    n, start, goal, blocked, correct = _planning_world(rng, difficulty)
    wrong: list[str] = []
    actions = list(MOVES)
    while len(wrong) < 3:
        candidate = "".join(rng.choice(actions) for _ in range(len(correct)))
        if candidate == correct or candidate in wrong:
            continue
        if _follow(start, candidate, blocked, n) != goal:
            wrong.append(candidate)
    correct_slot = index % 4
    options, slot = _balanced_options(rng, correct, wrong, correct_slot)
    blocked_coords = sorted(blocked)
    return {
        "family": "grounded-planning",
        "difficulty": difficulty,
        "case_id": f"plan-{index:02d}",
        "options": options,
        "correct_index": slot,
        "surfaces": {
            "grid": _format_mcq(
                "Plan from S to G without entering #. U,R,D,L are moves. "
                f"Grid rows top-to-bottom: {_ascii_grid(n, start, goal, blocked)}. "
                f"Choose the {len(correct)}-move route.",
                options,
                compact=False,
            ),
            "coordinates": _format_mcq(
                f"Board {n}x{n}; origin top-left; S={start}; G={goal}; blocked={blocked_coords}; "
                f"moves U,R,D,L; choose exact {len(correct)}-move route.",
                options,
                compact=True,
            ),
        },
    }


OPEN = ("ka", "mo", "su")
CLOSE = ("ti", "re", "va")
PAIR = dict(zip(OPEN, CLOSE))
INV_PAIR = dict(zip(CLOSE, OPEN))


def _balanced_token_sequence(rng: random.Random, pairs: int) -> list[str]:
    out: list[str] = []
    stack: list[str] = []
    opens_left = pairs
    while opens_left or stack:
        must_close = opens_left == 0
        may_close = bool(stack)
        if must_close or (may_close and rng.random() < 0.45):
            out.append(PAIR[stack.pop()])
        else:
            tok = rng.choice(OPEN)
            out.append(tok)
            stack.append(tok)
            opens_left -= 1
    return out


def _valid_stack(tokens: list[str]) -> bool:
    stack: list[str] = []
    for t in tokens:
        if t in PAIR:
            stack.append(t)
        elif t in INV_PAIR:
            if not stack or stack.pop() != INV_PAIR[t]:
                return False
        else:
            return False
    return not stack


def make_stack_case(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _stable_rng(seed, "stack", index)
    pairs = 3 if difficulty == "easy" else 5
    cut = 2 if difficulty == "easy" else 4
    while True:
        full = _balanced_token_sequence(rng, pairs)
        if len(full) > cut and any(t in OPEN for t in full[:-cut]):
            prefix = full[:-cut]
            completion = full[-cut:]
            if not _valid_stack(prefix):
                break
    correct = " ".join(completion)
    wrong: list[str] = []
    universe = list(OPEN + CLOSE)
    while len(wrong) < 3:
        candidate = completion[:]
        pos = rng.randrange(len(candidate))
        replacement = rng.choice([t for t in universe if t != candidate[pos]])
        candidate[pos] = replacement
        text = " ".join(candidate)
        if text != correct and text not in wrong and not _valid_stack(prefix + candidate):
            wrong.append(text)
    correct_slot = index % 4
    options, slot = _balanced_options(rng, correct, wrong, correct_slot)
    legend = ", ".join(f"{o} opens and {c} closes type {i+1}" for i, (o, c) in enumerate(zip(OPEN, CLOSE)))
    return {
        "family": "stack-language",
        "difficulty": difficulty,
        "case_id": f"stack-{index:02d}",
        "options": options,
        "correct_index": slot,
        "surfaces": {
            "prose": _format_mcq(
                f"Nested tokens must close in last-opened-first-closed order. {legend}. "
                f"Prefix: {' '.join(prefix)}. Which completion makes the whole sequence balanced?",
                options,
                compact=False,
            ),
            "compact": _format_mcq(
                f"pairs ka/ti mo/re su/va; LIFO; prefix={' '.join(prefix)}; completion=?",
                options,
                compact=True,
            ),
        },
    }


FAMILIES = (
    make_relation_case,
    make_grid_case,
    make_matrix_case,
    make_plan_case,
    make_stack_case,
)


def build_battery(seed: int, cases_per_family: int) -> list[dict[str, Any]]:
    cases = []
    for maker in FAMILIES:
        for i in range(cases_per_family):
            difficulty = "easy" if i < cases_per_family // 2 else "hard"
            cases.append(maker(seed, i, difficulty))
    return cases


class ChoiceScorer:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.label_prior = self.score_choices("Answer:", LABELS)

    @torch.inference_mode()
    def score_choices(self, prompt: str, choices: Iterable[str]) -> list[float]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        sequences: list[list[int]] = []
        choice_ids_list: list[list[int]] = []
        for choice in choices:
            choice_ids = self.tokenizer(" " + str(choice), add_special_tokens=False).input_ids
            if not choice_ids:
                raise ValueError("empty choice tokenization")
            choice_ids_list.append(choice_ids)
            sequences.append(prompt_ids + choice_ids)

        max_len = max(map(len, sequences))
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(sequences), max_len), pad, dtype=torch.long, device=self.device)
        attention = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[i, : len(seq)] = 1

        logits = self.model(input_ids=input_ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits, dim=-1)

        scores: list[float] = []
        prefix_len = len(prompt_ids)
        for i, choice_ids in enumerate(choice_ids_list):
            total = 0.0
            for j, tok in enumerate(choice_ids):
                token_pos = prefix_len + j
                total += float(log_probs[i, token_pos - 1, tok])
            scores.append(total / len(choice_ids))
        return scores

    def choose(self, prompt: str) -> dict[str, Any]:
        raw = self.score_choices(prompt, LABELS)
        mi = [raw[i] - self.label_prior[i] for i in range(4)]
        raw_pred = max(range(4), key=lambda i: raw[i])
        mi_pred = max(range(4), key=lambda i: mi[i])
        raw_sorted = sorted(raw, reverse=True)
        mi_sorted = sorted(mi, reverse=True)
        return {
            "raw_scores": raw,
            "mi_scores": mi,
            "raw_prediction": raw_pred,
            "mi_prediction": mi_pred,
            "raw_margin": raw_sorted[0] - raw_sorted[1],
            "mi_margin": mi_sorted[0] - mi_sorted[1],
        }


def evaluate(model_dir: str, seed: int, cases_per_family: int) -> dict[str, Any]:
    cases = build_battery(seed, cases_per_family)
    scorer = ChoiceScorer(model_dir)
    started = time.time()
    rows = []

    for case in cases:
        predictions = {}
        for surface, prompt in case["surfaces"].items():
            scored = scorer.choose(prompt)
            predictions[surface] = {
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                **scored,
            }
        rows.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "difficulty": case["difficulty"],
            "correct_index": case["correct_index"],
            "predictions": predictions,
        })

    aggregate: dict[str, Any] = {}
    for family in sorted({r["family"] for r in rows}):
        fam = [r for r in rows if r["family"] == family]
        surface_names = sorted(fam[0]["predictions"])
        surface_stats = {}
        for surface in surface_names:
            raw_correct = [r["predictions"][surface]["raw_prediction"] == r["correct_index"] for r in fam]
            mi_correct = [r["predictions"][surface]["mi_prediction"] == r["correct_index"] for r in fam]
            surface_stats[surface] = {
                "cases": len(fam),
                "raw_accuracy": sum(raw_correct) / len(fam),
                "mi_accuracy": sum(mi_correct) / len(fam),
                "mean_raw_margin": sum(r["predictions"][surface]["raw_margin"] for r in fam) / len(fam),
                "mean_mi_margin": sum(r["predictions"][surface]["mi_margin"] for r in fam) / len(fam),
                "easy_mi_accuracy": sum(
                    r["predictions"][surface]["mi_prediction"] == r["correct_index"]
                    for r in fam if r["difficulty"] == "easy"
                ) / max(1, sum(r["difficulty"] == "easy" for r in fam)),
                "hard_mi_accuracy": sum(
                    r["predictions"][surface]["mi_prediction"] == r["correct_index"]
                    for r in fam if r["difficulty"] == "hard"
                ) / max(1, sum(r["difficulty"] == "hard" for r in fam)),
                "predicted_label_counts": dict(Counter(
                    LABELS[r["predictions"][surface]["mi_prediction"]] for r in fam
                )),
            }
        agreement = sum(
            len({p["mi_prediction"] for p in r["predictions"].values()}) == 1 for r in fam
        ) / len(fam)
        both_correct = sum(
            all(p["mi_prediction"] == r["correct_index"] for p in r["predictions"].values())
            for r in fam
        ) / len(fam)
        aggregate[family] = {
            "surfaces": surface_stats,
            "surface_prediction_agreement": agreement,
            "all_surfaces_correct": both_correct,
        }

    all_decisions = [
        (r, surface, p)
        for r in rows
        for surface, p in r["predictions"].items()
    ]
    overall = {
        "cases": len(rows),
        "surface_decisions": len(all_decisions),
        "raw_accuracy": sum(p["raw_prediction"] == r["correct_index"] for r, _, p in all_decisions) / len(all_decisions),
        "mi_accuracy": sum(p["mi_prediction"] == r["correct_index"] for r, _, p in all_decisions) / len(all_decisions),
        "chance_accuracy": 0.25,
        "case_surface_agreement": sum(
            len({p["mi_prediction"] for p in r["predictions"].values()}) == 1 for r in rows
        ) / len(rows),
        "all_surfaces_correct": sum(
            all(p["mi_prediction"] == r["correct_index"] for p in r["predictions"].values())
            for r in rows
        ) / len(rows),
        "elapsed_seconds": time.time() - started,
    }

    public_spec = [
        {
            "case_id": c["case_id"],
            "family": c["family"],
            "difficulty": c["difficulty"],
            "correct_index": c["correct_index"],
            "surface_sha256": {
                k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in c["surfaces"].items()
            },
        }
        for c in cases
    ]
    return {
        "schema_version": BATTERY_VERSION,
        "status": "public-burned-calibration",
        "seed": seed,
        "cases_per_family": cases_per_family,
        "donors": DONORS,
        "scoring": {
            "method": "teacher-forced label log-likelihood",
            "labels": list(LABELS),
            "correct_label_positions_balanced_per_family": True,
            "mutual_information_normalization": "subtract log-likelihood of the same answer label under neutral context 'Answer:'",
            "single_scalar_score_authoritative": False,
        },
        "label_prior_loglikelihood": scorer.label_prior,
        "battery_sha256": sha256_json(public_spec),
        "family_results": aggregate,
        "overall_diagnostic": overall,
        "cases": rows,
        "claim_boundary": (
            "Fresh procedural public calibration only. This run is not an official score on donor benchmarks, "
            "is not protected held-out evidence, does not establish AGI, and cannot authorize model promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="offline-reasoning.json")
    args = parser.parse_args()

    path = Path(args.model_file)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    result = evaluate(args.model_dir, args.seed, args.cases_per_family)
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": observed,
        "model_bytes": path.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema_version": result["schema_version"],
        "battery_sha256": result["battery_sha256"],
        "family_results": result["family_results"],
        "overall_diagnostic": result["overall_diagnostic"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
