"""Run 030: representation transfer to unseen procedural geometries.

Five new deterministic reasoning families are generated from fresh seeds. Their
representation renderers receive only visible task state and answer values,
never the hidden solution used for scoring.

Families:
- directed graph shortest path
- temporal ordering
- deterministic finite-state execution
- permutation constraints
- register-program execution
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

from run_026_capability_substitution import AttackScorer
from run_028_canonical_representation import _score_values

RUN_VERSION = "helix-unseen-geometry-representation-001.0"
NODES = ("A", "B", "C", "D", "E", "F")
EVENTS = ("Ari", "Bex", "Cato", "Dara", "Eno")
STATES = ("S0", "S1", "S2", "S3")
ITEMS = ("Ari", "Bex", "Cato", "Dara")
REGS = ("X", "Y", "Z")


def _rng(seed: int, *parts: Any) -> random.Random:
    material = "|".join([str(seed), *(str(value) for value in parts)])
    return random.Random(
        int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")
    )


def _balanced(
    rng: random.Random,
    correct: str,
    distractors: list[str],
    slot: int,
) -> tuple[list[str], int]:
    wrong = []
    for value in distractors:
        if value != correct and value not in wrong:
            wrong.append(value)
    rng.shuffle(wrong)
    if len(wrong) < 3:
        raise ValueError("not enough distractors")
    choices = wrong[:3]
    choices.insert(slot % 4, correct)
    return choices, slot % 4


def _bfs_distance(
    nodes: tuple[str, ...],
    edges: set[tuple[str, str]],
    source: str,
    target: str,
) -> int | None:
    queue = deque([(source, 0)])
    seen = {source}
    adjacency = defaultdict(list)
    for left, right in edges:
        adjacency[left].append(right)
    while queue:
        node, distance = queue.popleft()
        if node == target:
            return distance
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def make_graph(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _rng(seed, "graph", index)
    nodes = NODES[: 5 if difficulty == "easy" else 6]
    target_distance = 1 + (index % 4)
    for _ in range(500):
        order = list(nodes)
        rng.shuffle(order)
        source = order[0]
        target = order[target_distance]
        chain = {(order[i], order[i + 1]) for i in range(target_distance)}
        edges = set(chain)
        candidates = [
            (a, b)
            for a in nodes
            for b in nodes
            if a != b and (a, b) not in edges
        ]
        rng.shuffle(candidates)
        for edge in candidates[: 4 if difficulty == "easy" else 7]:
            trial = set(edges)
            trial.add(edge)
            if _bfs_distance(nodes, trial, source, target) == target_distance:
                edges = trial
        if _bfs_distance(nodes, edges, source, target) == target_distance:
            choices = ["1", "2", "3", "4"]
            return {
                "family": "directed-graph-shortest-path",
                "difficulty": difficulty,
                "case_id": f"graph-{index:02d}",
                "visible": {
                    "nodes": list(nodes),
                    "edges": sorted([list(edge) for edge in edges]),
                    "source": source,
                    "target": target,
                },
                "choices": choices,
                "correct_index": choices.index(str(target_distance)),
            }
    raise RuntimeError("could not generate graph case")


def make_temporal(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _rng(seed, "temporal", index)
    events = list(EVENTS)
    rng.shuffle(events)
    facts = []
    for i in range(len(events) - 1):
        left, right = events[i], events[i + 1]
        if rng.random() < 0.5:
            facts.append({"left": left, "relation": "before", "right": right})
        else:
            facts.append({"left": right, "relation": "after", "right": left})
    if difficulty == "hard":
        for _ in range(2):
            i, j = sorted(rng.sample(range(len(events)), 2))
            left, right = events[i], events[j]
            facts.append({"left": left, "relation": "before", "right": right})
    rng.shuffle(facts)
    rank = 1 + (index % len(events))
    correct = events[rank - 1]
    choices, slot = _balanced(
        rng,
        correct,
        [event for event in events if event != correct],
        index % 4,
    )
    return {
        "family": "temporal-ordering",
        "difficulty": difficulty,
        "case_id": f"time-{index:02d}",
        "visible": {"events": sorted(events), "facts": facts, "rank": rank},
        "choices": choices,
        "correct_index": slot,
    }


def make_fsm(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _rng(seed, "fsm", index)
    alphabet = ("x", "y")
    transitions = {
        state: {symbol: rng.choice(STATES) for symbol in alphabet}
        for state in STATES
    }
    start = rng.choice(STATES)
    length = 4 if difficulty == "easy" else 7
    inputs = [rng.choice(alphabet) for _ in range(length)]
    state = start
    for symbol in inputs:
        state = transitions[state][symbol]
    choices = list(STATES)
    rng.shuffle(choices)
    return {
        "family": "finite-state-execution",
        "difficulty": difficulty,
        "case_id": f"fsm-{index:02d}",
        "visible": {
            "states": list(STATES),
            "alphabet": list(alphabet),
            "transitions": transitions,
            "start": start,
            "inputs": inputs,
        },
        "choices": choices,
        "correct_index": choices.index(state),
    }


def _constraint_holds(order: tuple[str, ...], clue: dict[str, str]) -> bool:
    pos = {item: i for i, item in enumerate(order)}
    kind = clue["kind"]
    a = clue["a"]
    b = clue.get("b")
    if kind == "before":
        return pos[a] < pos[str(b)]
    if kind == "immediately-before":
        return pos[a] + 1 == pos[str(b)]
    if kind == "not-first":
        return pos[a] != 0
    if kind == "not-last":
        return pos[a] != len(order) - 1
    raise ValueError(kind)


def make_permutation(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _rng(seed, "perm", index)
    target = list(ITEMS)
    rng.shuffle(target)
    pool: list[dict[str, str]] = []
    for i, a in enumerate(target):
        if i > 0:
            pool.append({"kind": "not-first", "a": a})
        if i < len(target) - 1:
            pool.append({"kind": "not-last", "a": a})
        for j in range(i + 1, len(target)):
            pool.append({"kind": "before", "a": a, "b": target[j]})
        if i + 1 < len(target):
            pool.append({"kind": "immediately-before", "a": a, "b": target[i + 1]})
    rng.shuffle(pool)
    candidates = list(permutations(ITEMS))
    clues = []
    for clue in pool:
        clues.append(clue)
        candidates = [order for order in candidates if _constraint_holds(order, clue)]
        if len(candidates) == 1 and tuple(target) == candidates[0]:
            break
    if len(candidates) != 1:
        raise RuntimeError("permutation clues did not isolate solution")
    if difficulty == "easy":
        # Keep only as many clues as required; hard instances retain extra redundant noise.
        pass
    else:
        extras = [clue for clue in pool if clue not in clues][:2]
        clues.extend(extras)
        rng.shuffle(clues)
    rank = 1 + (index % 4)
    correct = target[rank - 1]
    choices = list(ITEMS)
    rng.shuffle(choices)
    return {
        "family": "permutation-constraints",
        "difficulty": difficulty,
        "case_id": f"perm-{index:02d}",
        "visible": {
            "items": list(ITEMS),
            "clues": clues,
            "rank": rank,
        },
        "choices": choices,
        "correct_index": choices.index(correct),
    }


def _execute_program(initial: dict[str, int], program: list[dict[str, str]]) -> dict[str, int]:
    state = dict(initial)
    for instruction in program:
        op = instruction["op"]
        a = instruction["a"]
        b = instruction.get("b")
        if op == "inc":
            state[a] += 1
        elif op == "double":
            state[a] *= 2
        elif op == "add":
            state[a] += state[str(b)]
        elif op == "sub":
            state[a] -= state[str(b)]
        elif op == "swap":
            other = str(b)
            state[a], state[other] = state[other], state[a]
        else:
            raise ValueError(op)
    return state


def make_register(seed: int, index: int, difficulty: str) -> dict[str, Any]:
    rng = _rng(seed, "register", index)
    initial = {reg: rng.randrange(0, 6) for reg in REGS}
    length = 3 if difficulty == "easy" else 6
    program = []
    for _ in range(length):
        op = rng.choice(("inc", "double", "add", "sub", "swap"))
        a = rng.choice(REGS)
        if op in {"add", "sub", "swap"}:
            b = rng.choice([reg for reg in REGS if reg != a])
            program.append({"op": op, "a": a, "b": b})
        else:
            program.append({"op": op, "a": a})
    final = _execute_program(initial, program)
    query = REGS[index % len(REGS)]
    correct = final[query]
    distractors = []
    for delta in (-3, -2, -1, 1, 2, 3, 4):
        value = str(correct + delta)
        if value != str(correct):
            distractors.append(value)
    choices, slot = _balanced(rng, str(correct), distractors, index % 4)
    return {
        "family": "register-program-execution",
        "difficulty": difficulty,
        "case_id": f"reg-{index:02d}",
        "visible": {
            "initial": initial,
            "program": program,
            "query": query,
        },
        "choices": choices,
        "correct_index": slot,
    }


MAKERS: tuple[Callable[[int, int, str], dict[str, Any]], ...] = (
    make_graph,
    make_temporal,
    make_fsm,
    make_permutation,
    make_register,
)


def build_battery(seed: int, cases_per_family: int) -> list[dict[str, Any]]:
    cases = []
    for maker in MAKERS:
        for index in range(cases_per_family):
            difficulty = "easy" if index < cases_per_family // 2 else "hard"
            cases.append(maker(seed, index, difficulty))
    return cases


def _graph_reps(visible: dict[str, Any], choices: list[str]) -> dict[str, tuple[str, list[str]]]:
    edges = [tuple(edge) for edge in visible["edges"]]
    prose = (
        "A directed arrow can only be followed in its stated direction. "
        + " ".join(f"{a} points to {b}." for a, b in edges)
        + f" What is the minimum number of directed edges needed to get from {visible['source']} to {visible['target']}?"
        + "\nAnswer value:"
    )
    compact = (
        "directed edges="
        + ",".join(f"{a}>{b}" for a, b in edges)
        + f"; shortest_edge_count({visible['source']},{visible['target']})=?\nAnswer value:"
    )
    adjacency = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
    layout = (
        "Directed graph shortest-path task. One edge counts as one step.\n"
        "Adjacency list:\n"
        + "\n".join(
            f"{node}: {', '.join(sorted(adjacency[node])) or '(none)'}"
            for node in visible["nodes"]
        )
        + f"\nStart: {visible['source']}\nTarget: {visible['target']}\n"
        "Return the minimum number of edges.\nAnswer value:"
    )
    nodes = visible["nodes"]
    header = "    " + " ".join(nodes)
    matrix_rows = []
    edge_set = set(edges)
    for a in nodes:
        matrix_rows.append(a + " : " + " ".join("1" if (a, b) in edge_set else "0" for b in nodes))
    familiar = (
        "Use this directed adjacency matrix. Row -> column equals 1 when that edge exists. "
        "Shortest-path length is the fewest directed edge traversals from start to target.\n"
        + header + "\n" + "\n".join(matrix_rows)
        + f"\nStart={visible['source']} Target={visible['target']}\nAnswer value:"
    )
    return {
        "prose-original": (prose, choices),
        "compact-original": (compact, choices),
        "layout-normalized": (layout, choices),
        "familiar-structural-normalization": (familiar, choices),
    }


def _fact_text(clue: dict[str, Any]) -> str:
    if clue["relation"] == "before":
        return f"{clue['left']} happens before {clue['right']}."
    return f"{clue['left']} happens after {clue['right']}."


def _temporal_reps(visible: dict[str, Any], choices: list[str]) -> dict[str, tuple[str, list[str]]]:
    facts = visible["facts"]
    prose = (
        "Use all ordering facts. "
        + " ".join(_fact_text(fact) for fact in facts)
        + f" Which event is number {visible['rank']} from earliest to latest?\nAnswer value:"
    )
    symbols = []
    for fact in facts:
        if fact["relation"] == "before":
            symbols.append(f"{fact['left']}<{fact['right']}")
        else:
            symbols.append(f"{fact['right']}<{fact['left']}")
    compact = (
        "; ".join(symbols)
        + f"; rank_{visible['rank']}_earliest=?\nAnswer value:"
    )
    layout = (
        "Temporal ordering task. Every row means Earlier -> Later.\n"
        + "\n".join(
            f"- {fact['left']} -> {fact['right']}"
            if fact["relation"] == "before"
            else f"- {fact['right']} -> {fact['left']}"
            for fact in facts
        )
        + f"\nQuestion: which event occupies position {visible['rank']} in the full earliest-to-latest order?\n"
        "Answer value:"
    )
    familiar = (
        "Treat each event as having a time coordinate t(event). The visible constraints are:\n"
        + "\n".join(f"- t({expr.split('<')[0]}) < t({expr.split('<')[1]})" for expr in symbols)
        + f"\nSort the events by increasing time and return item {visible['rank']}.\nAnswer value:"
    )
    return {
        "prose-original": (prose, choices),
        "compact-original": (compact, choices),
        "layout-normalized": (layout, choices),
        "familiar-structural-normalization": (familiar, choices),
    }


def _fsm_reps(visible: dict[str, Any], choices: list[str]) -> dict[str, tuple[str, list[str]]]:
    transitions = visible["transitions"]
    sentences = []
    compact_parts = []
    for state in visible["states"]:
        for symbol in visible["alphabet"]:
            nxt = transitions[state][symbol]
            sentences.append(f"In {state}, input {symbol} moves to {nxt}.")
            compact_parts.append(f"{state}/{symbol}->{nxt}")
    sequence = "".join(visible["inputs"])
    prose = (
        "This is a deterministic state machine. "
        + " ".join(sentences)
        + f" Start in {visible['start']}. Read inputs {', '.join(visible['inputs'])} in order. What is the final state?\nAnswer value:"
    )
    compact = (
        "; ".join(compact_parts)
        + f"; start={visible['start']}; input={sequence}; final=?\nAnswer value:"
    )
    layout = (
        "Deterministic finite-state machine. Apply one transition per input symbol.\n"
        "Transition table:\nstate | x | y\n"
        + "\n".join(
            f"{state} | {transitions[state]['x']} | {transitions[state]['y']}"
            for state in visible["states"]
        )
        + f"\nStart: {visible['start']}\nInput sequence: {' '.join(visible['inputs'])}\nFinal state?\nAnswer value:"
    )
    familiar = (
        "Use transition function delta(state, symbol).\n"
        + "\n".join(
            f"delta({state},x)={transitions[state]['x']}; delta({state},y)={transitions[state]['y']}"
            for state in visible["states"]
        )
        + f"\nCompute delta repeatedly from {visible['start']} over [{', '.join(visible['inputs'])}].\nAnswer value:"
    )
    return {
        "prose-original": (prose, choices),
        "compact-original": (compact, choices),
        "layout-normalized": (layout, choices),
        "familiar-structural-normalization": (familiar, choices),
    }


def _clue_text(clue: dict[str, str]) -> str:
    kind = clue["kind"]
    if kind == "before":
        return f"{clue['a']} is somewhere before {clue['b']}."
    if kind == "immediately-before":
        return f"{clue['a']} is immediately before {clue['b']}."
    if kind == "not-first":
        return f"{clue['a']} is not first."
    if kind == "not-last":
        return f"{clue['a']} is not last."
    raise ValueError(kind)


def _perm_reps(visible: dict[str, Any], choices: list[str]) -> dict[str, tuple[str, list[str]]]:
    clues = visible["clues"]
    prose = (
        "Arrange the four items from position 1 to position 4. "
        + " ".join(_clue_text(clue) for clue in clues)
        + f" Which item is in position {visible['rank']}?\nAnswer value:"
    )
    compact_tokens = []
    for clue in clues:
        if clue["kind"] == "before":
            compact_tokens.append(f"{clue['a']}<{clue['b']}")
        elif clue["kind"] == "immediately-before":
            compact_tokens.append(f"{clue['a']}+1={clue['b']}")
        elif clue["kind"] == "not-first":
            compact_tokens.append(f"{clue['a']}!=1")
        else:
            compact_tokens.append(f"{clue['a']}!=4")
    compact = (
        "; ".join(compact_tokens)
        + f"; item_at_position_{visible['rank']}=?\nAnswer value:"
    )
    layout = (
        "Permutation constraint task. Positions are 1,2,3,4 from first to last.\n"
        "Constraints:\n"
        + "\n".join(f"- {_clue_text(clue)}" for clue in clues)
        + f"\nQuestion: item in position {visible['rank']}?\nAnswer value:"
    )
    familiar = (
        "Let p(item) be its integer position in {1,2,3,4}.\n"
        + "\n".join(
            (
                f"- p({clue['a']}) < p({clue['b']})"
                if clue["kind"] == "before"
                else f"- p({clue['a']}) + 1 = p({clue['b']})"
                if clue["kind"] == "immediately-before"
                else f"- p({clue['a']}) != 1"
                if clue["kind"] == "not-first"
                else f"- p({clue['a']}) != 4"
            )
            for clue in clues
        )
        + f"\nWhich item has p(item)={visible['rank']}?\nAnswer value:"
    )
    return {
        "prose-original": (prose, choices),
        "compact-original": (compact, choices),
        "layout-normalized": (layout, choices),
        "familiar-structural-normalization": (familiar, choices),
    }


def _instruction_text(ins: dict[str, str]) -> str:
    op, a, b = ins["op"], ins["a"], ins.get("b")
    if op == "inc":
        return f"increase {a} by 1"
    if op == "double":
        return f"double {a}"
    if op == "add":
        return f"set {a} to {a} plus {b}"
    if op == "sub":
        return f"set {a} to {a} minus {b}"
    if op == "swap":
        return f"swap {a} and {b}"
    raise ValueError(op)


def _instruction_code(ins: dict[str, str]) -> str:
    op, a, b = ins["op"], ins["a"], ins.get("b")
    if op == "inc":
        return f"{a} = {a} + 1"
    if op == "double":
        return f"{a} = 2 * {a}"
    if op == "add":
        return f"{a} = {a} + {b}"
    if op == "sub":
        return f"{a} = {a} - {b}"
    if op == "swap":
        return f"{a}, {b} = {b}, {a}"
    raise ValueError(op)


def _register_reps(visible: dict[str, Any], choices: list[str]) -> dict[str, tuple[str, list[str]]]:
    init = visible["initial"]
    program = visible["program"]
    prose = (
        f"Start with X={init['X']}, Y={init['Y']}, Z={init['Z']}. Execute in order: "
        + "; ".join(_instruction_text(ins) for ins in program)
        + f". What is the final value of {visible['query']}?\nAnswer value:"
    )
    compact = (
        f"X={init['X']},Y={init['Y']},Z={init['Z']}; "
        + "; ".join(_instruction_code(ins) for ins in program)
        + f"; final({visible['query']})=?\nAnswer value:"
    )
    layout = (
        "Sequential register program. Each instruction uses the current state produced by the previous row.\n"
        f"Initial state: X={init['X']}  Y={init['Y']}  Z={init['Z']}\nProgram:\n"
        + "\n".join(f"{i+1}. {_instruction_text(ins)}" for i, ins in enumerate(program))
        + f"\nQuestion: final {visible['query']}?\nAnswer value:"
    )
    familiar = (
        "Execute this pseudocode sequentially.\n"
        f"X = {init['X']}\nY = {init['Y']}\nZ = {init['Z']}\n"
        + "\n".join(_instruction_code(ins) for ins in program)
        + f"\nReturn {visible['query']}.\nAnswer value:"
    )
    return {
        "prose-original": (prose, choices),
        "compact-original": (compact, choices),
        "layout-normalized": (layout, choices),
        "familiar-structural-normalization": (familiar, choices),
    }


BUILDERS = {
    "directed-graph-shortest-path": _graph_reps,
    "temporal-ordering": _temporal_reps,
    "finite-state-execution": _fsm_reps,
    "permutation-constraints": _perm_reps,
    "register-program-execution": _register_reps,
}
ARMS = (
    "prose-original",
    "compact-original",
    "layout-normalized",
    "familiar-structural-normalization",
)


def representations(case: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    # Answer-blind by signature: hidden scoring labels are not passed.
    return BUILDERS[case["family"]](case["visible"], list(case["choices"]))


def validate_generators() -> dict[str, Any]:
    digests = []
    counts = Counter()
    for seed in (20260927, 20260928):
        for case in build_battery(seed, 8):
            reps = representations(case)
            if set(reps) != set(ARMS):
                raise AssertionError((case["case_id"], sorted(reps)))
            for arm, (prompt, choices) in reps.items():
                if not prompt.endswith("Answer value:") or len(choices) != 4:
                    raise AssertionError((case["case_id"], arm))
                if not 0 <= int(case["correct_index"]) < 4:
                    raise AssertionError(case["case_id"])
                counts[(case["family"], arm)] += 1
                digests.append(hashlib.sha256(
                    json.dumps(
                        {"family": case["family"], "arm": arm, "prompt": prompt, "choices": choices},
                        sort_keys=True,
                    ).encode()
                ).hexdigest())
    return {
        "cases": 80,
        "representations": len(digests),
        "family_arm_counts": {
            f"{family}/{arm}": count
            for (family, arm), count in sorted(counts.items())
        },
        "representation_digest": hashlib.sha256("|".join(digests).encode()).hexdigest(),
    }


def _summary(cases: list[dict[str, Any]], preds: dict[str, int]) -> dict[str, Any]:
    hits = 0
    by_family = defaultdict(list)
    by_seed = defaultdict(list)
    for case in cases:
        ok = int(preds[case["uid"]]) == int(case["correct_index"])
        hits += int(ok)
        by_family[case["family"]].append(ok)
        by_seed[int(case["seed"])].append(ok)
    return {
        "cases": len(cases),
        "correct": hits,
        "accuracy": hits / len(cases),
        "by_family": {
            family: sum(values) / len(values)
            for family, values in sorted(by_family.items())
        },
        "by_seed": {
            str(seed): sum(values) / len(values)
            for seed, values in sorted(by_seed.items())
        },
    }


def evaluate(model_dir: str, seeds: list[int], cases_per_family: int) -> dict[str, Any]:
    cases = []
    rendered = {}
    spec = []
    for seed in seeds:
        seed_rows = []
        for case in build_battery(seed, cases_per_family):
            uid = f"{seed}:{case['case_id']}"
            row = {**case, "uid": uid, "seed": seed}
            cases.append(row)
            reps = representations(case)
            rendered[uid] = reps
            seed_rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "visible_sha256": hashlib.sha256(
                    json.dumps(case["visible"], sort_keys=True).encode()
                ).hexdigest(),
                "representation_sha256": {
                    arm: hashlib.sha256(
                        json.dumps({"prompt": prompt, "choices": choices}, sort_keys=True).encode()
                    ).hexdigest()
                    for arm, (prompt, choices) in reps.items()
                },
                "correct_index": case["correct_index"],
            })
        spec.append({"seed": seed, "cases": seed_rows})

    scorer = AttackScorer(model_dir)
    started = time.time()
    scored: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    resources = {}
    for arm in ARMS:
        scorer.reset_counters()
        arm_started = time.time()
        for case in cases:
            prompt, choices = rendered[case["uid"]][arm]
            scored[arm][case["uid"]] = _score_values(scorer, prompt, choices)
        resources[arm] = {
            **scorer.counters(),
            "wall_seconds": time.time() - arm_started,
        }

    preds = {
        arm: {uid: row["prediction"] for uid, row in rows.items()}
        for arm, rows in scored.items()
    }
    summaries = {arm: _summary(cases, preds[arm]) for arm in ARMS}

    oracle_hits = 0
    oracle_by_family = defaultdict(list)
    high_margin = {}
    choice_counts = Counter()
    rows = []
    for case in cases:
        uid = case["uid"]
        candidates = {arm: scored[arm][uid] for arm in ARMS}
        selected_arm, selected = max(
            candidates.items(),
            key=lambda item: (float(item[1]["margin"]), item[0]),
        )
        high_margin[uid] = int(selected["prediction"])
        choice_counts[selected_arm] += 1
        correct_arms = [
            arm for arm, value in candidates.items()
            if int(value["prediction"]) == int(case["correct_index"])
        ]
        oracle = bool(correct_arms)
        oracle_hits += int(oracle)
        oracle_by_family[case["family"]].append(oracle)
        rows.append({
            "uid": uid,
            "seed": case["seed"],
            "case_id": case["case_id"],
            "family": case["family"],
            "difficulty": case["difficulty"],
            "predictions": {arm: int(value["prediction"]) for arm, value in candidates.items()},
            "margins": {arm: float(value["margin"]) for arm, value in candidates.items()},
            "correct_representation_count": len(correct_arms),
            "correct_representations": correct_arms,
            "highest_margin_arm": selected_arm,
        })

    summaries["highest-margin-selection"] = _summary(cases, high_margin)
    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-unseen-geometry-transfer",
        "seeds": seeds,
        "cases_per_family": cases_per_family,
        "cases": len(cases),
        "source_battery_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "unseen_families": sorted({case["family"] for case in cases}),
        "answer_blind_contract": {
            "representation_function": "BUILDERS[family](visible, choices)",
            "visible_state_only": True,
            "correct_index_passed_to_builder": False,
        },
        "summaries": summaries,
        "oracle_representation_coverage": {
            "correct": oracle_hits,
            "cases": len(cases),
            "coverage": oracle_hits / len(cases),
            "by_family": {
                family: sum(values) / len(values)
                for family, values in sorted(oracle_by_family.items())
            },
        },
        "highest_margin_representation_choice_counts": dict(choice_counts),
        "resources": resources,
        "rows": rows,
        "total_elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "Fresh public procedural unseen-family transfer only. No copied donor benchmark "
            "items are used. This does not authorize promotion, frontier equivalence, or a "
            "general reasoning claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seeds", default="20260927,20260928,20260929")
    parser.add_argument("--cases-per-family", type=int, default=8)
    parser.add_argument("--output", default="run-030-unseen-geometry.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(validate_generators(), indent=2, sort_keys=True))
        return
    if not args.model_dir or not args.model_file or not args.expected_sha256:
        raise SystemExit("model-dir, model-file and expected-sha256 are required")

    model_path = Path(args.model_file)
    observed = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if observed != args.expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    result = evaluate(args.model_dir, seeds, args.cases_per_family)
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
        "summaries": result["summaries"],
        "oracle_representation_coverage": result["oracle_representation_coverage"],
        "highest_margin_representation_choice_counts": result[
            "highest_margin_representation_choice_counts"
        ],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
