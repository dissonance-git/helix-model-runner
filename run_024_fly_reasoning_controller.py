"""Run 024 first execution: temporal representation-budget controller.

The controller never predicts the answer. It sees two cheap representation
states and ranks which cases deserve two additional representation evaluations
plus one bounded external verifier call.

Calibration labels may train the controller because this is public/burned
calibration. Fresh test correctness/benefit is hidden until scoring.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from offline_reasoning_battery import build_battery, sha256_json
from run_026_capability_substitution import AttackScorer
from run_032_cross_representation_invariant import representations_for_case, score_values
from connectome_open_weight_router import (
    projection_fit,
    reservoir_features,
    file_sha256,
)
from connectome_temporal_probe import BRAIN_SHA256, WEIGHTS_SHA256, sha256 as fly_sha256

RUN_VERSION = "helix-fly-reasoning-controller-first-execution-001.0"
SEED = 24024
FAMILIES = (
    "abstract-transformation",
    "grounded-planning",
    "relational-composition",
    "relational-matrix",
    "stack-language",
)
CHEAP = ("layout-normalized", "familiar-structural-normalization")
ALL_VIEWS = (
    "lexicographic-primary-surface",
    "alternate-secondary-surface",
    "layout-normalized",
    "familiar-structural-normalization",
)
BUDGETS = (0.0, 0.25, 0.5, 0.75, 1.0)


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, np.float32)
    out[int(index)] = 1.0
    return out


def case_features(case: dict[str, Any], scorer: AttackScorer) -> dict[str, Any]:
    reps = representations_for_case(case)
    scored = {
        name: score_values(scorer, prompt, choices)
        for name, (prompt, choices) in reps.items()
    }
    family = one_hot(FAMILIES.index(case["family"]), len(FAMILIES))
    sequence = []
    for name in CHEAP:
        row = scored[name]
        sequence.append(np.concatenate([
            np.asarray(row["scores"], np.float32),
            np.asarray([row["margin"]], np.float32),
            one_hot(int(row["prediction"]), 4),
            family,
        ]))
    seq = np.stack(sequence).astype(np.float32)
    agree = float(scored[CHEAP[0]]["prediction"] == scored[CHEAP[1]]["prediction"])
    static = np.concatenate([seq.reshape(-1), np.asarray([agree], np.float32)]).astype(np.float32)
    correct = int(case["correct_index"])
    layout_correct = int(scored["layout-normalized"]["prediction"]) == correct
    full_oracle = any(int(scored[name]["prediction"]) == correct for name in ALL_VIEWS)
    benefit = (not layout_correct) and full_oracle
    return {
        "static": static,
        "sequence": seq,
        "benefit": int(benefit),
        "layout_correct": bool(layout_correct),
        "full_oracle": bool(full_oracle),
        "cheap_disagree": not bool(agree),
        "layout_margin": float(scored["layout-normalized"]["margin"]),
        "predictions": {name: int(scored[name]["prediction"]) for name in ALL_VIEWS},
        "margins": {name: float(scored[name]["margin"]) for name in ALL_VIEWS},
    }


def build_split(
    scorer: AttackScorer,
    seeds: list[int],
    cases_per_family: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    spec = []
    for seed in seeds:
        cases = build_battery(seed, cases_per_family)
        seed_spec = []
        for case in cases:
            feat = case_features(case, scorer)
            row = {
                "uid": f"{seed}:{case['case_id']}",
                "seed": seed,
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": int(case["correct_index"]),
                **feat,
            }
            rows.append(row)
            seed_spec.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "difficulty": case["difficulty"],
                "correct_index": int(case["correct_index"]),
                "surface_sha256": {
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in case["surfaces"].items()
                },
            })
        spec.append({"seed": seed, "cases": seed_spec})
    return rows, {"seeds": seeds, "sha256": sha256_json(spec), "cases": len(rows)}


def stack(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.stack([row[key] for row in rows]).astype(np.float32)


def targets(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row["benefit"] for row in rows], np.float32)


def ridge_scores(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, rank: int = 16, lam: float = 1.0) -> np.ndarray:
    Xtr = np.asarray(Xtr, np.float64)
    Xte = np.asarray(Xte, np.float64)
    ytr = np.asarray(ytr, np.float64)
    mu = Xtr.mean(0)
    xc = Xtr - mu
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = max(1, min(rank, len(vt), Xtr.shape[0] - 1))
    P = vt[:k].T
    ztr = xc @ P
    scale = ztr.std(0) + 1e-6
    ztr /= scale
    zte = ((Xte - mu) @ P) / scale
    zm = ztr.mean(0)
    ym = float(ytr.mean())
    zc = ztr - zm
    w = np.linalg.solve(
        zc.T @ zc + lam * len(ytr) * np.eye(k),
        zc.T @ (ytr - ym),
    )
    b = ym - zm @ w
    return (zte @ w + b).astype(np.float64)


def standardize(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


class TinyMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class TinyGRU(nn.Module):
    def __init__(self, d: int, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h = self.gru(x)
        return self.out(h[-1]).squeeze(-1)


def train_torch_scores(
    model: nn.Module,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    *,
    seed: int,
    epochs: int = 240,
    lr: float = 0.02,
) -> np.ndarray:
    torch.manual_seed(seed)
    model.train()
    tx = torch.tensor(Xtr, dtype=torch.float32)
    ty = torch.tensor(ytr, dtype=torch.float32)
    te = torch.tensor(Xte, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    pos = max(1.0, float((len(ytr) - ytr.sum()) / max(1.0, ytr.sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos))
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(tx)
        loss = loss_fn(logits, ty)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(te).cpu().numpy().astype(np.float64)


def random_recurrent_features(
    seq: np.ndarray,
    *,
    seed: int,
    width: int = 128,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = seq.shape[-1]
    Win = rng.normal(scale=1.0 / math.sqrt(d), size=(d, width)).astype(np.float32)
    Wrec = rng.normal(scale=1.0 / math.sqrt(width), size=(width, width)).astype(np.float32)
    eig = np.max(np.abs(np.linalg.eigvals(Wrec.astype(np.float64))))
    if eig > 0:
        Wrec *= np.float32(0.85 / eig)
    state = np.zeros((len(seq), width), np.float32)
    for t in range(seq.shape[1]):
        state = np.tanh(seq[:, t, :] @ Win + state @ Wrec)
    return state


def curve(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    if len(scores) != len(rows):
        raise ValueError("score length mismatch")
    order = sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i]["uid"]))
    points = []
    for frac in BUDGETS:
        k = int(round(len(rows) * frac))
        widen = set(order[:k])
        success = 0
        useful_widens = 0
        wasted_widens = 0
        for i, row in enumerate(rows):
            if row["layout_correct"]:
                success += 1
            elif i in widen and row["full_oracle"]:
                success += 1
            if i in widen:
                if row["benefit"]:
                    useful_widens += 1
                else:
                    wasted_widens += 1
        points.append({
            "widen_fraction": frac,
            "widen_cases": k,
            "accuracy": success / len(rows),
            "correct": success,
            "useful_widens": useful_widens,
            "wasted_widens": wasted_widens,
            "mean_representation_evaluations_per_case": 2.0 + 2.0 * (k / len(rows)),
            "mean_verifier_calls_per_case": k / len(rows),
        })
    auc = 0.0
    for a, b in zip(points, points[1:]):
        auc += (b["widen_fraction"] - a["widen_fraction"]) * (a["accuracy"] + b["accuracy"]) / 2.0
    return {"points": points, "capability_vs_widen_auc": auc}


def rank_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, np.int64)
    order = np.argsort(-scores)
    positives = int(y.sum())
    if positives == 0:
        return {"benefit_cases": 0, "average_precision_like": None}
    hits = 0
    precisions = []
    for rank, idx in enumerate(order, start=1):
        if y[idx]:
            hits += 1
            precisions.append(hits / rank)
    return {
        "benefit_cases": positives,
        "average_precision_like": float(sum(precisions) / positives),
        "score_mean_benefit": float(scores[y == 1].mean()) if np.any(y == 1) else None,
        "score_mean_no_benefit": float(scores[y == 0].mean()) if np.any(y == 0) else None,
    }


def evaluate(
    model_dir: str,
    calibration_seeds: list[int],
    test_seeds: list[int],
    cases_per_family: int,
) -> dict[str, Any]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    started = time.time()
    scorer = AttackScorer(model_dir)
    scorer.reset_counters()

    calibration, calibration_spec = build_split(scorer, calibration_seeds, cases_per_family)
    test, test_spec = build_split(scorer, test_seeds, cases_per_family)

    Xtr = stack(calibration, "static")
    Xte = stack(test, "static")
    Str = stack(calibration, "sequence")
    Ste = stack(test, "sequence")
    ytr = targets(calibration)
    yte = targets(test)

    scores: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(SEED + 1)
    scores["random-quota"] = rng.random(len(test))
    scores["disagreement-heuristic"] = np.asarray([
        (10.0 if row["cheap_disagree"] else 0.0) - row["layout_margin"]
        for row in test
    ], np.float64)
    scores["linear-ridge"] = ridge_scores(Xtr, ytr, Xte, rank=16)

    Xtrz, Xtez = standardize(Xtr, Xte)
    scores["tiny-mlp"] = train_torch_scores(
        TinyMLP(Xtr.shape[1], 16), Xtrz, ytr, Xtez, seed=SEED + 2
    )

    flat_tr = Str.reshape(-1, Str.shape[-1])
    flat_te = Ste.reshape(-1, Ste.shape[-1])
    flat_trz, flat_tez = standardize(flat_tr, flat_te)
    Strz = flat_trz.reshape(Str.shape)
    Stez = flat_tez.reshape(Ste.shape)
    scores["tiny-gru"] = train_torch_scores(
        TinyGRU(Str.shape[-1], 16), Strz, ytr, Stez, seed=SEED + 3
    )

    random_cal = random_recurrent_features(Strz, seed=SEED + 4)
    random_test = random_recurrent_features(Stez, seed=SEED + 4)
    scores["random-recurrent-reservoir"] = ridge_scores(
        random_cal, ytr, random_test, rank=16
    )

    all_static = np.concatenate([Xtr, Xte], axis=0)
    projection = projection_fit(Xtr)
    real_X, real_rates, real_wall = reservoir_features(
        all_static, projection, kind="real", rewire_seed=None, gain=3.0, batch_size=8
    )
    split = len(Xtr)
    scores["real-malecns-temporal"] = ridge_scores(
        real_X[:split], ytr, real_X[split:], rank=16
    )

    rewire_scores = []
    rewire_meta = []
    for rep in range(3):
        rs = SEED + 10000 * (rep + 1)
        Xd, rates, wall = reservoir_features(
            all_static, projection, kind="degree", rewire_seed=rs, gain=3.0, batch_size=8
        )
        s = ridge_scores(Xd[:split], ytr, Xd[split:], rank=16)
        scores[f"degree-rewire-{rep+1}"] = s
        rewire_scores.append(s)
        rewire_meta.append({
            "replicate": rep + 1,
            "rewire_seed": rs,
            "wall_seconds": wall,
            "mean_rate": float(rates.mean()),
        })
    scores["degree-rewire-mean-score"] = np.mean(np.stack(rewire_scores), axis=0)

    curves = {name: curve(test, value) for name, value in scores.items()}
    ranking = {name: rank_metrics(yte, value) for name, value in scores.items()}

    direct_correct = sum(row["layout_correct"] for row in test)
    oracle_correct = sum(row["full_oracle"] for row in test)
    benefit_count = int(yte.sum())

    resources = scorer.counters()
    resources.update({
        "wall_seconds_total": time.time() - started,
        "male_cns_real_wall_seconds": real_wall,
        "male_cns_real_mean_rate": float(real_rates.mean()),
        "degree_rewires": rewire_meta,
        "controller_static_feature_dim": int(Xtr.shape[1]),
        "controller_sequence_shape": list(Str.shape[1:]),
    })

    return {
        "schema_version": RUN_VERSION,
        "status": "fresh-public-temporal-controller-calibration",
        "calibration": calibration_spec,
        "test": test_spec,
        "test_base": {
            "layout_correct": direct_correct,
            "layout_accuracy": direct_correct / len(test),
            "four_view_oracle_correct": oracle_correct,
            "four_view_oracle_coverage": oracle_correct / len(test),
            "recoverable_widen_benefit_cases": benefit_count,
            "recoverable_widen_benefit_fraction": benefit_count / len(test),
        },
        "controller_curves": curves,
        "controller_ranking": ranking,
        "fly_identity": {
            "brain_npz_sha256": BRAIN_SHA256,
            "weights_npz_sha256": WEIGHTS_SHA256,
            "connectome_weights_trainable": False,
        },
        "resources": resources,
        "rows": [{
            "uid": row["uid"],
            "seed": row["seed"],
            "case_id": row["case_id"],
            "family": row["family"],
            "difficulty": row["difficulty"],
            "layout_correct": row["layout_correct"],
            "full_oracle": row["full_oracle"],
            "benefit": row["benefit"],
            "cheap_disagree": row["cheap_disagree"],
            "layout_margin": row["layout_margin"],
            "predictions": row["predictions"],
        } for row in test],
        "claim_boundary": (
            "Fresh public procedural controller experiment. Controllers rank whether to spend "
            "additional representation and verifier budget; they never receive held-out correctness "
            "before action. The procedural verifier is an external tool and its calls are charged. "
            "Real MaleCNS topology earns explanatory credit only if it beats rewired and simpler "
            "controllers at matched widening budgets. No protected promotion or general connectome claim."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--calibration-seeds", default="20261028,20261029,20261030")
    ap.add_argument("--test-seeds", default="20261031,20261101")
    ap.add_argument("--cases-per-family", type=int, default=8)
    ap.add_argument("--output", default="run-024-controller.json")
    args = ap.parse_args()

    model_file = Path(args.model_file)
    actual = file_sha256(model_file)
    if actual != args.expected_sha256.lower():
        raise SystemExit(f"model SHA mismatch {actual}")

    data_dir = Path(os.environ.get("FLY_DATA", Path.home() / "fly-data"))
    if fly_sha256(data_dir / "brain.npz") != BRAIN_SHA256:
        raise SystemExit("MaleCNS brain digest mismatch")
    if fly_sha256(data_dir / "weights.npz") != WEIGHTS_SHA256:
        raise SystemExit("MaleCNS weights digest mismatch")

    result = evaluate(
        args.model_dir,
        [int(v) for v in args.calibration_seeds.split(",") if v.strip()],
        [int(v) for v in args.test_seeds.split(",") if v.strip()],
        args.cases_per_family,
    )
    result["actor"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
        "model_file_sha256": actual,
        "model_bytes": model_file.stat().st_size,
        "dtype": "float32",
        "device": "cpu",
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema_version": result["schema_version"],
        "test_base": result["test_base"],
        "controller_curves": {
            name: value for name, value in result["controller_curves"].items()
        },
        "controller_ranking": result["controller_ranking"],
        "resources": result["resources"],
        "claim_boundary": result["claim_boundary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
