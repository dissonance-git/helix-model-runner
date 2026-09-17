#!/usr/bin/env python3
"""Bounded MaleCNS topology x dynamics probe for Helix Model.

Question
--------
Does the real MaleCNS topology retain task-relevant state better than strong
matched rewires, and is any advantage specific to a temporal regime?

The biological graph is never trained. Four known visual-projection populations
are used only as an input interface. A topology-independent signed hash sketch of
all non-input spikes is the readout surface, so the experiment tests the frozen
connectome as a general recurrent reservoir rather than asking only whether its
native motor outputs respond.

Strong control
--------------
The rewire is the fly.ai control: each presynaptic neuron keeps its number of
outgoing edges and every edge keeps its weight, while the multiset of postsynaptic
targets is permuted. Consequently every neuron also keeps its incoming edge count.
Only who-connects-to-whom changes.

No result produced here is model or Helix claim authority. The output is a compact
worker receipt for later ingestion and external judgment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from flybrain import FlyBrain
import flybrain


DONOR_COMMIT = "fcf825d1f506db698a84850b005657df6ec8df3c"
BRAIN_SHA256 = "cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca"
WEIGHTS_SHA256 = "c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c"

CONTEXTS = (
    ("mate-left", (("LC10a",), "L", 0.70)),
    ("mate-right", (("LC10a",), "R", 0.70)),
    ("threat-left", (("LC4", "LPLC2"), "L", 0.80)),
    ("threat-right", (("LC4", "LPLC2"), "R", 0.80)),
)


@dataclass(frozen=True)
class Timing:
    warmup_s: float = 0.10
    stimulus_s: float = 0.10
    delay_s: float = 0.20
    bin_s: float = 0.020


TIMING = Timing()
PROBE_DELAYS_S = (0.0, 0.04, 0.10, 0.20)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rewire_degree_preserving(brain: FlyBrain, seed: int) -> None:
    """Exact control used by fly.ai/flytalk.py, CPU path."""
    if brain.device != "cpu":
        raise ValueError("probe requires CPU so the control mutates one canonical sparse representation")
    rng = np.random.default_rng(seed)
    brain.indices = rng.permutation(brain.indices).astype(brain.indices.dtype)
    if not brain.sensory_input:
        sensory = np.char.find(brain.superclass.astype(str), "sensory") >= 0
        brain.weights = np.where(sensory[brain.indices], 0, brain.weights).astype(brain.weights.dtype)


def input_cells(brain: FlyBrain) -> tuple[list[tuple[np.ndarray, float]], np.ndarray]:
    encoded = []
    all_idx = []
    for _, (types, side, amount) in CONTEXTS:
        idx = brain.cells(list(types), side=side)
        if len(idx) == 0:
            raise RuntimeError(f"empty encoder population: types={types} side={side}")
        encoded.append((idx, amount))
        all_idx.append(idx)
    return encoded, np.unique(np.concatenate(all_idx))


def make_sketch(n: int, excluded: np.ndarray, width: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Topology-independent signed feature hashing over neuron ids."""
    # SplitMix64-like integer mixing gives a deterministic, cheap projection with
    # no dependency on graph structure. Input populations are excluded so the
    # post-stimulus task cannot be solved by directly reading the driven cells.
    ids = np.arange(n, dtype=np.uint64)
    z = ids + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    bins = (z % np.uint64(width)).astype(np.int64)
    signs = np.where((z >> np.uint64(63)) == 0, 1.0, -1.0).astype(np.float32)
    signs[np.asarray(excluded, dtype=np.int64)] = 0.0
    return bins, signs


def sketch_fired(fired: np.ndarray, bins: np.ndarray, signs: np.ndarray, width: int) -> np.ndarray:
    if len(fired) == 0:
        return np.zeros(width, np.float32)
    w = signs[fired]
    keep = w != 0
    if not np.any(keep):
        return np.zeros(width, np.float32)
    return np.bincount(bins[fired[keep]], weights=w[keep], minlength=width).astype(np.float32)


def fit_predict_ridge(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float = 1.0, rank: int = 16) -> np.ndarray:
    """Fixed PCA + ridge multiclass readout; no hyperparameter tuning on results."""
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    # n_samples is deliberately small; economy SVD is cheap and deterministic.
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = max(1, min(rank, len(vt), Xtr.shape[0] - 1))
    P = vt[:k].T
    Ztr = Xc @ P
    scale = Ztr.std(0) + 1e-6
    Ztr = Ztr / scale
    Zte = ((Xte - mu) @ P) / scale
    classes = int(ytr.max()) + 1
    Y = np.eye(classes, dtype=np.float64)[ytr]
    zm, ym = Ztr.mean(0), Y.mean(0)
    Zc2 = Ztr - zm
    W = np.linalg.solve(Zc2.T @ Zc2 + lam * len(ytr) * np.eye(k), Zc2.T @ (Y - ym))
    b = ym - zm @ W
    return np.argmax(Zte @ W + b, axis=1)


def grouped_cv_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    correct = total = 0
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        pred = fit_predict_ridge(X[tr], y[tr], X[te])
        correct += int(np.sum(pred == y[te]))
        total += int(np.sum(te))
    return correct / total if total else float("nan")


def centroid_margin(X: np.ndarray, y: np.ndarray) -> float:
    """Between-class centroid spread divided by within-class RMS spread."""
    centers = np.stack([X[y == c].mean(0) for c in range(len(CONTEXTS))])
    global_center = X.mean(0)
    between = float(np.mean(np.sum((centers - global_center) ** 2, axis=1)))
    within_terms = []
    for c in range(len(CONTEXTS)):
        d = X[y == c] - centers[c]
        within_terms.extend(np.sum(d * d, axis=1).tolist())
    within = float(np.mean(within_terms)) if within_terms else 0.0
    return between / (within + 1e-12)


def steps(seconds: float, dt: float) -> int:
    return max(1, int(round(seconds / dt)))


def run_condition(*, dt: float, topology: str, rewire_seed: int | None, groups_n: int,
                  batch: int, sketch_width: int, base_seed: int) -> dict:
    t0 = time.perf_counter()
    brain = FlyBrain(device="cpu", batch=batch, seed=base_seed, dt=dt,
                     sensory_input=False, refractory=max(dt, 0.004) if dt < 0.020 else 0.0)
    if rewire_seed is not None:
        rewire_degree_preserving(brain, rewire_seed)
    encoders, input_idx = input_cells(brain)
    bins, signs = make_sketch(brain.n, input_idx, sketch_width, seed=0x48454C4958)

    warm = steps(TIMING.warmup_s, dt)
    stim = steps(TIMING.stimulus_s, dt)
    total = warm + stim + steps(TIMING.delay_s, dt)
    bin_steps = steps(TIMING.bin_s, dt)
    probe_ends = {
        delay: warm + stim + int(round(delay / dt))
        for delay in PROBE_DELAYS_S
    }
    # delay=0 uses the last bin of the stimulus window.
    probe_ends[0.0] = warm + stim

    by_delay: dict[float, list[np.ndarray]] = {d: [] for d in PROBE_DELAYS_S}
    labels: list[int] = []
    trial_groups: list[int] = []
    firing_rates = []
    rng = np.random.default_rng(base_seed + int(round(dt * 1e6)) + (rewire_seed or 0))

    for group in range(groups_n):
        # Batch is balanced whenever divisible by number of classes; otherwise
        # resize then permute. All topologies see the same label construction.
        yb = rng.permutation(np.resize(np.arange(len(CONTEXTS)), batch))
        labels.extend(yb.tolist())
        trial_groups.extend([group] * batch)
        brain.reset(base_seed * 100_000 + group)
        # Rolling 20ms activity window for each independent fly.
        rolling = [np.zeros((bin_steps, sketch_width), np.float32) for _ in range(batch)]
        rolling_pos = 0
        spike_total = np.zeros(batch, np.int64)

        for s in range(total):
            inject = []
            if warm <= s < warm + stim:
                for c, (idx, amount) in enumerate(encoders):
                    drive = (yb == c).astype(np.float32) * np.float32(amount)
                    if np.any(drive):
                        inject.append((idx, drive))
            fired = brain.step(inject=inject)
            # FlyBrain returns list[array] for batch > 1.
            for b, fb in enumerate(fired):
                spike_total[b] += len(fb)
                rolling[b][rolling_pos] = sketch_fired(fb, bins, signs, sketch_width)
            rolling_pos = (rolling_pos + 1) % bin_steps
            end_step = s + 1
            for delay, probe_end in probe_ends.items():
                if end_step == probe_end:
                    for b in range(batch):
                        by_delay[delay].append(rolling[b].sum(axis=0).copy())

        firing_rates.extend((spike_total / (total * dt * brain.n)).tolist())

    y = np.asarray(labels, np.int64)
    g = np.asarray(trial_groups, np.int64)
    results = {}
    for delay in PROBE_DELAYS_S:
        X = np.asarray(by_delay[delay], np.float32)
        if len(X) != len(y):
            raise RuntimeError(f"snapshot count mismatch at delay {delay}: {len(X)} != {len(y)}")
        results[str(delay)] = {
            "accuracy": grouped_cv_accuracy(X, y, g),
            "centroid_margin": centroid_margin(X, y),
            "feature_rms": float(np.sqrt(np.mean(X * X))),
            "nonzero_fraction": float(np.count_nonzero(X) / X.size),
        }

    return {
        "dt": dt,
        "topology": topology,
        "rewire_seed": rewire_seed,
        "samples": len(y),
        "classes": [x[0] for x in CONTEXTS],
        "readout": "fixed signed-hash internal-spike sketch + fixed PCA16/ridge(lambda=1)",
        "probe_delays_s": list(PROBE_DELAYS_S),
        "mean_neuron_firing_hz": float(np.mean(firing_rates)),
        "result_by_delay": results,
        "wall_s": time.perf_counter() - t0,
    }


def summarize(rows: list[dict]) -> dict:
    real = {(r["dt"], d): m for r in rows if r["topology"] == "real" for d, m in r["result_by_delay"].items()}
    shuffled: dict[tuple[float, str], list[dict]] = {}
    for r in rows:
        if r["topology"].startswith("rewire"):
            for d, m in r["result_by_delay"].items():
                shuffled.setdefault((r["dt"], d), []).append(m)
    comparisons = []
    for key, rm in sorted(real.items()):
        controls = shuffled.get(key, [])
        if not controls:
            continue
        acc = np.array([x["accuracy"] for x in controls], dtype=float)
        margin = np.array([x["centroid_margin"] for x in controls], dtype=float)
        comparisons.append({
            "dt": key[0],
            "delay_s": float(key[1]),
            "real_accuracy": rm["accuracy"],
            "rewire_accuracy_mean": float(acc.mean()),
            "rewire_accuracy_min": float(acc.min()),
            "rewire_accuracy_max": float(acc.max()),
            "real_minus_rewire_accuracy_mean": float(rm["accuracy"] - acc.mean()),
            "real_beats_all_rewires_accuracy": bool(rm["accuracy"] > acc.max()),
            "real_centroid_margin": rm["centroid_margin"],
            "rewire_centroid_margin_mean": float(margin.mean()),
            "real_minus_rewire_margin_mean": float(rm["centroid_margin"] - margin.mean()),
        })
    # Primary endpoint was fixed before execution: delayed (100 ms) decoding.
    primary = [x for x in comparisons if math.isclose(x["delay_s"], 0.10, abs_tol=1e-9)]
    return {
        "primary_endpoint": "100ms delayed four-way grouped-CV accuracy; real topology versus mean and max of degree-preserving rewires",
        "comparisons": comparisons,
        "primary": primary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="connectome-temporal-receipt.json")
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sketch-width", type=int, default=512)
    ap.add_argument("--rewires", type=int, default=3)
    ap.add_argument("--dts", nargs="+", type=float, default=[0.020, 0.005, 0.002])
    ap.add_argument("--seed", type=int, default=731)
    args = ap.parse_args()

    data_dir = Path(os.environ.get("FLY_DATA", Path.home() / "fly-data"))
    # Force one real load first; ensure_data performs donor SHA checks. We repeat
    # the digests in the receipt so the worker output binds to exact bytes.
    warm = FlyBrain(device="cpu", batch=1, seed=args.seed, dt=args.dts[0], sensory_input=False)
    del warm
    identities = {
        "flyai_commit": DONOR_COMMIT,
        "flybrain_version": getattr(flybrain, "__version__", "0.1.0-source"),
        "brain_npz_sha256": sha256(data_dir / "brain.npz"),
        "weights_npz_sha256": sha256(data_dir / "weights.npz"),
        "expected_brain_sha256": BRAIN_SHA256,
        "expected_weights_sha256": WEIGHTS_SHA256,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    if identities["brain_npz_sha256"] != BRAIN_SHA256 or identities["weights_npz_sha256"] != WEIGHTS_SHA256:
        raise RuntimeError("MaleCNS processed artifact digest mismatch")

    rows = []
    for dt in args.dts:
        rows.append(run_condition(dt=dt, topology="real", rewire_seed=None,
                                  groups_n=args.groups, batch=args.batch,
                                  sketch_width=args.sketch_width, base_seed=args.seed))
        for i in range(args.rewires):
            rs = args.seed + 10_000 * (i + 1)
            rows.append(run_condition(dt=dt, topology=f"rewire-{i+1}", rewire_seed=rs,
                                      groups_n=args.groups, batch=args.batch,
                                      sketch_width=args.sketch_width, base_seed=args.seed))

    receipt = {
        "schema": "helix-model-connectome-temporal-probe-v1",
        "status": "worker-observation-not-authority",
        "question": "Does real MaleCNS topology preserve task-relevant state better than degree-preserving rewires, and is the difference dynamics/timescale dependent?",
        "preregistered_before_result": {
            "primary_metric": "four-way grouped-CV accuracy at 100 ms post-stimulus",
            "strong_control": "three degree-preserving rewires preserving presynaptic edge counts, edge weights, and postsynaptic edge-count multiset",
            "secondary_metrics": ["accuracy at 0/40/200 ms", "centroid separation margin", "activity sparsity/rate"],
            "interpretation_gate": "a topology claim requires the real graph to beat strong rewires; a timescale-local effect is retained as such, not generalized",
        },
        "timing": TIMING.__dict__,
        "identities": identities,
        "parameters": vars(args),
        "rows": rows,
        "summary": summarize(rows),
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print("HELIX_CONNECTOME_TEMPORAL " + json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
