#!/usr/bin/env python3
"""MaleCNS hierarchy-preserving null probe.

Uses the exact Run-014 temporal worker, but replaces the ordinary whole-graph
degree-preserving rewire with a stronger block-preserving null derived from the
official MaleCNS level-6 hSBM communities. Targets are permuted only among edges
with the same (source-community, target-community) pair. This preserves:

* every presynaptic neuron's out-degree and its number of edges to each community,
* the global target multiset / per-neuron in-degree,
* every edge weight attached to its presynaptic edge position,
* the full community-to-community edge-count matrix,
* the chosen 311-community partition (plus one explicit unassigned group),

while destroying finer neuron-to-neuron pairing inside each block.

The worker is non-authoritative. Helix Model/Helix must judge the receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import connectome_temporal_probe as base
from flybrain import FlyBrain

MALECNS_COMMIT = "67767d2233657983993ff6c2be48e836a935863c"
COMMUNITY_FILE = "mcns_lvl_6_hsbm_communities.feather"

_COMM: np.ndarray | None = None
_COMM_META: dict | None = None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_communities() -> tuple[np.ndarray, dict]:
    global _COMM, _COMM_META
    if _COMM is not None:
        return _COMM, _COMM_META  # type: ignore[return-value]

    data_dir = Path(os.environ["FLY_DATA"])
    community_path = Path(os.environ["MALECNS_COMMUNITIES"])
    ids = np.load(data_dir / "brain.npz")["ids"]
    df = pd.read_feather(community_path)

    body_to_comm: dict[int, int] = {}
    # Compact labels are independent of the published display ids.
    for compact, bodies in enumerate(df["bodyId"].tolist()):
        for body in bodies:
            body_to_comm[int(body)] = compact
    unassigned = len(df)
    comm = np.fromiter((body_to_comm.get(int(x), unassigned) for x in ids), dtype=np.int32, count=len(ids))

    _COMM = comm
    _COMM_META = {
        "source_commit": MALECNS_COMMIT,
        "community_file": COMMUNITY_FILE,
        "community_file_sha256": _sha256(community_path),
        "published_communities": int(len(df)),
        "effective_groups": int(len(df) + 1),
        "assigned_neurons": int(np.sum(comm != unassigned)),
        "unassigned_neurons": int(np.sum(comm == unassigned)),
        "assigned_fraction": float(np.mean(comm != unassigned)),
    }
    return _COMM, _COMM_META


def community_preserving_rewire(brain: FlyBrain, seed: int) -> None:
    """Permute targets only within source-community -> target-community blocks."""
    if brain.device != "cpu":
        raise ValueError("community null requires the canonical CPU sparse representation")
    comm, meta = load_communities()
    rng = np.random.default_rng(seed)

    original = brain.indices.copy()
    target_comm = comm[original]
    source_counts = np.diff(brain.indptr)
    source_comm = np.repeat(comm, source_counts)
    groups = int(meta["effective_groups"])
    key = source_comm.astype(np.int64) * groups + target_comm.astype(np.int64)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    cuts = np.flatnonzero(np.diff(sorted_key)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(order)]))
    for a, b in zip(starts, ends):
        pos = order[a:b]
        if len(pos) > 1:
            brain.indices[pos] = rng.permutation(original[pos])

    # Hard invariants. These are cheap relative to the sort and make the null auditable.
    if not np.array_equal(np.bincount(original, minlength=brain.n), np.bincount(brain.indices, minlength=brain.n)):
        raise RuntimeError("community rewire changed per-neuron in-degree")
    new_key = source_comm.astype(np.int64) * groups + comm[brain.indices].astype(np.int64)
    if not np.array_equal(key, new_key):
        raise RuntimeError("community rewire crossed a source/target community block")
    if not brain.sensory_input:
        sensory = np.char.find(brain.superclass.astype(str), "sensory") >= 0
        brain.weights = np.where(sensory[brain.indices], 0, brain.weights).astype(brain.weights.dtype)


def run_arm(*, dt: float, kind: str, replicate: int, groups: int, batch: int, sketch_width: int, seed: int) -> dict:
    original_rewire = base.rewire_degree_preserving
    try:
        if kind == "real":
            return base.run_condition(dt=dt, topology="real", rewire_seed=None,
                                      groups_n=groups, batch=batch, sketch_width=sketch_width, base_seed=seed)
        rewire_seed = seed + 10_000 * (replicate + 1)
        if kind == "degree":
            base.rewire_degree_preserving = original_rewire
            name = f"degree-rewire-{replicate+1}"
        elif kind == "community":
            base.rewire_degree_preserving = community_preserving_rewire
            name = f"community-rewire-{replicate+1}"
        else:
            raise ValueError(kind)
        return base.run_condition(dt=dt, topology=name, rewire_seed=rewire_seed,
                                  groups_n=groups, batch=batch, sketch_width=sketch_width, base_seed=seed)
    finally:
        base.rewire_degree_preserving = original_rewire


def summarize(rows: list[dict]) -> dict:
    out = []
    for dt in sorted({r["dt"] for r in rows}):
        real = next(r for r in rows if r["dt"] == dt and r["topology"] == "real")
        for delay in base.PROBE_DELAYS_S:
            key = str(delay)
            rm = real["result_by_delay"][key]
            record = {"dt": dt, "delay_s": delay, "real_accuracy": rm["accuracy"],
                      "real_margin": rm["centroid_margin"]}
            for kind, prefix in (("degree", "degree-rewire-"), ("community", "community-rewire-")):
                vals = [r["result_by_delay"][key] for r in rows if r["dt"] == dt and r["topology"].startswith(prefix)]
                acc = np.asarray([v["accuracy"] for v in vals], float)
                margin = np.asarray([v["centroid_margin"] for v in vals], float)
                record[f"{kind}_accuracy_mean"] = float(acc.mean())
                record[f"{kind}_accuracy_max"] = float(acc.max())
                record[f"real_minus_{kind}_accuracy_mean"] = float(rm["accuracy"] - acc.mean())
                record[f"real_beats_all_{kind}_accuracy"] = bool(rm["accuracy"] > acc.max())
                record[f"{kind}_margin_mean"] = float(margin.mean())
                record[f"real_minus_{kind}_margin_mean"] = float(rm["centroid_margin"] - margin.mean())
            out.append(record)
    return {
        "primary_endpoints": [
            "dt=0.020, delay=0.040 normalized four-way accuracy",
            "dt=0.002, delay=0.020 normalized four-way accuracy",
        ],
        "comparisons": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="connectome-hierarchy.json")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--sketch-width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--dts", nargs="+", type=float, default=[0.020, 0.002])
    args = ap.parse_args()

    _, community_meta = load_communities()
    rows = []
    t0 = time.perf_counter()
    for dt in args.dts:
        rows.append(run_arm(dt=dt, kind="real", replicate=0, groups=args.groups, batch=args.batch,
                            sketch_width=args.sketch_width, seed=args.seed))
        for kind in ("degree", "community"):
            for rep in range(args.replicates):
                rows.append(run_arm(dt=dt, kind=kind, replicate=rep, groups=args.groups, batch=args.batch,
                                    sketch_width=args.sketch_width, seed=args.seed))

    receipt = {
        "schema": "helix-model-connectome-hierarchy-probe-v1",
        "status": "worker-observation-not-authority",
        "question": "Does the MaleCNS transient-state advantage survive a null that preserves the official 311-community block organization?",
        "preregistered_before_result": {
            "primary": ["20ms dynamics at 40ms delay", "2ms dynamics at 20ms delay"],
            "readout": "per-sample L2-normalized topology-independent internal-state sketch; PCA16 + ridge(lambda=1)",
            "interpretation": {
                "community_matches_real": "published block organization is sufficient for most observed advantage",
                "community_matches_degree": "useful structure lies below the published block partition",
                "community_intermediate": "both hierarchy and finer wiring contribute",
            },
        },
        "community_null": community_meta,
        "parameters": vars(args),
        "rows": rows,
        "summary": summarize(rows),
        "wall_s": time.perf_counter() - t0,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print("HELIX_CONNECTOME_HIERARCHY " + json.dumps(receipt["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
