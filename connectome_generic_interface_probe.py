#!/usr/bin/env python3
"""Generic-interface MaleCNS reservoir probe.

Reuses Run 014's controlled temporal worker but replaces biologically named LC
inputs with four fixed disjoint populations sampled from non-sensory neurons.
Because the same source neurons are used in real and degree-preserving rewired
brains, their source degrees and attached edge weights are identical. Only exact
who-connects-to-whom topology differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import connectome_temporal_probe as base

GENERIC_SEED = 0x48454C495847454E
POPULATION_SIZE = 128


def generic_input_cells(brain):
    eligible = np.ones(brain.n, dtype=bool)
    if brain.superclass is not None:
        eligible &= np.char.find(brain.superclass.astype(str), "sensory") < 0
    pool = np.flatnonzero(eligible)
    rng = np.random.default_rng(GENERIC_SEED)
    selected = rng.choice(pool, size=4 * POPULATION_SIZE, replace=False)
    groups = np.split(selected, 4)
    encoded = [(idx.astype(np.int64), 0.75) for idx in groups]
    return encoded, np.sort(selected.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="connectome-generic-interface.json")
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rewires", type=int, default=5)
    ap.add_argument("--sketch-width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=731)
    ap.add_argument("--dts", nargs="+", type=float, default=[0.020, 0.002])
    args = ap.parse_args()

    original_inputs = base.input_cells
    base.input_cells = generic_input_cells
    try:
        rows = []
        for dt in args.dts:
            rows.append(base.run_condition(dt=dt, topology="real", rewire_seed=None,
                                           groups_n=args.groups, batch=args.batch,
                                           sketch_width=args.sketch_width, base_seed=args.seed))
            for i in range(args.rewires):
                rs = args.seed + 10_000 * (i + 1)
                rows.append(base.run_condition(dt=dt, topology=f"rewire-{i+1}", rewire_seed=rs,
                                               groups_n=args.groups, batch=args.batch,
                                               sketch_width=args.sketch_width, base_seed=args.seed))
    finally:
        base.input_cells = original_inputs

    summary = base.summarize(rows)
    receipt = {
        "schema": "helix-model-connectome-generic-interface-v1",
        "status": "worker-observation-not-authority",
        "question": "Does real MaleCNS topology produce the transient-state advantage when the input interface is arbitrary rather than biologically named?",
        "preregistered_before_result": {
            "input": "four fixed disjoint sets of 128 non-sensory neurons selected by a frozen seed",
            "primary": ["dt=0.020 delay=0.040", "dt=0.002 delay=0.020"],
            "strong_control": "five degree-preserving target rewires with identical input neurons, source degrees, edge weights, labels and noise schedule",
            "interpretation": "survival supports a more generic reservoir effect; collapse means the earlier effect is interface/pathway-specific",
        },
        "generic_seed": GENERIC_SEED,
        "population_size": POPULATION_SIZE,
        "parameters": vars(args),
        "rows": rows,
        "summary": summary,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print("HELIX_CONNECTOME_GENERIC " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
