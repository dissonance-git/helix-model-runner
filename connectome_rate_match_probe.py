#!/usr/bin/env python3
"""Held-out firing-rate-matched degree-null for MaleCNS Run 014.

This test addresses the main remaining confound from the normalized temporal grid:
real MaleCNS wiring operates at a higher mean firing rate than degree-preserving
rewires under the same global synaptic gain. We tune only the rewire's global gain
against a separate calibration seed, never against readout accuracy, then evaluate
on the frozen Run-014 task with a disjoint seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

import connectome_temporal_probe as base


def run_with_gain(*, gain: float, dt: float, topology: str, rewire_seed: int | None,
                  groups: int, batch: int, sketch_width: int, seed: int) -> dict:
    old = FlyBrain.gain
    try:
        FlyBrain.gain = float(gain)
        row = base.run_condition(dt=dt, topology=topology, rewire_seed=rewire_seed,
                                 groups_n=groups, batch=batch, sketch_width=sketch_width,
                                 base_seed=seed)
        row["global_gain"] = float(gain)
        return row
    finally:
        FlyBrain.gain = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="connectome-rate-match.json")
    ap.add_argument("--rewires", type=int, default=5)
    ap.add_argument("--groups", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sketch-width", type=int, default=512)
    args = ap.parse_args()

    dt = 0.020
    calibration_seed = 9091
    evaluation_seed = 731
    calibration_groups = 4
    gain_grid = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)

    # Target comes only from firing rate on the separate calibration seed.
    real_cal = run_with_gain(gain=3.0, dt=dt, topology="real-calibration", rewire_seed=None,
                             groups=calibration_groups, batch=args.batch,
                             sketch_width=128, seed=calibration_seed)
    target_rate = real_cal["mean_neuron_firing_hz"]

    selected = []
    calibration = []
    for i in range(args.rewires):
        rs = 10_000 * (i + 1) + evaluation_seed
        candidates = []
        for gain in gain_grid:
            row = run_with_gain(gain=gain, dt=dt, topology=f"rewire-{i+1}-calibration",
                                rewire_seed=rs, groups=calibration_groups, batch=args.batch,
                                sketch_width=128, seed=calibration_seed)
            candidates.append({"gain": gain, "rate": row["mean_neuron_firing_hz"]})
        best = min(candidates, key=lambda x: abs(x["rate"] - target_rate))
        selected.append(float(best["gain"]))
        calibration.append({"rewire": i + 1, "candidates": candidates, "selected": best})

    # Evaluation uses the original fixed Run-014 seed and task packet. Accuracy
    # never participates in gain selection.
    real = run_with_gain(gain=3.0, dt=dt, topology="real", rewire_seed=None,
                         groups=args.groups, batch=args.batch,
                         sketch_width=args.sketch_width, seed=evaluation_seed)
    controls = []
    for i, gain in enumerate(selected):
        rs = 10_000 * (i + 1) + evaluation_seed
        controls.append(run_with_gain(gain=gain, dt=dt,
                                      topology=f"rate-matched-rewire-{i+1}",
                                      rewire_seed=rs, groups=args.groups, batch=args.batch,
                                      sketch_width=args.sketch_width, seed=evaluation_seed))

    rows = [real, *controls]
    summary = base.summarize(rows)
    # summarize() recognizes topology names beginning rewire-, so construct the
    # primary comparison explicitly for rate-matched names.
    delay_key = "0.04"
    rm = real["result_by_delay"][delay_key]
    vals = [r["result_by_delay"][delay_key] for r in controls]
    acc = np.asarray([v["accuracy"] for v in vals], float)
    rates = np.asarray([r["mean_neuron_firing_hz"] for r in controls], float)
    primary = {
        "delay_s": 0.04,
        "real_accuracy": rm["accuracy"],
        "rate_matched_accuracy_mean": float(acc.mean()),
        "rate_matched_accuracy_max": float(acc.max()),
        "real_minus_rate_matched_accuracy_mean": float(rm["accuracy"] - acc.mean()),
        "real_beats_all_rate_matched": bool(rm["accuracy"] > acc.max()),
        "real_rate": real["mean_neuron_firing_hz"],
        "rate_matched_rate_mean": float(rates.mean()),
        "rate_matched_rate_range": [float(rates.min()), float(rates.max())],
    }

    receipt = {
        "schema": "helix-model-connectome-rate-matched-null-v1",
        "status": "worker-observation-not-authority",
        "question": "Does the 20ms/40ms MaleCNS transient advantage survive when degree-preserving rewires are globally gain-tuned to match real-graph firing rate?",
        "preregistered_before_result": {
            "calibration_only_metric": "mean neuron firing Hz on seed 9091",
            "gain_grid": list(gain_grid),
            "evaluation_seed": evaluation_seed,
            "primary": "normalized four-way accuracy at 40ms delay",
            "interpretation": "survival rules out mean global activity level as a sufficient explanation; collapse means operating point is sufficient under this test",
        },
        "target_calibration_rate": target_rate,
        "calibration": calibration,
        "selected_gains": selected,
        "rows": rows,
        "primary": primary,
        "secondary": summary,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print("HELIX_CONNECTOME_RATE_MATCH " + json.dumps(primary, sort_keys=True))


if __name__ == "__main__":
    main()
