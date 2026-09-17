#!/usr/bin/env python3
"""Multi-fact interference probe for a Pollard FlyBrain checkpoint.

This extends Pollard's published single-fact verifier without changing the memory
mechanism. Each sample writes several fresh keyed secrets in early chunks, buries
them under filler, and queries one secret using the same continuation key. The
no-write floor uses the identical sequence. This is a worker observation only.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from pollard_flybrain import FlyBrain, load_backbone

WORD_LEN = 6
N_CHUNKS = 8
KEYS = (
    "amber", "cobalt", "dahlia", "ember", "fjord", "garnet", "hazel", "indigo",
    "juniper", "kelp", "lilac", "marble", "nectar", "onyx", "pebble", "quartz",
)


def fresh_word(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(WORD_LEN))


def build(tok, brain: FlyBrain, filler: str, fact_count: int, rng: random.Random):
    words = [fresh_word(rng) for _ in range(fact_count)]
    facts = " ".join(f"The {KEYS[i]} secret word is {words[i]}." for i in range(fact_count)) + " "
    target = rng.randrange(fact_count)
    tail = f" Question: what is the {KEYS[target]} secret word? Answer: The {KEYS[target]} secret word is"
    head = tok(facts, return_tensors="pt").input_ids
    qt = tok(tail, add_special_tokens=False, return_tensors="pt").input_ids
    budget = N_CHUNKS * brain.win - head.shape[1] - qt.shape[1]
    if budget < brain.win:
        raise RuntimeError(f"insufficient filler budget for {fact_count} facts")
    start = rng.randrange(0, max(len(filler) - 100_000, 1))
    fil = tok(filler[start:start + 100_000], add_special_tokens=False, return_tensors="pt").input_ids[:, :budget]
    ids = torch.cat([head, fil, qt], 1).to(brain.device)
    want = tok(" " + words[target], add_special_tokens=False).input_ids[:brain.span]
    return ids, want, target


def run_once(brain: FlyBrain, ids: torch.Tensor, *, use_brain: bool):
    brain.reset(1)
    raw = None
    for s0 in range(0, ids.shape[1], brain.win):
        ch = ids[:, s0:s0 + brain.win]
        if ch.shape[1]:
            _logits, raw = brain._chunk(ch, write=use_brain)
    return raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=15016)
    ap.add_argument("--counts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if max(args.counts) > len(KEYS):
        raise ValueError(f"fact count exceeds {len(KEYS)} frozen keys")

    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_backbone(args.model, torch.float32, args.device)
    brain = FlyBrain.load(args.brain, device=args.device).bind(model, tok, verbose=True)
    filler = Path(args.filler).read_text(encoding="utf-8", errors="replace")
    rng = random.Random(args.seed)

    rows = []
    for fact_count in args.counts:
        hit = floor = 0
        by_target = [0] * fact_count
        by_target_n = [0] * fact_count
        for _ in range(args.samples):
            ids, want, target = build(tok, brain, filler, fact_count, rng)
            raw = run_once(brain, ids, use_brain=True)
            pred = brain.decode(raw)[0].tolist()[:len(want)]
            ok = pred == want
            hit += int(ok)
            by_target[target] += int(ok)
            by_target_n[target] += 1

            raw0 = run_once(brain, ids, use_brain=False)
            pred0 = brain.decode(raw0)[0].tolist()[:len(want)]
            floor += int(pred0 == want)

        row = {
            "fact_count": fact_count,
            "samples": args.samples,
            "recall_accuracy": hit / args.samples,
            "floor_accuracy": floor / args.samples,
            "target_accuracy": [by_target[i] / by_target_n[i] if by_target_n[i] else None for i in range(fact_count)],
        }
        rows.append(row)
        print("MULTIFACT", json.dumps(row, sort_keys=True), flush=True)

    receipt = {
        "schema": "helix-model-run-015-multifact-v1",
        "status": "worker-observation-not-authority",
        "settings": {"samples": args.samples, "seed": args.seed, "counts": args.counts, "chunks": N_CHUNKS},
        "rows": rows,
        "state_bytes": brain.state_bytes,
        "wall_seconds": time.time() - started,
        "claim_boundary": "memory-selection/interference probe on one trained FlyBrain; topology attribution requires separately trained matched graph controls",
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
