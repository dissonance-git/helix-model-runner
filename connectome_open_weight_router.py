#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from flybrain import FlyBrain

from probe import randomized_low_rank
from connectome_temporal_probe import (
    BRAIN_SHA256, DONOR_COMMIT, WEIGHTS_SHA256,
    make_sketch, rewire_degree_preserving, sha256, sketch_fired,
)
from connectome_hierarchy_probe import MALECNS_COMMIT, community_preserving_rewire, load_communities

SCHEMA = "helix-model-run-016-connectome-router-v1"
SEED = 16016
TARGET = "model.layers.15.self_attn.o_proj"
TARGET_WEIGHT = TARGET + ".weight"
LESION_RANK = 32
REPAIR_FRACTION = 0.01
ROUTER_RANK = 16
DIAGNOSTIC_RANKS = (1, 2, 4, 8, 16)
DT = 0.020
WARM_STEPS, STIM_STEPS, DELAY_STEPS = 5, 5, 2
INPUT_POPULATIONS, INPUT_POP_SIZE, SKETCH_WIDTH = 4, 128, 512
REWIRE_REPLICATES = 3
RATE_GAIN_GRID = (3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0)
FAMILIES = ("arithmetic", "translation", "python", "factual")
PROMPTS = {
    "arithmetic": [
        "Compute 7 + 11. Answer:", "Compute 13 + 29. Answer:", "Compute 24 + 18. Answer:",
        "Compute 31 + 9. Answer:", "Compute 45 + 27. Answer:", "Compute 52 + 16. Answer:",
        "Compute 61 + 22. Answer:", "Compute 74 + 19. Answer:", "Compute 83 + 17. Answer:",
        "Compute 96 + 25. Answer:", "Compute 38 + 47. Answer:", "Compute 59 + 34. Answer:",
    ],
    "translation": [
        'Translate into Spanish: "the red tree". Translation:', 'Translate into Spanish: "a quiet room". Translation:',
        'Translate into Spanish: "the old bridge". Translation:', 'Translate into Spanish: "a small black cat". Translation:',
        'Translate into Spanish: "the cold morning". Translation:', 'Translate into Spanish: "we read the book". Translation:',
        'Translate into Spanish: "the blue river". Translation:', 'Translate into Spanish: "a dark forest". Translation:',
        'Translate into Spanish: "the open window". Translation:', 'Translate into Spanish: "they walk slowly". Translation:',
        'Translate into Spanish: "a warm light". Translation:', 'Translate into Spanish: "the distant moon". Translation:',
    ],
    "python": [
        "In Python, write an expression that squares x. Code:", "In Python, get the length of items. Code:",
        "In Python, join words with a space. Code:", "In Python, test whether n is even. Code:",
        "In Python, take the first three values of xs. Code:", "In Python, convert text to lowercase. Code:",
        "In Python, sum the values in nums. Code:", "In Python, reverse the list xs. Code:",
        "In Python, test whether key is in mapping. Code:", "In Python, get the maximum of values. Code:",
        "In Python, make a set from items. Code:", "In Python, remove surrounding whitespace from text. Code:",
    ],
    "factual": [
        "Complete the factual statement: The capital of France is", "Complete the factual statement: The capital of Japan is",
        "Complete the factual statement: The capital of Canada is", "Complete the factual statement: The capital of Brazil is",
        "Complete the factual statement: The capital of Italy is", "Complete the factual statement: The capital of Egypt is",
        "Complete the factual statement: The capital of Spain is", "Complete the factual statement: The capital of Australia is",
        "Complete the factual statement: The capital of Peru is", "Complete the factual statement: The capital of Norway is",
        "Complete the factual statement: The capital of Kenya is", "Complete the factual statement: The capital of Thailand is",
    ],
}

class ProbeError(RuntimeError): pass

def prompt_packet():
    cal, held = [], []
    for fid, family in enumerate(FAMILIES):
        rows = PROMPTS[family]
        if len(rows) != 12: raise ProbeError(f"{family} must have 12 prompts")
        cal.extend((x, fid) for x in rows[:6]); held.extend((x, fid) for x in rows[6:])
    return cal, held

def file_sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def tokenize(tok, prompts):
    return tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=96, add_special_tokens=True)

def get_weight(model):
    p = dict(model.named_parameters()).get(TARGET_WEIGHT)
    if p is None or p.ndim != 2: raise ProbeError(f"missing 2-D target {TARGET_WEIGHT}")
    return p

def get_module(model):
    m = dict(model.named_modules()).get(TARGET)
    if m is None: raise ProbeError(f"missing module {TARGET}")
    return m

def final_logits_and_rep(model, batch, capture_module=None):
    captured = []
    hook = capture_module.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().cpu().float())) if capture_module else None
    try:
        with torch.inference_mode():
            out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
    finally:
        if hook: hook.remove()
    mask = batch["attention_mask"].bool(); final = mask.sum(dim=1) - 1; bi = torch.arange(mask.shape[0])
    logits = out.logits[bi, final].detach().cpu().float()
    rep = out.hidden_states[-1][bi, final].detach().cpu().float().numpy()
    return logits, rep, (captured[0].numpy() if captured else None)

def final_logits(model, batch):
    with torch.inference_mode(): out = model(**batch, use_cache=False, return_dict=True)
    mask = batch["attention_mask"].bool(); final = mask.sum(dim=1) - 1; bi = torch.arange(mask.shape[0])
    return out.logits[bi, final].detach().cpu().float()

def per_sample_metrics(base, cand):
    bl = F.log_softmax(base, dim=-1); cl = F.log_softmax(cand, dim=-1); bp = bl.exp()
    kl = torch.sum(bp * (bl - cl), dim=-1); mse = torch.mean((base - cand) ** 2, dim=-1)
    k = min(10, int(base.shape[-1]), int(cand.shape[-1]))
    bt = torch.topk(base, k, dim=-1).indices; ct = torch.topk(cand, k, dim=-1).indices
    overlap = (bt.unsqueeze(-1) == ct.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / float(k)
    return {"kl": kl.numpy(), "mse": mse.numpy(), "top10": overlap.numpy()}

def repair_bank(original, low_rank, token_inputs, token_family):
    residual = (original - low_rank).astype(np.float32, copy=False); count = max(1, int(round(original.size * REPAIR_FRACTION)))
    banks, meta = [], []
    for fid, family in enumerate(FAMILIES):
        x = token_inputs[token_family == fid]
        if len(x) == 0: raise ProbeError(f"no calibration token activations for {family}")
        activity = np.mean(np.abs(x), axis=0).astype(np.float32)
        flat = (np.abs(residual) * (activity[None, :] + np.float32(1e-8))).ravel()
        idx = np.argpartition(flat, flat.size - count)[-count:]; idx = np.sort(idx.astype(np.int64, copy=False))
        candidate = low_rank.copy(); candidate.ravel()[idx] += residual.ravel()[idx]
        banks.append(candidate.astype(np.float32, copy=False))
        meta.append({"family": family, "selected_coordinates": int(idx.size), "support_sha256": hashlib.sha256(idx.tobytes()).hexdigest()})
    return banks, meta

def classifier_predict(Xtr, ytr, Xte, rank):
    Xtr = np.asarray(Xtr, np.float64); Xte = np.asarray(Xte, np.float64); ytr = np.asarray(ytr, np.int64)
    mu = Xtr.mean(0); xc = Xtr - mu; _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = max(1, min(int(rank), len(vt), Xtr.shape[0] - 1)); P = vt[:k].T
    ztr = xc @ P; scale = ztr.std(0) + 1e-6; ztr /= scale; zte = ((Xte - mu) @ P) / scale
    Y = np.eye(4, dtype=np.float64)[ytr]; zm = ztr.mean(0); ym = Y.mean(0); zc = ztr - zm
    W = np.linalg.solve(zc.T @ zc + float(len(ytr)) * np.eye(k), zc.T @ (Y - ym)); b = ym - zm @ W
    return np.argmax(zte @ W + b, axis=1).astype(np.int64)

def frozen_input_populations(brain):
    eligible = np.flatnonzero(np.char.find(brain.superclass.astype(str), "sensory") < 0)
    pick = np.random.default_rng(SEED + 1).choice(eligible, size=INPUT_POPULATIONS * INPUT_POP_SIZE, replace=False)
    return [np.sort(pick[i * INPUT_POP_SIZE:(i + 1) * INPUT_POP_SIZE]).astype(np.int64) for i in range(INPUT_POPULATIONS)]

def projection_fit(cal_rep):
    rng = np.random.default_rng(SEED + 2); d = cal_rep.shape[1]
    P = rng.normal(size=(STIM_STEPS * INPUT_POPULATIONS, d)).astype(np.float32); P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-8
    center = cal_rep.mean(0).astype(np.float32); raw = (cal_rep - center) @ P.T
    return P, center, raw.mean(0).astype(np.float32), (raw.std(0) + 1e-4).astype(np.float32)

def projection_schedule(rep, params):
    P, center, mean, scale = params; z = ((rep - center) @ P.T - mean) / scale; z = np.clip(z, -8.0, 8.0)
    return (0.15 + 0.70 / (1.0 + np.exp(-z))).reshape(len(rep), STIM_STEPS, INPUT_POPULATIONS).astype(np.float32)

def apply_topology(brain, kind, rewire_seed):
    if kind == "real": return
    if rewire_seed is None: raise ProbeError(f"{kind} needs rewire seed")
    if kind in ("degree", "rate"): rewire_degree_preserving(brain, rewire_seed)
    elif kind == "community": community_preserving_rewire(brain, rewire_seed)
    else: raise ProbeError(kind)

def reservoir_features(rep, projection_params, *, kind, rewire_seed, gain, batch_size=8):
    if len(rep) % batch_size: raise ProbeError("sample count must divide reservoir batch size")
    old_gain = FlyBrain.gain; t0 = time.perf_counter()
    try:
        FlyBrain.gain = float(gain); brain = FlyBrain(device="cpu", batch=batch_size, seed=SEED, dt=DT, sensory_input=False)
        apply_topology(brain, kind, rewire_seed); pops = frozen_input_populations(brain); excluded = np.unique(np.concatenate(pops))
        bins, signs = make_sketch(brain.n, excluded, SKETCH_WIDTH, seed=0x48454C4958); schedules = projection_schedule(rep, projection_params)
        rows, rates = [], []; total_steps = WARM_STEPS + STIM_STEPS + DELAY_STEPS
        for group, start in enumerate(range(0, len(rep), batch_size)):
            brain.reset(SEED * 100000 + group); spike_total = np.zeros(batch_size, np.int64); final_sketch = [np.zeros(SKETCH_WIDTH, np.float32) for _ in range(batch_size)]
            for step in range(total_steps):
                inject = []
                if WARM_STEPS <= step < WARM_STEPS + STIM_STEPS:
                    si = step - WARM_STEPS
                    inject = [(idx, schedules[start:start + batch_size, si, pid]) for pid, idx in enumerate(pops)]
                fired = brain.step(inject=inject)
                for b, fb in enumerate(fired):
                    spike_total[b] += len(fb)
                    if step == total_steps - 1: final_sketch[b] = sketch_fired(fb, bins, signs, SKETCH_WIDTH)
            rows.extend(final_sketch); rates.extend((spike_total / (total_steps * DT * brain.n)).astype(float).tolist())
        X = np.asarray(rows, np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-6
        return X, np.asarray(rates, np.float64), time.perf_counter() - t0
    finally: FlyBrain.gain = old_gain

def route_metrics(pred, oracle, mats):
    pred = np.asarray(pred, np.int64); oracle = np.asarray(oracle, np.int64); i = np.arange(len(pred))
    return {"routing_accuracy": float(np.mean(pred == oracle)), "next_token_kl": float(np.mean(mats["kl"][pred, i])), "logit_mse": float(np.mean(mats["mse"][pred, i])), "top10_overlap": float(np.mean(mats["top10"][pred, i])), "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()}

def aggregate(rows):
    out = {"replicates": rows}
    for field in ("routing_accuracy", "next_token_kl", "logit_mse", "top10_overlap"):
        v = np.asarray([x[field] for x in rows], float); out[field + "_mean"] = float(v.mean()); out[field + "_min"] = float(v.min()); out[field + "_max"] = float(v.max())
    return out

def self_test():
    rng = np.random.default_rng(5); X = rng.normal(size=(24, 30)); y = np.tile(np.arange(4), 6)
    if classifier_predict(X, y, X, 4).shape != y.shape: raise ProbeError("classifier self-test failed")
    m = per_sample_metrics(torch.tensor([[2.0, 1.0]]), torch.tensor([[2.0, 1.0]]))
    if not np.allclose(m["kl"], 0): raise ProbeError("metric self-test failed")
    print("SELF_TEST_OK")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir"); ap.add_argument("--model-file"); ap.add_argument("--expected-sha256"); ap.add_argument("--output", default="run-016-router.json"); ap.add_argument("--self-test", action="store_true"); args = ap.parse_args()
    if args.self_test: self_test(); return 0
    if not args.model_dir or not args.model_file or not args.expected_sha256: raise ProbeError("model arguments required")
    started = time.time(); model_file = Path(args.model_file); actual_sha = file_sha256(model_file)
    if actual_sha != args.expected_sha256.lower(): raise ProbeError(f"model SHA mismatch {actual_sha}")
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    calibration, heldout = prompt_packet(); cal_text = [x[0] for x in calibration]; held_text = [x[0] for x in heldout]
    cal_family = np.asarray([x[1] for x in calibration], np.int64); held_family = np.asarray([x[1] for x in heldout], np.int64)
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"; cal_batch = tokenize(tok, cal_text); held_batch = tokenize(tok, held_text)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    target_param = get_weight(model); target_module = get_module(model); original_t = target_param.detach().cpu().float().clone(); original = original_t.numpy().copy()
    base_cal, rep_cal, cap_cal = final_logits_and_rep(model, cal_batch, target_module); base_held, rep_held, _ = final_logits_and_rep(model, held_batch)
    if cap_cal is None: raise ProbeError("activation capture failed")
    mask = cal_batch["attention_mask"].bool().cpu().numpy(); family_tokens = np.broadcast_to(cal_family[:, None], mask.shape)[mask]; token_inputs = cap_cal[mask]
    low_rank = randomized_low_rank(original, LESION_RANK, seed=SEED + LESION_RANK); banks, bank_meta = repair_bank(original, low_rank, token_inputs, family_tokens)
    cal_candidates, held_candidates = [], []
    try:
        with torch.no_grad(): target_param.copy_(torch.from_numpy(low_rank).to(target_param.dtype))
        lesion_held = final_logits(model, held_batch)
        for candidate in banks:
            with torch.no_grad(): target_param.copy_(torch.from_numpy(candidate).to(target_param.dtype))
            cal_candidates.append(final_logits(model, cal_batch)); held_candidates.append(final_logits(model, held_batch))
    finally:
        with torch.no_grad(): target_param.copy_(original_t.to(target_param.dtype))
    if not torch.equal(target_param.detach().cpu().float(), original_t): raise ProbeError("weight rollback failed")
    lesion_metrics = per_sample_metrics(base_held, lesion_held)
    def stack(base, candidates):
        rows = [per_sample_metrics(base, x) for x in candidates]; return {k: np.stack([r[k] for r in rows]) for k in ("kl", "mse", "top10")}
    cal_mats = stack(base_cal, cal_candidates); held_mats = stack(base_held, held_candidates)
    oracle_cal = np.argmin(cal_mats["kl"], axis=0).astype(np.int64); oracle_held = np.argmin(held_mats["kl"], axis=0).astype(np.int64)
    direct_pred = classifier_predict(rep_cal, oracle_cal, rep_held, ROUTER_RANK); random_pred = np.random.default_rng(SEED + 50).integers(0, 4, size=len(heldout), dtype=np.int64)
    proj = projection_fit(rep_cal); all_rep = np.concatenate([rep_cal, rep_held]); split = len(rep_cal); _, community_meta = load_communities()
    real_X, real_rates, real_wall = reservoir_features(all_rep, proj, kind="real", rewire_seed=None, gain=3.0); real_rank = {}
    for rank in DIAGNOSTIC_RANKS: real_rank[rank] = route_metrics(classifier_predict(real_X[:split], oracle_cal, real_X[split:], rank), oracle_held, held_mats)
    degree = {r: [] for r in DIAGNOSTIC_RANKS}; community = {r: [] for r in DIAGNOSTIC_RANKS}; rate = {r: [] for r in DIAGNOSTIC_RANKS}; walls = {"real": real_wall, "degree": [], "community": [], "rate": []}; rate_match = []; target_rate = float(real_rates[:split].mean())
    for rep in range(REWIRE_REPLICATES):
        rs = SEED + 10000 * (rep + 1)
        Xd, _, w = reservoir_features(all_rep, proj, kind="degree", rewire_seed=rs, gain=3.0); walls["degree"].append(w)
        Xc, _, w = reservoir_features(all_rep, proj, kind="community", rewire_seed=rs, gain=3.0); walls["community"].append(w)
        trials = []; best_gain, best_delta = None, None
        for gain in RATE_GAIN_GRID:
            _, rr, _ = reservoir_features(rep_cal, proj, kind="rate", rewire_seed=rs, gain=gain); rv = float(rr.mean()); delta = abs(rv - target_rate); trials.append({"gain": gain, "calibration_rate": rv})
            if best_delta is None or delta < best_delta: best_gain, best_delta = gain, delta
        Xr, rr, w = reservoir_features(all_rep, proj, kind="rate", rewire_seed=rs, gain=float(best_gain)); walls["rate"].append(w); rate_match.append({"replicate": rep + 1, "rewire_seed": rs, "target_real_calibration_rate": target_rate, "gain_trials": trials, "selected_gain": float(best_gain), "heldout_rate": float(rr[split:].mean())})
        for rank in DIAGNOSTIC_RANKS:
            degree[rank].append(route_metrics(classifier_predict(Xd[:split], oracle_cal, Xd[split:], rank), oracle_held, held_mats)); community[rank].append(route_metrics(classifier_predict(Xc[:split], oracle_cal, Xc[split:], rank), oracle_held, held_mats)); rate[rank].append(route_metrics(classifier_predict(Xr[:split], oracle_cal, Xr[split:], rank), oracle_held, held_mats))
    direct_metrics = route_metrics(direct_pred, oracle_held, held_mats); random_metrics = route_metrics(random_pred, oracle_held, held_mats); oracle_metrics = route_metrics(oracle_held, oracle_held, held_mats); family_metrics = route_metrics(held_family, oracle_held, held_mats); real_metrics = real_rank[ROUTER_RANK]; degree_metrics = aggregate(degree[ROUTER_RANK]); community_metrics = aggregate(community[ROUTER_RANK]); rate_metrics = aggregate(rate[ROUTER_RANK])
    signatures, hashes = {}, {}
    for rank in DIAGNOSTIC_RANKS:
        r = real_rank[rank]; dmax = max(x["routing_accuracy"] for x in degree[rank]); cmax = max(x["routing_accuracy"] for x in community[rank]); mmax = max(x["routing_accuracy"] for x in rate[rank]); dr = route_metrics(classifier_predict(rep_cal, oracle_cal, rep_held, rank), oracle_held, held_mats)
        signatures[rank] = [r["routing_accuracy"] > dmax, r["routing_accuracy"] > cmax, r["routing_accuracy"] > mmax, r["routing_accuracy"] > dr["routing_accuracy"]]; hashes[rank] = r["prediction_sha256"]
    relation_floor = next((r for r in DIAGNOSTIC_RANKS if signatures[r] == signatures[ROUTER_RANK]), None); exact_floor = next((r for r in DIAGNOSTIC_RANKS if hashes[r] == hashes[ROUTER_RANK]), None)
    graph_best_acc = max(degree_metrics["routing_accuracy_max"], community_metrics["routing_accuracy_max"], rate_metrics["routing_accuracy_max"]); graph_best_kl = min(degree_metrics["next_token_kl_min"], community_metrics["next_token_kl_min"], rate_metrics["next_token_kl_min"])
    reject_direct = direct_metrics["routing_accuracy"] >= real_metrics["routing_accuracy"] and direct_metrics["next_token_kl"] <= real_metrics["next_token_kl"]; topology_specific = real_metrics["routing_accuracy"] > graph_best_acc and real_metrics["next_token_kl"] < graph_best_kl
    data_dir = Path(os.environ.get("FLY_DATA", Path.home() / "fly-data")); brain_sha = sha256(data_dir / "brain.npz"); weight_sha = sha256(data_dir / "weights.npz")
    if brain_sha != BRAIN_SHA256 or weight_sha != WEIGHTS_SHA256: raise ProbeError("MaleCNS digest mismatch")
    receipt = {
      "schema": SCHEMA, "status": "worker-observation-not-authority", "question": "Can real MaleCNS transient state route equal-budget reversible repair adapters inside frozen SmolLM2-360M better than direct and graph-null controls?", "pre_registered_owner": "helix-model/runs/016-connectome-open-weight-router/Run.json",
      "source": {"model": "HuggingFaceTB/SmolLM2-360M", "model_sha256": actual_sha, "model_bytes": model_file.stat().st_size, "target": TARGET_WEIGHT, "flyai_commit": DONOR_COMMIT, "brain_npz_sha256": brain_sha, "weights_npz_sha256": weight_sha, "malecns_commit": MALECNS_COMMIT, "community_null": community_meta},
      "prompt_packet": {"families": list(FAMILIES), "calibration_per_family": 6, "heldout_per_family": 6, "sha256": hashlib.sha256(json.dumps({"calibration": calibration, "heldout": heldout}, sort_keys=True).encode()).hexdigest()},
      "intervention": {"lesion_rank": LESION_RANK, "repair_fraction": REPAIR_FRACTION, "repair_banks": bank_meta, "base_weights_mutated_persistently": False, "rollback_exact": True},
      "oracle": {"calibration_distribution": np.bincount(oracle_cal, minlength=4).tolist(), "heldout_distribution": np.bincount(oracle_held, minlength=4).tolist(), "family_to_best_adapter_agreement_calibration": float(np.mean(cal_family == oracle_cal)), "family_to_best_adapter_agreement_heldout": float(np.mean(held_family == oracle_held))},
      "arms": {"low-rank-lesion-only": {"next_token_kl": float(np.mean(lesion_metrics["kl"])), "logit_mse": float(np.mean(lesion_metrics["mse"])), "top10_overlap": float(np.mean(lesion_metrics["top10"]))}, "oracle-repair-router": oracle_metrics, "family-oracle-diagnostic": family_metrics, "direct-linear-router-on-model-representation": direct_metrics, "random-router": random_metrics, "degree-preserving-rewired-malecns-router": degree_metrics, "community-preserving-malecns-router": community_metrics, "firing-rate-matched-rewired-malecns-router": rate_metrics, "real-malecns-router": real_metrics},
      "rate_match": rate_match,
      "secondary_distinction_diagnostic": {"ranks": list(DIAGNOSTIC_RANKS), "coarse_signature_order": ["real_beats_every_degree_rewire_accuracy", "real_beats_every_community_rewire_accuracy", "real_beats_every_rate_matched_rewire_accuracy", "real_beats_direct_linear_accuracy"], "signature_by_rank": {str(k): v for k, v in signatures.items()}, "real_prediction_sha256_by_rank": {str(k): v for k, v in hashes.items()}, "coarse_relation_floor_rank": relation_floor, "exact_behavior_floor_rank": exact_floor, "status": "secondary analysis added after run 017 and before run 016 execution; does not alter preregistered endpoint"},
      "decision": {"direct_linear_matches_or_beats_real_at_lower_cost": bool(reject_direct), "real_beats_all_tested_graph_nulls_on_accuracy_and_kl": bool(topology_specific), "graph_null_best_routing_accuracy": float(graph_best_acc), "graph_null_best_next_token_kl": float(graph_best_kl), "promotion_authorized": False},
      "cost": {"connectome_state_bytes_per_sample": SKETCH_WIDTH * 4, "router_readout_rank": ROUTER_RANK, "direct_model_representation_bytes_per_sample": int(rep_cal.shape[1] * 4), "reservoir_wall_seconds": walls, "wall_seconds_total": time.time() - started},
      "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
      "claim_boundary": "bounded held-out restoration/routing experiment on one frozen model layer and four fixed repair adapters; next-token restoration is not downstream task capability, and this run does not establish a general connectome advantage or authorize promotion"
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RUN016_ROUTER " + json.dumps({"arms": receipt["arms"], "decision": receipt["decision"], "secondary_distinction_diagnostic": receipt["secondary_distinction_diagnostic"], "wall_seconds_total": receipt["cost"]["wall_seconds_total"]}, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
