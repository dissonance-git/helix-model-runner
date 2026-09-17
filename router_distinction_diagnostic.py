#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import connectome_open_weight_router as r
from probe import randomized_low_rank

SCHEMA = "helix-model-router-distinction-diagnostic-v1"
RANKS = (1, 2, 4, 8, 16)


def relation_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    truth = np.asarray(truth, np.int64)
    pred = np.asarray(pred, np.int64)
    n_classes = 4
    confusion = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(truth, pred):
        confusion[t, p] += 1
    recalls = []
    for c in range(n_classes):
        total = int(np.sum(truth == c))
        recalls.append(float(np.mean(pred[truth == c] == c)) if total else None)

    required_pairs = kept_required = same_pairs = false_splits = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if truth[i] != truth[j]:
                required_pairs += 1
                kept_required += int(pred[i] != pred[j])
            else:
                same_pairs += 1
                false_splits += int(pred[i] != pred[j])

    return {
        "accuracy": float(np.mean(pred == truth)),
        "per_class_recall": recalls,
        "balanced_accuracy": float(np.mean([x for x in recalls if x is not None])),
        "predicted_class_counts": np.bincount(pred, minlength=n_classes).tolist(),
        "predicted_classes_used": int(len(np.unique(pred))),
        "confusion_matrix_rows_truth_cols_prediction": confusion.tolist(),
        "required_different_pairs": required_pairs,
        "required_distinctions_preserved": kept_required,
        "required_distinction_recall": float(kept_required / required_pairs) if required_pairs else 1.0,
        "same_target_pairs": same_pairs,
        "false_splits": false_splits,
        "false_split_rate": float(false_splits / same_pairs) if same_pairs else 0.0,
        "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--output", default="router-distinction.json")
    args = ap.parse_args()

    started = time.time()
    model_file = Path(args.model_file)
    actual_sha = r.file_sha256(model_file)
    if actual_sha != args.expected_sha256.lower():
        raise r.ProbeError(f"model SHA mismatch {actual_sha}")

    torch.manual_seed(r.SEED)
    np.random.seed(r.SEED)
    calibration, heldout = r.prompt_packet()
    cal_text = [x[0] for x in calibration]
    held_text = [x[0] for x in heldout]
    cal_family = np.asarray([x[1] for x in calibration], np.int64)

    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    cal_batch = r.tokenize(tok, cal_text)
    held_batch = r.tokenize(tok, held_text)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target_param = r.get_weight(model)
    target_module = r.get_module(model)
    original_t = target_param.detach().cpu().float().clone()
    original = original_t.numpy().copy()

    base_cal, rep_cal, cap_cal = r.final_logits_and_rep(model, cal_batch, target_module)
    base_held, rep_held, _ = r.final_logits_and_rep(model, held_batch)
    if cap_cal is None:
        raise r.ProbeError("activation capture failed")

    mask = cal_batch["attention_mask"].bool().cpu().numpy()
    token_family = np.broadcast_to(cal_family[:, None], mask.shape)[mask]
    token_inputs = cap_cal[mask]
    low_rank = randomized_low_rank(original, r.LESION_RANK, seed=r.SEED + r.LESION_RANK)
    banks, _ = r.repair_bank(original, low_rank, token_inputs, token_family)

    cal_candidates, held_candidates = [], []
    try:
        for candidate in banks:
            with torch.no_grad():
                target_param.copy_(torch.from_numpy(candidate).to(target_param.dtype))
            cal_candidates.append(r.final_logits(model, cal_batch))
            held_candidates.append(r.final_logits(model, held_batch))
    finally:
        with torch.no_grad():
            target_param.copy_(original_t.to(target_param.dtype))
    if not torch.equal(target_param.detach().cpu().float(), original_t):
        raise r.ProbeError("weight rollback failed")

    def stack(base, candidates):
        rows = [r.per_sample_metrics(base, x) for x in candidates]
        return {k: np.stack([row[k] for row in rows]) for k in ("kl", "mse", "top10")}

    cal_mats = stack(base_cal, cal_candidates)
    held_mats = stack(base_held, held_candidates)
    oracle_cal = np.argmin(cal_mats["kl"], axis=0).astype(np.int64)
    oracle_held = np.argmin(held_mats["kl"], axis=0).astype(np.int64)
    majority_adapter = int(np.argmax(np.bincount(oracle_cal, minlength=4)))
    heldout_majority_adapter = int(np.argmax(np.bincount(oracle_held, minlength=4)))
    majority_pred = np.full(len(oracle_held), majority_adapter, np.int64)
    heldout_oracle_majority_pred = np.full(len(oracle_held), heldout_majority_adapter, np.int64)

    direct = {}
    for rank in RANKS:
        pred = r.classifier_predict(rep_cal, oracle_cal, rep_held, rank)
        direct[str(rank)] = {
            **relation_metrics(oracle_held, pred),
            "restoration": r.route_metrics(pred, oracle_held, held_mats),
        }

    receipt = {
        "schema": SCHEMA,
        "status": "post-run-016-diagnostic",
        "parent_run": "016-connectome-open-weight-router",
        "source": {
            "model": "HuggingFaceTB/SmolLM2-360M",
            "model_sha256": actual_sha,
            "model_bytes": model_file.stat().st_size,
            "worker_dependency": "connectome_open_weight_router.py",
        },
        "oracle": {
            "calibration_distribution": np.bincount(oracle_cal, minlength=4).tolist(),
            "heldout_distribution": np.bincount(oracle_held, minlength=4).tolist(),
        },
        "baselines": {
            "calibration_majority_router": {
                "adapter": majority_adapter,
                **relation_metrics(oracle_held, majority_pred),
                "restoration": r.route_metrics(majority_pred, oracle_held, held_mats),
            },
            "heldout_oracle_majority_ceiling_diagnostic": {
                "adapter": heldout_majority_adapter,
                **relation_metrics(oracle_held, heldout_oracle_majority_pred),
                "restoration": r.route_metrics(heldout_oracle_majority_pred, oracle_held, held_mats),
                "not_a_deployable_baseline": True,
            },
        },
        "direct_linear_by_rank": direct,
        "run_016_graph_prediction": {
            "prediction_sha256": "d335b3bc1461ed30f75513e7c17ed90d15c756ef02e8911aee0c66b1bddaf425",
            "identified_as_constant_adapter": 1,
            "required_distinction_recall": 0.0,
            "predicted_classes_used": 1,
        },
        "interpretation_rule": "report target-distinction recall and false-split rate separately from scalar accuracy; a constant majority router cannot count as preserving task distinctions merely because the target distribution is imbalanced",
        "claim_boundary": "post-execution diagnostic selected after run 016 exposed prediction collapse; does not alter the preregistered run-016 result",
        "wall_seconds": time.time() - started,
    }
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("ROUTER_DISTINCTION " + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
