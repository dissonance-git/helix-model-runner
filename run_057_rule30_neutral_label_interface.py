from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

CANONICAL_HELIX_COMMIT = "4e6270e7e4ca198bd9adea2d8261e855b98b6283"
HELIX_INPUT = Path(__file__).resolve().parent / "verification" / "helix-4e6270e7"
sys.path.insert(0, str(HELIX_INPUT))

from engine.runtime.experiments.cellular_automata.rule30_transform_program import (  # noqa: E402
    compile_rule30_branch_mechanism_task,
)

RUN_VERSION = "helix-rule30-neutral-label-interface-001.0"
LABELS = ("A","B","C","D","E","F","G","H")
FORMULAS = (
    "OR(E,S)",
    "AND(E,S)",
    "E",
    "S",
    "NOT(E)",
    "NOT(S)",
    "FALSE",
    "TRUE",
)
EXPECTED = "OR(E,S)"
ROTATIONS = tuple(range(len(LABELS)))


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda i: values[i])


def _rank(values: list[float], index: int) -> int:
    target = values[index]
    return 1 + sum(value > target for value in values)


def _mapping(rotation: int) -> dict[str, str]:
    return {
        LABELS[(formula_index + rotation) % len(LABELS)]: formula
        for formula_index, formula in enumerate(FORMULAS)
    }


def _training_patterns() -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in compile_rule30_branch_mechanism_task()["task"]["training_patterns"]
    ]


def _table_text(patterns: list[dict[str, Any]], surface: str) -> str:
    if surface == "plain":
        return "; ".join(
            f"E={int(row['E'])}, S={int(row['S'])} -> branch={int(row['branch'])} ({row['count']} states)"
            for row in patterns
        )
    if surface == "compact":
        return " ".join(
            f"{int(row['E'])}{int(row['S'])}->{int(row['branch'])}x{row['count']}"
            for row in patterns
        )
    raise KeyError(surface)


def _prompt(patterns: list[dict[str, Any]], surface: str, mapping: dict[str, str]) -> str:
    mapping_text = "; ".join(f"{label}={mapping[label]}" for label in LABELS)
    return (
        f"Exact branch(E,S) training table: {_table_text(patterns, surface)}. "
        f"Candidate labels: {mapping_text}. "
        "Choose the label for the shortest exact Boolean formula matching every training row. "
        "Answer label:"
    )


def _score_labels(model_dir: str, prompt: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    torch.set_num_threads(4)

    def render(user_text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def label_ids() -> dict[str, list[int]]:
        return {
            label: tokenizer(" " + label, add_special_tokens=False).input_ids
            for label in LABELS
        }

    token_ids = label_ids()
    lengths = {label: len(ids) for label, ids in token_ids.items()}
    if set(lengths.values()) != {1}:
        raise RuntimeError(f"neutral-label tokenization guard failed: {lengths}")

    @torch.inference_mode()
    def score(user_text: str) -> tuple[list[float], int]:
        rendered = render(user_text)
        prompt_ids = tokenizer(rendered, add_special_tokens=False).input_ids
        sequences = [prompt_ids + token_ids[label] for label in LABELS]
        max_len = max(map(len, sequences))
        input_ids = torch.full(
            (len(sequences), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[i, :len(seq)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        pos = len(prompt_ids)
        return [
            float(log_probs[i, pos - 1, token_ids[label][0]])
            for i, label in enumerate(LABELS)
        ], len(prompt_ids)

    started = time.time()
    raw, prompt_tokens = score(prompt)
    prior, prior_tokens = score("Choose one answer label from A, B, C, D, E, F, G, H. Answer label:")
    pmi = [raw[i] - prior[i] for i in range(len(LABELS))]
    return {
        "raw": dict(zip(LABELS, raw)),
        "prior": dict(zip(LABELS, prior)),
        "pmi": dict(zip(LABELS, pmi)),
        "raw_prediction": LABELS[_argmax(raw)],
        "pmi_prediction": LABELS[_argmax(pmi)],
        "label_token_lengths": lengths,
        "resources": {
            "forward_batches": 2,
            "prompt_tokens": prompt_tokens + prior_tokens,
            "choice_tokens": len(LABELS) * 2,
            "generate_calls": 0,
            "wall_seconds": time.time() - started,
        },
    }


def self_test() -> dict[str, Any]:
    patterns = _training_patterns()
    mappings = [_mapping(rotation) for rotation in ROTATIONS]
    for formula in FORMULAS:
        observed = sorted(
            label
            for mapping in mappings
            for label, mapped_formula in mapping.items()
            if mapped_formula == formula
        )
        if observed != sorted(LABELS):
            raise AssertionError((formula, observed))
    prompts = {
        f"{surface}:{rotation}": _prompt(patterns, surface, mappings[rotation])
        for surface in ("plain","compact")
        for rotation in ROTATIONS
    }
    return {
        "schema_version": RUN_VERSION,
        "status": "self-test-pass",
        "canonical_helix_commit": CANONICAL_HELIX_COMMIT,
        "rotations": len(ROTATIONS),
        "surfaces": 2,
        "formula_count": len(FORMULAS),
        "each_formula_occupies_every_label_once": True,
        "prompt_digests": {
            key: hashlib.sha256(value.encode()).hexdigest()
            for key, value in sorted(prompts.items())
        },
    }


def evaluate(model_dir: str, model_id: str, revision: str, model_file: str, expected_sha256: str) -> dict[str, Any]:
    path = Path(model_file)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise SystemExit(f"model digest mismatch: {observed}")

    patterns = _training_patterns()
    rows = []
    correct_counts = {"plain": 0, "compact": 0}
    formula_pmi = {
        surface: {formula: [] for formula in FORMULAS}
        for surface in ("plain","compact")
    }
    total_resources = {
        "forward_batches": 0,
        "prompt_tokens": 0,
        "choice_tokens": 0,
        "generate_calls": 0,
        "wall_seconds": 0.0,
    }

    # Load once per surface/rotation call is intentionally avoided by keeping
    # this bounded runner simple at first; score function loads model each call
    # would be wasteful, so instead below we cache results through one helper
    # process boundary? No: instantiate scorer inline once by monkey-free local
    # class is clearer. Reuse model by delegating all prompts in one function.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    torch.set_num_threads(4)
    token_ids = {
        label: tokenizer(" " + label, add_special_tokens=False).input_ids
        for label in LABELS
    }
    lengths = {label: len(ids) for label, ids in token_ids.items()}
    if set(lengths.values()) != {1}:
        raise RuntimeError(f"neutral-label tokenization guard failed: {lengths}")

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role":"user","content":text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def score(text: str) -> tuple[list[float], int]:
        prompt_ids = tokenizer(render(text), add_special_tokens=False).input_ids
        sequences = [prompt_ids + token_ids[label] for label in LABELS]
        max_len = max(map(len,sequences))
        input_ids = torch.full((8,max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            input_ids[i,:len(seq)] = torch.tensor(seq,dtype=torch.long)
            attention[i,:len(seq)] = 1
        log_probs = torch.log_softmax(model(input_ids=input_ids,attention_mask=attention).logits,dim=-1)
        pos=len(prompt_ids)
        return [float(log_probs[i,pos-1,token_ids[label][0]]) for i,label in enumerate(LABELS)], len(prompt_ids)

    prior, prior_tokens = score("Choose one answer label from A, B, C, D, E, F, G, H. Answer label:")
    started=time.time()
    for surface in ("plain","compact"):
        for rotation in ROTATIONS:
            mapping=_mapping(rotation)
            prompt=_prompt(patterns,surface,mapping)
            raw,prompt_tokens=score(prompt)
            pmi=[raw[i]-prior[i] for i in range(8)]
            raw_label=LABELS[_argmax(raw)]
            pmi_label=LABELS[_argmax(pmi)]
            expected_label=next(label for label,formula in mapping.items() if formula==EXPECTED)
            correct_counts[surface]+=int(pmi_label==expected_label)
            correct_index=LABELS.index(expected_label)
            for label,formula in mapping.items():
                formula_pmi[surface][formula].append(pmi[LABELS.index(label)])
            rows.append({
                "surface":surface,
                "rotation":rotation,
                "mapping":mapping,
                "expected_label":expected_label,
                "raw_prediction_label":raw_label,
                "raw_prediction_formula":mapping[raw_label],
                "pmi_prediction_label":pmi_label,
                "pmi_prediction_formula":mapping[pmi_label],
                "correct_formula_rank":_rank(pmi,correct_index),
                "correct_formula_pmi":pmi[correct_index],
                "pmi_margin":sorted(pmi,reverse=True)[0]-sorted(pmi,reverse=True)[1],
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            })
            total_resources["forward_batches"]+=1
            total_resources["prompt_tokens"]+=prompt_tokens
            total_resources["choice_tokens"]+=8

    total_resources["forward_batches"]+=1
    total_resources["prompt_tokens"]+=prior_tokens
    total_resources["choice_tokens"]+=8
    total_resources["wall_seconds"]=time.time()-started

    overall=sum(correct_counts.values())
    aggregate_formula_pmi={
        surface:{
            formula:sum(values)/len(values)
            for formula,values in formula_pmi[surface].items()
        }
        for surface in formula_pmi
    }
    decisions={
        "robust_neutral_label_recovery":(
            correct_counts["plain"]>=6
            and correct_counts["compact"]>=6
            and overall>=12
        ),
        "strong_recovery_16_of_16":overall==16,
    }
    return {
        "schema_version":RUN_VERSION,
        "status":"completed-public-neutral-label-interface",
        "actor":{
            "model":model_id,
            "revision":revision,
            "model_file_sha256":observed,
            "model_bytes":path.stat().st_size,
            "dtype":"float32",
            "device":"cpu",
            "weights_frozen":True,
        },
        "compiler":{
            "canonical_helix_commit":CANONICAL_HELIX_COMMIT,
            "training_patterns":patterns,
            "basis_operators":compile_rule30_branch_mechanism_task()["program"]["basis_operators"],
        },
        "interface":{
            "labels":list(LABELS),
            "formulas":list(FORMULAS),
            "rotations":len(ROTATIONS),
            "label_token_lengths":lengths,
            "primary_scoring":"single-label log-likelihood minus neutral label prior",
        },
        "correct_rotation_counts":{**correct_counts,"overall":overall},
        "aggregate_formula_pmi":aggregate_formula_pmi,
        "rows":rows,
        "resources":total_resources,
        "decisions":decisions,
        "claim_boundary":[
            "Public diagnostic of bounded formula recognition under label-rotation controls.",
            "The formula set and rotations are frozen and the task does not test independent formula discovery.",
            "A model-side success does not establish the all-dyadic Rule 30 theorem or authorize model promotion.",
        ],
    }


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True))
        return
    if not all((args.model_dir,args.model_id,args.revision,args.model_file,args.expected_sha256,args.output)):
        raise SystemExit("all actor and output arguments are required")
    result=evaluate(args.model_dir,args.model_id,args.revision,args.model_file,args.expected_sha256)
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({
        "status":result["status"],
        "actor":result["actor"],
        "correct_rotation_counts":result["correct_rotation_counts"],
        "aggregate_formula_pmi":result["aggregate_formula_pmi"],
        "decisions":result["decisions"],
        "resources":result["resources"],
    },indent=2,sort_keys=True))


if __name__=="__main__":
    main()
