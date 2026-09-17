#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe import (
    HELIX_SCALE_WEIGHTS,
    UNIFORM_SCALE_WEIGHTS,
    magnitude_indices,
    multiscale_indices,
    random_indices,
    randomized_low_rank,
)

SCHEMA_VERSION = "helix-model-functional-residual-probe-001.0"
DEFAULT_SEED = 13013
RANK = 32
BUDGET = 0.01
TENSORS = (
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.15.self_attn.o_proj.weight",
    "model.layers.31.self_attn.o_proj.weight",
)
ARMS = (
    "low-rank-only",
    "magnitude-residual",
    "uniform-block-coverage-residual",
    "multiscale-coverage-residual",
    "random-residual",
)

class ProbeError(ValueError):
    pass

def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)

def load_prompts(path: str | Path) -> tuple[list[str], str]:
    raw = Path(path).read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    prompts = parsed.get("prompts")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(x, str) and x.strip() for x in prompts):
        raise ProbeError("prompt packet must contain a non-empty string list named prompts")
    return prompts, hashlib.sha256(raw).hexdigest()

def get_parameter(model: torch.nn.Module, name: str) -> torch.nn.Parameter:
    table = dict(model.named_parameters())
    if name not in table:
        raise ProbeError(f"parameter not found: {name}")
    p = table[name]
    if p.ndim != 2:
        raise ProbeError(f"{name} must be 2-D, got {tuple(p.shape)}")
    return p

def build_candidate(original: np.ndarray, arm: str, *, seed: int) -> np.ndarray:
    low_rank = randomized_low_rank(original, RANK, seed=seed + RANK)
    residual = (original - low_rank).astype(np.float32, copy=False)
    count = max(1, int(round(original.size * BUDGET)))
    if arm == "low-rank-only":
        selected = np.empty(0, dtype=np.int64)
    elif arm == "magnitude-residual":
        selected = magnitude_indices(residual, count)
    elif arm == "uniform-block-coverage-residual":
        selected = multiscale_indices(residual, count, scale_weights=UNIFORM_SCALE_WEIGHTS)
    elif arm == "multiscale-coverage-residual":
        selected = multiscale_indices(residual, count, scale_weights=HELIX_SCALE_WEIGHTS)
    elif arm == "random-residual":
        selected = random_indices(residual.size, count, seed=seed + RANK * 100000 + int(BUDGET * 1_000_000))
    else:
        raise ProbeError(f"unknown arm: {arm}")
    candidate = low_rank.copy()
    if selected.size:
        candidate.ravel()[selected] += residual.ravel()[selected]
    return candidate.astype(np.float32, copy=False)

def forward_logits(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        return model(**batch).logits.detach().cpu().float()

def functional_metrics(base: torch.Tensor, cand: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, float]:
    mask = attention_mask.bool().cpu()
    valid = mask.unsqueeze(-1).expand_as(base)
    b = base[valid].view(-1, base.shape[-1])
    c = cand[valid].view(-1, cand.shape[-1])
    b_logp = F.log_softmax(b, dim=-1)
    c_logp = F.log_softmax(c, dim=-1)
    b_p = b_logp.exp()
    top_b = torch.topk(b, k=10, dim=-1).indices
    top_c = torch.topk(c, k=10, dim=-1).indices
    overlap = (top_b.unsqueeze(-1) == top_c.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1) / 10.0
    labels = input_ids[:, 1:].cpu()
    target_mask = attention_mask[:, 1:].bool().cpu()
    b_shift, c_shift = base[:, :-1, :], cand[:, :-1, :]
    base_nll = F.cross_entropy(b_shift[target_mask], labels[target_mask], reduction="mean").item()
    cand_nll = F.cross_entropy(c_shift[target_mask], labels[target_mask], reduction="mean").item()
    final_indices = attention_mask.sum(dim=1).cpu() - 1
    batch_indices = torch.arange(attention_mask.shape[0])
    bf, cf = base[batch_indices, final_indices, :], cand[batch_indices, final_indices, :]
    bf_logp, cf_logp = F.log_softmax(bf, dim=-1), F.log_softmax(cf, dim=-1)
    bf_p = bf_logp.exp()
    return {
        "logit_mse": float(torch.mean((b-c)**2).item()),
        "kl_baseline_to_candidate": float(torch.mean(torch.sum(b_p*(b_logp-c_logp), dim=-1)).item()),
        "top10_overlap": float(overlap.mean().item()),
        "argmax_agreement": float((b.argmax(dim=-1)==c.argmax(dim=-1)).float().mean().item()),
        "baseline_nll": float(base_nll),
        "candidate_nll": float(cand_nll),
        "nll_delta": float(cand_nll-base_nll),
        "final_position_kl": float(torch.mean(torch.sum(bf_p*(bf_logp-cf_logp), dim=-1)).item()),
        "final_top1_agreement": float((bf.argmax(dim=-1)==cf.argmax(dim=-1)).float().mean().item()),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    started=time.time()
    actual=sha256_file(args.model_file)
    if actual != args.expected_sha256.lower():
        raise ProbeError(f"source SHA mismatch: {actual} != {args.expected_sha256}")
    prompts,prompt_sha=load_prompts(args.prompts)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    batch=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=128,add_special_tokens=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,low_cpu_mem_usage=True)
    model.eval()
    baseline=forward_logits(model,batch)
    results=[]
    for tensor_name in TENSORS:
        p=get_parameter(model,tensor_name)
        original_t=p.detach().cpu().float().clone()
        original=original_t.numpy()
        for arm in ARMS:
            candidate=build_candidate(original,arm,seed=args.seed)
            with torch.no_grad(): p.copy_(torch.from_numpy(candidate).to(p.dtype))
            logits=forward_logits(model,batch)
            results.append({"tensor":tensor_name,"rank":RANK,"residual_fraction":BUDGET,"arm":arm,"metrics":functional_metrics(baseline,logits,batch["input_ids"],batch["attention_mask"])})
            with torch.no_grad(): p.copy_(original_t.to(p.dtype))
    receipt={
        "schema_version":SCHEMA_VERSION,
        "source":{"model_file":Path(args.model_file).name,"sha256":actual,"bytes":Path(args.model_file).stat().st_size},
        "prompt_packet":{"file":Path(args.prompts).name,"sha256":prompt_sha,"count":len(prompts)},
        "environment":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"platform":platform.platform(),"cpu_count":os.cpu_count()},
        "settings":{"rank":RANK,"residual_fraction":BUDGET,"seed":args.seed,"tensors":list(TENSORS),"arms":list(ARMS),"scope":"stage-1-generated attention-output-channel hypothesis"},
        "results":results,
        "wall_seconds":time.time()-started,
        "claim_boundary":"bounded CPU forward-pass perturbation evidence on one frozen base model and one fresh prompt packet; not a capability improvement, mathematical transfer, or model promotion"
    }
    Path(args.output).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
