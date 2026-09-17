#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from probe import HELIX_SCALE_WEIGHTS, UNIFORM_SCALE_WEIGHTS, magnitude_indices, multiscale_indices, randomized_low_rank

SCHEMA_VERSION = "helix-model-functional-residual-grid-001.0"
SEED = 13013
RANK = 32
LAYERS = (0, 4, 8, 12, 15, 19, 23, 27, 31)
BUDGETS = (0.005, 0.01, 0.02)
ARMS = ("magnitude-residual", "uniform-block-coverage-residual", "multiscale-coverage-residual")

class ProbeError(ValueError):
    pass

def sha256_file(path, chunk_bytes=8*1024*1024):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()

def load_prompts(path):
    raw=Path(path).read_bytes(); data=json.loads(raw.decode("utf-8")); prompts=data.get("prompts")
    if not isinstance(prompts,list) or not prompts or not all(isinstance(x,str) and x.strip() for x in prompts):
        raise ProbeError("invalid prompt packet")
    return prompts, hashlib.sha256(raw).hexdigest()

def forward_logits(model,batch):
    with torch.inference_mode():
        return model(**batch).logits.detach().cpu().float()

def metrics(base,cand,input_ids,attention_mask):
    mask=attention_mask.bool().cpu(); valid=mask.unsqueeze(-1).expand_as(base)
    b=base[valid].view(-1,base.shape[-1]); c=cand[valid].view(-1,cand.shape[-1])
    bl=F.log_softmax(b,dim=-1); cl=F.log_softmax(c,dim=-1); bp=bl.exp()
    topb=torch.topk(b,k=10,dim=-1).indices; topc=torch.topk(c,k=10,dim=-1).indices
    overlap=(topb.unsqueeze(-1)==topc.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)/10.0
    labels=input_ids[:,1:].cpu(); tm=attention_mask[:,1:].bool().cpu(); bs,cs=base[:,:-1,:],cand[:,:-1,:]
    bnll=F.cross_entropy(bs[tm],labels[tm],reduction="mean").item(); cnll=F.cross_entropy(cs[tm],labels[tm],reduction="mean").item()
    return {
        "logit_mse":float(torch.mean((b-c)**2).item()),
        "kl_baseline_to_candidate":float(torch.mean(torch.sum(bp*(bl-cl),dim=-1)).item()),
        "top10_overlap":float(overlap.mean().item()),
        "argmax_agreement":float((b.argmax(dim=-1)==c.argmax(dim=-1)).float().mean().item()),
        "baseline_nll":float(bnll),"candidate_nll":float(cnll),"nll_delta":float(cnll-bnll),"absolute_nll_delta":float(abs(cnll-bnll))
    }

def candidate(low,residual,arm,budget):
    count=max(1,int(round(residual.size*budget)))
    if arm=="magnitude-residual": idx=magnitude_indices(residual,count)
    elif arm=="uniform-block-coverage-residual": idx=multiscale_indices(residual,count,scale_weights=UNIFORM_SCALE_WEIGHTS)
    elif arm=="multiscale-coverage-residual": idx=multiscale_indices(residual,count,scale_weights=HELIX_SCALE_WEIGHTS)
    else: raise ProbeError(arm)
    out=low.copy(); out.ravel()[idx]+=residual.ravel()[idx]
    return out.astype(np.float32,copy=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",required=True); ap.add_argument("--model-file",required=True); ap.add_argument("--expected-sha256",required=True); ap.add_argument("--prompts",required=True); ap.add_argument("--output",required=True); args=ap.parse_args()
    started=time.time(); actual=sha256_file(args.model_file)
    if actual != args.expected_sha256.lower(): raise ProbeError(f"source SHA mismatch: {actual}")
    prompts,prompt_sha=load_prompts(args.prompts); torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(max(1,min(4,os.cpu_count() or 1)))
    tok=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    batch=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=128,add_special_tokens=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,dtype=torch.float32,low_cpu_mem_usage=True); model.eval(); baseline=forward_logits(model,batch); params=dict(model.named_parameters()); results=[]
    for layer in LAYERS:
        name=f"model.layers.{layer}.self_attn.o_proj.weight"; p=params[name]; original_t=p.detach().cpu().float().clone(); original=original_t.numpy(); low=randomized_low_rank(original,RANK,seed=SEED+RANK); residual=(original-low).astype(np.float32,copy=False)
        for budget in BUDGETS:
            for arm in ARMS:
                arr=candidate(low,residual,arm,budget)
                with torch.no_grad(): p.copy_(torch.from_numpy(arr).to(p.dtype))
                logits=forward_logits(model,batch); results.append({"layer":layer,"tensor":name,"rank":RANK,"residual_fraction":budget,"arm":arm,"metrics":metrics(baseline,logits,batch["input_ids"],batch["attention_mask"])})
                with torch.no_grad(): p.copy_(original_t.to(p.dtype))
    receipt={"schema_version":SCHEMA_VERSION,"source":{"model_file":Path(args.model_file).name,"sha256":actual,"bytes":Path(args.model_file).stat().st_size},"prompt_packet":{"file":Path(args.prompts).name,"sha256":prompt_sha,"count":len(prompts)},"environment":{"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"platform":platform.platform(),"cpu_count":os.cpu_count()},"settings":{"primary_metric":"kl_baseline_to_candidate","rank":RANK,"layers":list(LAYERS),"residual_budgets":list(BUDGETS),"arms":list(ARMS),"seed":SEED},"results":results,"wall_seconds":time.time()-started,"claim_boundary":"fresh-prompt depth-and-budget robustness test on one frozen base model; no capability, mathematical-transfer, or promotion claim"}
    Path(args.output).write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return 0

if __name__=="__main__": raise SystemExit(main())
