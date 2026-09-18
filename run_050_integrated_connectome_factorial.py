"""Run 050 stage 0: reversible integrated-connectome actor smoke.

This is an architecture/reproducibility test only. It does not train the organ
and cannot establish a capability gain.

The recurrent component is inside the transformer forward path. It is not a
sidecar/router. The base model and connectome stay frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


RUN_VERSION = "helix-integrated-connectome-factorial-stage0-001.0"
GRAPH_SHA256 = "738d23289b7345dcf49a25e9305d106e7349fca87b93f97df8698db2753ce935"
GRAPH_NODES = 5600
GRAPH_EDGES = 1187928
INTERFACE_WIDTH = 32
ALPHA = 0.15
BETA = 0.10
PROBE_GATE = 0.01
SEED = 50050


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_graph(path: Path) -> tuple[torch.Tensor, int, int]:
    observed = sha256_file(path)
    if observed != GRAPH_SHA256:
        raise RuntimeError(f"graph digest mismatch: expected {GRAPH_SHA256}, observed {observed}")
    with np.load(path, allow_pickle=False) as data:
        source = np.asarray(data["source"], dtype=np.int64)
        target = np.asarray(data["target"], dtype=np.int64)
        weight = np.asarray(data["weight"], dtype=np.float32)
        node_ids = data["node_ids"]
    n = int(len(node_ids))
    if n != GRAPH_NODES:
        raise RuntimeError(f"graph node mismatch: {n} != {GRAPH_NODES}")
    if len(source) != GRAPH_EDGES or len(target) != GRAPH_EDGES:
        raise RuntimeError(f"graph edge mismatch: {len(source)} != {GRAPH_EDGES}")
    if np.any(source < 0) or np.any(target < 0) or np.any(source >= n) or np.any(target >= n):
        raise RuntimeError("graph contains out-of-range indices")
    if not np.isfinite(weight).all():
        raise RuntimeError("graph contains non-finite weights")

    # Recurrent influx at target i from source j: W[i,j] * h[j].
    indegree = np.bincount(target, minlength=n).astype(np.float32)
    values = (weight / np.sqrt(indegree[target] + 1.0)) * np.float32(BETA)
    indices = torch.from_numpy(np.stack([target, source], axis=0)).long()
    values_t = torch.from_numpy(values).float()
    matrix = torch.sparse_coo_tensor(indices, values_t, (n, n), dtype=torch.float32).coalesce()
    return matrix, n, len(source)


def fixed_bridge(hidden_size: int, width: int, *, seed: int) -> torch.Tensor:
    """Return D x K deterministic orthonormal columns."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + hidden_size)
    raw = torch.randn(hidden_size, width, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q.contiguous()


class IntegratedConnectome(nn.Module):
    """SmolFly-style persistent leaky reservoir with matched trainable capacity."""

    def __init__(self, graph: torch.Tensor, hidden_size: int, width: int = INTERFACE_WIDTH):
        super().__init__()
        if width >= hidden_size:
            raise ValueError("interface width must be smaller than model hidden size")
        self.hidden_size = int(hidden_size)
        self.width = int(width)
        self.neurons = int(graph.shape[0])
        self.register_buffer("graph", graph)
        bridge = fixed_bridge(hidden_size, width, seed=SEED)
        self.register_buffer("bridge", bridge)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(SEED + 1)
        scale_in = 1.0 / math.sqrt(width)
        scale_out = 1.0 / math.sqrt(self.neurons)
        self.connectome_in = nn.Parameter(
            torch.randn(self.neurons, width, generator=gen, dtype=torch.float32) * scale_in
        )
        self.connectome_out = nn.Parameter(
            torch.randn(width, self.neurons, generator=gen, dtype=torch.float32) * scale_out
        )
        self.gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.state: torch.Tensor | None = None
        self.last_state_rms = 0.0
        self.steps = 0

    @property
    def declared_trainable_parameters(self) -> int:
        return int(self.connectome_in.numel() + self.connectome_out.numel() + self.gate.numel())

    def reset_state(self) -> None:
        self.state = None
        self.last_state_rms = 0.0
        self.steps = 0

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        # Reservoir math remains float32 even when the actor uses a smaller dtype.
        xf = x.float()
        batch = xf.shape[0]
        if self.state is None or self.state.shape[0] != batch:
            self.state = torch.zeros(batch, self.neurons, dtype=torch.float32, device=xf.device)

        z = xf @ self.bridge
        drive = z @ self.connectome_in.T
        recurrent = torch.sparse.mm(self.graph, self.state.T).T
        candidate = torch.tanh(drive + recurrent)
        self.state = (1.0 - ALPHA) * self.state + ALPHA * candidate
        rms = torch.sqrt(torch.mean(self.state * self.state, dim=-1, keepdim=True) + 1e-8)
        normalized = self.state / rms
        compact = normalized @ self.connectome_out.T
        delta = compact @ self.bridge.T
        self.last_state_rms = float(torch.sqrt(torch.mean(self.state * self.state)).item())
        self.steps += 1
        return delta.to(dtype=x.dtype)

    def forward_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise RuntimeError(f"expected [batch,seq,hidden], got {tuple(hidden.shape)}")
        # Zero gate is a strict identity boundary while the recurrent state still runs.
        out = []
        gate = float(self.gate.detach().cpu())
        for index in range(hidden.shape[1]):
            x = hidden[:, index, :]
            delta = self._step(x)
            out.append((x if gate == 0.0 else x + self.gate.to(x.dtype) * delta).unsqueeze(1))
        return torch.cat(out, dim=1)

    def hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            return (self.forward_hidden(output[0]),) + output[1:]
        return self.forward_hidden(output)


def final_logits(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        out = model(**batch, use_cache=False, return_dict=True)
    mask = batch["attention_mask"].bool()
    final = mask.sum(dim=1) - 1
    rows = torch.arange(mask.shape[0], device=final.device)
    return out.logits[rows, final].detach().cpu().float()


def run_smoke(
    *,
    model_dir: str,
    model_file: Path,
    expected_model_sha256: str,
    graph_path: Path,
    actor_model: str,
    actor_revision: str,
    output: Path,
) -> dict[str, Any]:
    observed_model = sha256_file(model_file)
    if observed_model != expected_model_sha256:
        raise RuntimeError(
            f"model digest mismatch: expected {expected_model_sha256}, observed {observed_model}"
        )
    graph, graph_nodes, graph_edges = load_graph(graph_path)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    layers = model.model.layers
    layer_count = len(layers)
    midpoint_index = layer_count // 2 - 1
    hidden_size = int(model.config.hidden_size)

    organ = IntegratedConnectome(graph, hidden_size)
    if organ.declared_trainable_parameters != 358401:
        raise RuntimeError(
            f"unexpected trainable interface count: {organ.declared_trainable_parameters}"
        )

    prompts = [
        "Ari is north of Bex. Where is Ari relative to Bex? Answer:",
        "The sequence opens with ka mo and closes in reverse order. Completion:",
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)

    base = final_logits(model, batch)

    handle = layers[midpoint_index].register_forward_hook(organ.hook)
    try:
        organ.gate.data.zero_()
        organ.reset_state()
        zero = final_logits(model, batch)
        zero_state_rms = organ.last_state_rms
        zero_steps = organ.steps

        exact_zero_identity = bool(torch.equal(base, zero))
        zero_max_abs_diff = float(torch.max(torch.abs(base - zero)).item())

        organ.gate.data.fill_(PROBE_GATE)
        organ.reset_state()
        probe_a = final_logits(model, batch)
        probe_state_rms = organ.last_state_rms
        probe_steps = organ.steps

        organ.reset_state()
        probe_b = final_logits(model, batch)
        reset_max_abs_diff = float(torch.max(torch.abs(probe_a - probe_b)).item())
        probe_parent_max_abs_diff = float(torch.max(torch.abs(base - probe_a)).item())
    finally:
        handle.remove()

    finite = all(
        math.isfinite(v)
        for v in (
            zero_state_rms,
            zero_max_abs_diff,
            probe_state_rms,
            reset_max_abs_diff,
            probe_parent_max_abs_diff,
        )
    )
    if not exact_zero_identity:
        raise RuntimeError(f"zero-gate identity failed: max_abs_diff={zero_max_abs_diff}")
    if not finite or zero_state_rms <= 0.0 or probe_state_rms <= 0.0:
        raise RuntimeError("connectome state is collapsed or non-finite")
    if reset_max_abs_diff != 0.0:
        raise RuntimeError(f"state reset is not deterministic: max_abs_diff={reset_max_abs_diff}")
    if probe_parent_max_abs_diff <= 0.0:
        raise RuntimeError("nonzero probe gate did not affect actor logits")

    receipt = {
        "schema_version": RUN_VERSION,
        "status": "completed-architecture-smoke",
        "actor": {
            "model": actor_model,
            "revision": actor_revision,
            "model_file_sha256": observed_model,
            "model_bytes": model_file.stat().st_size,
            "hidden_size": hidden_size,
            "layers": layer_count,
            "dtype": "bfloat16",
        },
        "integration": {
            "role": "actor-internal recurrent component, not sidecar",
            "midpoint_layer_human_ordinal": midpoint_index + 1,
            "midpoint_layer_zero_based_index": midpoint_index,
            "interface_width": INTERFACE_WIDTH,
            "declared_trainable_parameters": organ.declared_trainable_parameters,
            "backbone_frozen": True,
            "connectome_frozen": True,
            "probe_gate": PROBE_GATE,
            "zero_gate_exact_parent_identity": exact_zero_identity,
            "zero_gate_max_abs_logit_diff": zero_max_abs_diff,
            "zero_gate_state_rms": zero_state_rms,
            "zero_gate_state_steps": zero_steps,
            "probe_gate_parent_max_abs_logit_diff": probe_parent_max_abs_diff,
            "probe_gate_state_rms": probe_state_rms,
            "probe_gate_state_steps": probe_steps,
            "reset_repeat_max_abs_logit_diff": reset_max_abs_diff,
        },
        "graph": {
            "source": "leetae9yu/fly-as-a-lm@8787647a09f0c2730516b1f47e6bf5248a1ee0c4",
            "path": "data/central_connectome/malecns_v1_cb_intrinsic_n5600.npz",
            "sha256": GRAPH_SHA256,
            "bytes": graph_path.stat().st_size,
            "nodes": graph_nodes,
            "directed_pairs": graph_edges,
        },
        "resources": {
            "wall_seconds": time.time() - started,
        },
        "claim_boundary": (
            "Architecture/reversibility smoke only. No interface training, capability gain, "
            "biological-topology gain, scale interaction, or model promotion is established."
        ),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--actor-model", required=True)
    parser.add_argument("--actor-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    receipt = run_smoke(
        model_dir=args.model_dir,
        model_file=Path(args.model_file),
        expected_model_sha256=args.expected_model_sha256,
        graph_path=Path(args.graph),
        actor_model=args.actor_model,
        actor_revision=args.actor_revision,
        output=Path(args.output),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
