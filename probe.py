#!/usr/bin/env python3
"""Measure sparse residual-placement strategies on exact transformer weight tensors.

Disposable execution copy of Helix Model run 013's probe. Canonical experiment
ownership remains in dissonance-git/helix-model on GitLab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "helix-model-weight-residual-probe-001.0"
DEFAULT_SCALES = (1, 2, 4, 8)
HELIX_SCALE_WEIGHTS = (1, 3, 5, 7)
UNIFORM_SCALE_WEIGHTS = (1, 1, 1, 1)

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

def safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    model_path = Path(path)
    with model_path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ProbeError("safetensors file is shorter than its 8-byte header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 0 or header_len > model_path.stat().st_size - 8:
            raise ProbeError(f"invalid safetensors header length: {header_len}")
        header_raw = handle.read(header_len)
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except Exception as exc:
        raise ProbeError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ProbeError("safetensors header must be an object")
    return header, 8 + header_len

def tensor_names(path: str | Path) -> list[str]:
    header, _ = safetensors_header(path)
    return sorted(name for name in header if name != "__metadata__")

def load_tensor(path: str | Path, name: str) -> np.ndarray:
    header, data_start = safetensors_header(path)
    meta = header.get(name)
    if not isinstance(meta, dict):
        raise ProbeError(f"tensor not found: {name}")
    dtype = str(meta.get("dtype", "")).upper()
    shape = tuple(int(x) for x in meta.get("shape", []))
    offsets = meta.get("data_offsets")
    if not shape or len(shape) != 2:
        raise ProbeError(f"{name} must be a 2-D matrix, got shape={shape!r}")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ProbeError(f"{name} has invalid data_offsets")
    start, end = (int(offsets[0]), int(offsets[1]))
    count = math.prod(shape)
    expected_bytes = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
    if expected_bytes is None:
        raise ProbeError(f"{name} dtype {dtype!r} is unsupported")
    if end - start != count * expected_bytes:
        raise ProbeError(f"{name} byte span does not match shape/dtype")
    offset = data_start + start
    if dtype == "BF16":
        raw = np.memmap(path, dtype="<u2", mode="r", offset=offset, shape=(count,))
        words = raw.astype(np.uint32)
        values = (words << np.uint32(16)).view(np.float32)
    elif dtype == "F16":
        raw = np.memmap(path, dtype="<f2", mode="r", offset=offset, shape=(count,))
        values = np.asarray(raw, dtype=np.float32)
    else:
        raw = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count,))
        values = np.asarray(raw, dtype=np.float32).copy()
    return np.asarray(values, dtype=np.float32).reshape(shape)

def randomized_low_rank(matrix: np.ndarray, rank: int, *, seed: int, oversample: int = 8, power_iterations: int = 1) -> np.ndarray:
    rows, cols = matrix.shape
    max_rank = min(rows, cols)
    if not 0 < rank < max_rank:
        raise ProbeError(f"rank must be in [1,{max_rank - 1}], got {rank}")
    width = min(max_rank, rank + max(0, int(oversample)))
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((cols, width), dtype=np.float32)
    sample = matrix @ omega
    for _ in range(max(0, int(power_iterations))):
        sample = matrix @ (matrix.T @ sample)
    q, _ = np.linalg.qr(sample, mode="reduced")
    compressed = q.T @ matrix
    u_small, singular, vh = np.linalg.svd(compressed, full_matrices=False)
    u = q @ u_small[:, :rank]
    return ((u * singular[:rank]) @ vh[:rank, :]).astype(np.float32, copy=False)

def _weighted_quota(total: int, weights: Iterable[int]) -> list[int]:
    values = [int(x) for x in weights]
    if total < 0 or not values or any(x <= 0 for x in values):
        raise ProbeError("quota weights must be positive and total must be non-negative")
    denominator = sum(values)
    raw = [total * value / denominator for value in values]
    quotas = [math.floor(value) for value in raw]
    remainder = total - sum(quotas)
    order = sorted(range(len(values)), key=lambda i: (-(raw[i] - quotas[i]), i))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas

def _even_quota(total: int, buckets: int) -> list[int]:
    if buckets <= 0:
        raise ProbeError("buckets must be positive")
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]

def magnitude_indices(residual: np.ndarray, count: int) -> np.ndarray:
    flat = np.abs(residual).ravel()
    count = min(max(0, int(count)), flat.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    if count == flat.size:
        return np.arange(flat.size, dtype=np.int64)
    candidate = np.argpartition(flat, flat.size - count)[-count:]
    return candidate[np.lexsort((candidate, -flat[candidate]))].astype(np.int64, copy=False)

def random_indices(size: int, count: int, *, seed: int) -> np.ndarray:
    count = min(max(0, int(count)), int(size))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=count, replace=False).astype(np.int64, copy=False))

def multiscale_indices(residual: np.ndarray, count: int, *, scale_weights: tuple[int, ...], scales: tuple[int, ...] = DEFAULT_SCALES) -> np.ndarray:
    if len(scales) != len(scale_weights):
        raise ProbeError("scales and scale_weights must have equal length")
    rows, cols = residual.shape
    total = residual.size
    count = min(max(0, int(count)), total)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    selected = np.zeros(total, dtype=np.bool_)
    abs_flat = np.abs(residual).ravel()
    per_scale = _weighted_quota(count, scale_weights)
    for scale, scale_quota in zip(scales, per_scale):
        if scale <= 0:
            raise ProbeError("scales must be positive")
        per_block = _even_quota(scale_quota, scale * scale)
        block_index = 0
        for row_block in range(scale):
            r0 = row_block * rows // scale
            r1 = (row_block + 1) * rows // scale
            for col_block in range(scale):
                quota = per_block[block_index]
                block_index += 1
                if quota <= 0 or r1 <= r0:
                    continue
                c0 = col_block * cols // scale
                c1 = (col_block + 1) * cols // scale
                if c1 <= c0:
                    continue
                rr = np.arange(r0, r1, dtype=np.int64)[:, None]
                cc = np.arange(c0, c1, dtype=np.int64)[None, :]
                block_global = (rr * cols + cc).ravel()
                already = int(np.count_nonzero(selected[block_global]))
                candidate_count = min(block_global.size, quota + already)
                block_abs = abs_flat[block_global]
                if candidate_count == block_global.size:
                    candidates = np.arange(block_global.size, dtype=np.int64)
                else:
                    candidates = np.argpartition(block_abs, block_global.size - candidate_count)[-candidate_count:]
                candidate_global = block_global[candidates]
                candidate_global = candidate_global[np.lexsort((candidate_global, -abs_flat[candidate_global]))]
                available = candidate_global[~selected[candidate_global]]
                selected[available[:quota]] = True
    selected_count = int(np.count_nonzero(selected))
    if selected_count < count:
        remaining = count - selected_count
        available = np.flatnonzero(~selected)
        values = abs_flat[available]
        if remaining < available.size:
            local = np.argpartition(values, available.size - remaining)[-remaining:]
            fill = available[local]
            fill = fill[np.lexsort((fill, -abs_flat[fill]))]
        else:
            fill = available
        selected[fill[:remaining]] = True
    indices = np.flatnonzero(selected).astype(np.int64, copy=False)
    if indices.size != count:
        raise ProbeError(f"multiscale selection count mismatch: {indices.size} != {count}")
    return indices

def reconstruction_metrics(matrix: np.ndarray, residual: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    rows, cols = matrix.shape
    flat = residual.ravel()
    selected = np.asarray(selected, dtype=np.int64)
    selected_sq = np.square(flat[selected], dtype=np.float64) if selected.size else np.empty(0)
    residual_sq = np.square(residual, dtype=np.float64)
    total_residual_sq = float(residual_sq.sum(dtype=np.float64))
    retained_sq = float(selected_sq.sum(dtype=np.float64))
    remaining_sq = max(0.0, total_residual_sq - retained_sq)
    matrix_sq = np.square(matrix, dtype=np.float64)
    matrix_norm = math.sqrt(float(matrix_sq.sum(dtype=np.float64)))
    residual_norm = math.sqrt(total_residual_sq)
    eps = 1e-30
    row_remaining = residual_sq.sum(axis=1, dtype=np.float64)
    col_remaining = residual_sq.sum(axis=0, dtype=np.float64)
    if selected.size:
        selected_rows = selected // cols
        selected_cols = selected % cols
        row_remaining -= np.bincount(selected_rows, weights=selected_sq, minlength=rows)
        col_remaining -= np.bincount(selected_cols, weights=selected_sq, minlength=cols)
    row_remaining = np.maximum(row_remaining, 0.0)
    col_remaining = np.maximum(col_remaining, 0.0)
    row_base = np.sqrt(matrix_sq.sum(axis=1, dtype=np.float64))
    col_base = np.sqrt(matrix_sq.sum(axis=0, dtype=np.float64))
    row_rel = np.sqrt(row_remaining) / np.maximum(row_base, eps)
    col_rel = np.sqrt(col_remaining) / np.maximum(col_base, eps)
    unique_rows = int(np.unique(selected // cols).size) if selected.size else 0
    unique_cols = int(np.unique(selected % cols).size) if selected.size else 0
    return {
        "selected_coordinates": int(selected.size),
        "relative_frobenius_error": math.sqrt(remaining_sq) / max(matrix_norm, eps),
        "remaining_residual_fraction": math.sqrt(remaining_sq) / max(residual_norm, eps),
        "residual_energy_retained": retained_sq / max(total_residual_sq, eps),
        "worst_row_relative_error": float(row_rel.max(initial=0.0)),
        "p95_row_relative_error": float(np.quantile(row_rel, 0.95)),
        "mean_row_relative_error": float(row_rel.mean()),
        "worst_column_relative_error": float(col_rel.max(initial=0.0)),
        "p95_column_relative_error": float(np.quantile(col_rel, 0.95)),
        "mean_column_relative_error": float(col_rel.mean()),
        "row_coverage_fraction": unique_rows / rows,
        "column_coverage_fraction": unique_cols / cols,
    }

def summarize_replicates(rows: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max())}
    return out

def probe_tensor(matrix: np.ndarray, *, tensor_name: str, ranks: list[int], budgets: list[float], random_replicates: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in ranks:
        low_rank = randomized_low_rank(matrix, rank, seed=seed + rank)
        residual = (matrix - low_rank).astype(np.float32, copy=False)
        rows.append({"tensor": tensor_name, "shape": list(matrix.shape), "rank": rank, "residual_fraction": 0.0, "arm": "low-rank-only", "metrics": reconstruction_metrics(matrix, residual, np.empty(0, dtype=np.int64))})
        for fraction in budgets:
            if not 0 < fraction < 1:
                raise ProbeError(f"residual fraction must be between 0 and 1: {fraction}")
            count = max(1, int(round(matrix.size * fraction)))
            magnitude = magnitude_indices(residual, count)
            rows.append({"tensor": tensor_name, "shape": list(matrix.shape), "rank": rank, "residual_fraction": fraction, "arm": "magnitude-residual", "metrics": reconstruction_metrics(matrix, residual, magnitude)})
            helix = multiscale_indices(residual, count, scale_weights=HELIX_SCALE_WEIGHTS)
            rows.append({"tensor": tensor_name, "shape": list(matrix.shape), "rank": rank, "residual_fraction": fraction, "arm": "multiscale-coverage-residual", "scale_weights": list(HELIX_SCALE_WEIGHTS), "metrics": reconstruction_metrics(matrix, residual, helix)})
            uniform = multiscale_indices(residual, count, scale_weights=UNIFORM_SCALE_WEIGHTS)
            rows.append({"tensor": tensor_name, "shape": list(matrix.shape), "rank": rank, "residual_fraction": fraction, "arm": "uniform-block-coverage-residual", "scale_weights": list(UNIFORM_SCALE_WEIGHTS), "metrics": reconstruction_metrics(matrix, residual, uniform)})
            random_rows: list[dict[str, float | int]] = []
            for replicate in range(random_replicates):
                indices = random_indices(residual.size, count, seed=seed + rank * 100000 + int(fraction * 1_000_000) + replicate)
                random_rows.append(reconstruction_metrics(matrix, residual, indices))
            rows.append({"tensor": tensor_name, "shape": list(matrix.shape), "rank": rank, "residual_fraction": fraction, "arm": "random-residual", "replicates": random_replicates, "summary": summarize_replicates(random_rows), "raw_replicates": random_rows})
    return rows

def environment_receipt() -> dict[str, Any]:
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count()}

def self_test() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((24, 40), dtype=np.float32)
    low = randomized_low_rank(matrix, 4, seed=11)
    residual = matrix - low
    count = 37
    for weights in (HELIX_SCALE_WEIGHTS, UNIFORM_SCALE_WEIGHTS):
        indices = multiscale_indices(residual, count, scale_weights=weights)
        assert indices.size == count
        assert np.unique(indices).size == count
    magnitude = magnitude_indices(residual, count)
    random = random_indices(residual.size, count, seed=13)
    for indices in (magnitude, random):
        assert indices.size == count
        metrics = reconstruction_metrics(matrix, residual, indices)
        assert 0.0 <= float(metrics["residual_energy_retained"]) <= 1.0
    mag_energy = reconstruction_metrics(matrix, residual, magnitude)["residual_energy_retained"]
    random_energy = reconstruction_metrics(matrix, residual, random)["residual_energy_retained"]
    assert float(mag_energy) >= float(random_energy)
    print(json.dumps({"self_test": "pass", "schema_version": SCHEMA_VERSION}, sort_keys=True))

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--budget", action="append", type=float, default=[])
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=13013)
    parser.add_argument("--output")
    parser.add_argument("--list-tensors", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if not args.model_file:
        raise ProbeError("--model-file is required")
    model_file = Path(args.model_file)
    if not model_file.is_file():
        raise ProbeError(f"model file not found: {model_file}")
    if args.list_tensors:
        print("\n".join(tensor_names(model_file))); return 0
    expected = str(args.expected_sha256 or "").strip().lower()
    if len(expected) != 64:
        raise ProbeError("--expected-sha256 must be a 64-character digest")
    started = time.time()
    actual = sha256_file(model_file)
    if actual != expected:
        raise ProbeError(f"source SHA-256 mismatch: actual={actual} expected={expected}")
    if not args.tensor:
        raise ProbeError("at least one --tensor is required")
    ranks = args.rank or [16, 32]
    budgets = args.budget or [0.005, 0.01, 0.02]
    if args.random_replicates < 1:
        raise ProbeError("--random-replicates must be positive")
    results: list[dict[str, Any]] = []
    for name in args.tensor:
        matrix = load_tensor(model_file, name)
        results.extend(probe_tensor(matrix, tensor_name=name, ranks=ranks, budgets=budgets, random_replicates=args.random_replicates, seed=args.seed))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "source": {"file": model_file.name, "sha256": actual, "bytes": model_file.stat().st_size},
        "environment": environment_receipt(),
        "settings": {"tensors": args.tensor, "ranks": ranks, "residual_budgets": budgets, "random_replicates": args.random_replicates, "seed": args.seed, "low_rank": {"method": "fixed-seed randomized low-rank approximation", "oversample": 8, "power_iterations": 1}, "multiscale": {"scales": list(DEFAULT_SCALES), "helix_scale_weights": list(HELIX_SCALE_WEIGHTS), "uniform_scale_weights": list(UNIFORM_SCALE_WEIGHTS)}},
        "results": results,
        "wall_seconds": time.time() - started,
        "claim_boundary": "weight-reconstruction evidence only; no model-forward-pass, capability, mathematical-transfer, or promotion claim",
    }
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
