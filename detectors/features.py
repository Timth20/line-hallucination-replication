"""Token features of a record: the feature blocks of its detector configuration, concatenated in order.

RichTP has 14 values per row: the eight top probabilities, their renormalized entropy, and five statistics of the line.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def _layers(metadata, name):
    value = json.loads(metadata[name])
    if not isinstance(value, list):
        raise ValueError(f"layer metadata is not a list: {name}")
    return [int(item) for item in value]


def _indices(available, requested, block):
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(f"{block} layers absent from the features: {missing}")
    return [available.index(layer) for layer in requested]


def _flatten(tensor):
    return tensor.reshape(tensor.shape[0], -1).float()


def richtp(sample):
    p = sample.top_p.float().clamp(0, 1)
    if p.ndim != 2 or p.shape[1] != 8:
        raise ValueError("expected eight probabilities per row")
    q = p / p.sum(dim=1, keepdim=True).clamp_min(1e-30)
    entropy = -(q * q.clamp_min(1e-30).log()).sum(dim=1)
    base = torch.cat([p, entropy[:, None]], dim=1)
    neglog = -base[:, 0].double().clamp_min(1e-30).log()
    ent = base[:, 8].double()
    line = torch.zeros(base.shape[0], 5, dtype=torch.float64)
    for mask in sample.claim_masks:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise ValueError("a line without rows")
        a, e = neglog[idx], ent[idx]
        line[idx] = torch.stack([a.mean(), a.max(), e.mean(), e.max(), a.std(unbiased=False)])
    return torch.cat([base, line.float()], dim=1)


class StudentStates:
    """Hidden states of a Student on the Teacher's tokens (written by distillation/)."""

    def __init__(self, root):
        self.root = Path(root)

    def get(self, sample):
        path = self.root / sample.row["dataset"] / f"{sample.row['sample_id']}.features.safetensors"
        with safe_open(str(path), framework="pt") as f:
            return f.get_tensor("hidden_states"), dict(f.metadata() or {})


def build(sample, blocks, students=None):
    """Feature matrix [rows, width] of one record."""
    rows = int(sample.top_p.shape[0])
    response_row = sample.response_row
    out = []
    for spec in blocks:
        name = str(spec["name"]).lower()
        if name == "tp":
            value = sample.top_p[:, :int(spec["top_n"])].float().clone()
            if bool(spec.get("zero_prompt", True)):
                value[:response_row] = 0
        elif name == "richtp":
            value = richtp(sample)
            value[:response_row] = 0
        elif name == "hs":
            requested = list(map(int, spec["layers"]))
            if spec.get("source"):
                states, metadata = students[spec["source"]].get(sample)
            else:
                states, metadata = sample.tensor("hidden_states"), sample.metadata
            chosen = _indices(_layers(metadata, "hidden_state_layers"), requested, "hidden states")
            value = _flatten(states[:, chosen, :])
        elif name == "aw":
            chosen = _indices(_layers(sample.metadata, "attention_layers"), list(map(int, spec["layers"])),
                              "attention window")
            raw = sample.tensor("attention_window")[:, :int(spec["history"]), :, chosen].float()
            if bool(spec.get("pool", False)):
                raw = raw.max(dim=-1).values
            value = _flatten(raw)
        elif name == "layer_logits":
            chosen = _indices(_layers(sample.metadata, "layer_top_logits_layers"), list(map(int, spec["layers"])),
                              "layer logits")
            logits = sample.tensor("layer_top_logits")
            value = _flatten(logits[:, chosen, :int(spec.get("top_n", logits.shape[-1]))])
        elif name == "ts":
            requested = list(map(int, spec["layers"]))
            available = _layers(sample.metadata, "token_similarity_tuple_indices")
            if len(requested) < 2:
                raise ValueError("token similarity needs two or more configured layers")
            chosen = _indices(list(zip(available[:-1], available[1:])), list(zip(requested[:-1], requested[1:])),
                              "token similarity")
            sims = sample.tensor("token_similarities")
            value = _flatten(sims[:, chosen, :int(spec.get("top_n", sims.shape[-1]))])
        elif name == "or":
            chosen = _indices(_layers(sample.metadata, "output_rank_layers"), list(map(int, spec["layers"])),
                              "output ranks")
            ranks = sample.tensor("output_ranks")[:, chosen].float()
            if (ranks < 1).any():
                raise ValueError("output ranks must be positive integers")
            value = ranks.reciprocal()
        elif name == "lr":
            chosen = _indices(_layers(sample.metadata, "lookback_layers"), list(map(int, spec["layers"])),
                              "lookback ratio")
            value = _flatten(sample.tensor("lookback_lens")[:, chosen, :])
        else:
            raise ValueError(f"unknown feature block: {name}")
        if value.shape[0] != rows:
            raise ValueError(f"feature rows differ for {name}")
        out.append(value)
    matrix = torch.cat(out, dim=1)
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("nonfinite features")
    return matrix
