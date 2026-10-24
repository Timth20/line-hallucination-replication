"""Load the extracted records of a split and align their code lines with the feature rows.

Feature row r predicts token r + 1, so a line is the set of rows that predict its tokens.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value["data"] if isinstance(value, dict) and "data" in value and "sha256" in value else value


def _find_subsequence(tokens: Sequence[Any], claim_tokens: Sequence[Any], start: int):
    width = len(claim_tokens)
    for offset in range(start, len(tokens) - width + 1):
        if list(tokens[offset:offset + width]) == list(claim_tokens):
            return offset, offset + width - 1
    return None


def line_spans(output: Mapping[str, Any], *, predictor_rows: int) -> list[tuple[int, int]]:
    """First and last feature row of each released line, from its token offsets or its tokens matched in order."""
    claims, tokens, response_start = output["claims"], output["input_tokens"], output["response_start"]
    response_tokens = tokens[response_start:]
    search_start, previous_line, spans = 0, -1, []
    for claim in claims:
        if "start" in claim or "end" in claim:
            start, end = int(claim["start"]), int(claim["end"])
        else:
            claim_tokens, line = claim["claim_text"], claim["line"]
            if not claim_tokens or line < previous_line:
                raise ValueError("malformed line")
            match = _find_subsequence(response_tokens, claim_tokens, search_start)
            if match is None:
                raise ValueError("line does not align with the response tokens")
            start, end = match
            search_start, previous_line = end + 1, line
        first, last = response_start + start - 1, response_start + end - 1
        if start < 0 or end < start or first < 0 or last >= predictor_rows:
            raise ValueError("line span outside the response")
        spans.append((first, last))
    return spans


def masks_released(output, labels, predictor_rows):
    if output["claims"] != labels["claims"] or len(labels["verified"]) != len(output["claims"]):
        raise ValueError("lines of output and labels differ")
    masks = torch.zeros((len(output["claims"]), predictor_rows), dtype=torch.bool)
    for i, (first, last) in enumerate(line_spans(output, predictor_rows=predictor_rows)):
        masks[i, first:last + 1] = True
    return masks, torch.tensor(labels["verified"], dtype=torch.float32)


def masks_generated(output, labels):
    """Generated responses: each line gives the positions of its tokens in the response."""
    p = len(output["input_ids"]) - 1
    start = output["response_start"] - 1
    masks = torch.zeros(len(labels["claims"]), p, dtype=torch.bool)
    seen = set()
    for c, claim in enumerate(labels["claims"]):
        indices = [start + i for i in claim["aligned_token_ids"]]
        if not indices or min(indices) < start or max(indices) >= p or len(set(indices)) != len(indices) \
                or seen.intersection(indices):
            raise ValueError("invalid line position")
        masks[c, indices] = True
        seen.update(indices)
    y = torch.tensor(labels["verified"], dtype=torch.float32)
    if len(y) != len(masks) or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("invalid line labels")
    return masks, y


class Sample:
    """One record: line masks, labels and token probabilities; other features are read on demand."""

    def __init__(self, model_dir, dataset, sample_id, row):
        base = Path(model_dir) / dataset / sample_id
        self.output = read_json(f"{base}.output.json")
        self.labels = read_json(f"{base}.labels.json")
        self.features_path = f"{base}.features.safetensors"
        with safe_open(self.features_path, framework="pt") as f:
            self.metadata = dict(f.metadata() or {})
            self.top_p = f.get_tensor("topk_probabilities")
            self.top_ids = f.get_tensor("topk_token_ids")
            self.next_ids = f.get_tensor("next_token_ids")
        rows = int(self.top_p.shape[0])
        if "aligned_token_ids" in (self.labels["claims"][0] if self.labels["claims"] else {}):
            self.claim_masks, self.label_values = masks_generated(self.output, self.labels)
        else:
            self.claim_masks, self.label_values = masks_released(self.output, self.labels, rows)
        if self.claim_masks.shape[1] != rows:
            raise ValueError("line masks and feature rows differ: " + sample_id)
        self.response_row = int(self.output["response_start"]) - 1
        self.row = dict(row)

    def tensor(self, name):
        with safe_open(self.features_path, framework="pt") as f:
            return f.get_tensor(name)


def load_split(model_dir, split_csv, role):
    """Records of one role ("train", "validation" or "test") of a split file, in file order."""
    with open(split_csv, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == role]
    samples = [Sample(model_dir, r["dataset"], r["sample_id"], r) for r in rows]
    for s, r in zip(samples, rows):
        if len(s.label_values) != int(r["lines"]):
            raise ValueError("line count differs from the split file: " + r["sample_id"])
    return samples
