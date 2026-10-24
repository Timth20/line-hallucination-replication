from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Alignment:
    response_start: int
    target_positions: tuple[int, ...]
    predictor_positions: tuple[int, ...]


@dataclass(frozen=True)
class PaperSample:
    sample_id: str
    row: Mapping[str, Any]
    occurrences: tuple[Mapping[str, Any], ...]
    source_row_sha256: str


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Any
    model: Any
    device: str
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class FeatureSet:
    tensors: Mapping[str, Any]
    metadata: Mapping[str, str]
