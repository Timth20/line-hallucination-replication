from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .config import implementation_hash


@dataclass(frozen=True)
class SamplePaths:
    directory: Path
    output: Path
    labels: Path
    features: Path
    done: Path


def _safe_component(value: Any, name: str) -> str:
    component = str(value)
    if (
        component in {"", ".", ".."}
        or "/" in component
        or "\\" in component
        or "\0" in component
    ):
        raise ValueError(f"{name} path component is invalid")
    return component


def model_output_dir(output_dir: Path, config: Mapping[str, Any]) -> Path:
    model = _safe_component(config["source"]["source_model"], "model")
    return output_dir / model


def sample_paths(
    output_dir: Path, model: str, dataset: str, sample_id: str
) -> SamplePaths:
    model = _safe_component(model, "model")
    dataset = _safe_component(dataset, "dataset")
    sample_id = _safe_component(sample_id, "sample_id")
    directory = output_dir / model / dataset
    return SamplePaths(
        directory=directory,
        output=directory / f"{sample_id}.output.json",
        labels=directory / f"{sample_id}.labels.json",
        features=directory / f"{sample_id}.features.safetensors",
        done=directory / f"{sample_id}.done.json",
    )


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _save_safetensors(
    tensors: Mapping[str, Any], filename: str, metadata: Mapping[str, str]
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required") from exc
    save_file(dict(tensors), filename, metadata=dict(metadata))


def atomic_save_safetensors(
    path: Path,
    tensors: Mapping[str, Any],
    metadata: Mapping[str, str],
    save_file: Callable[[Mapping[str, Any], str, Mapping[str, str]], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    writer = save_file or _save_safetensors
    try:
        writer(tensors, str(temporary), metadata)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_protocol(
    output_dir: Path,
    config: Mapping[str, Any],
    protocol_sha256: str,
    *,
    create: bool = True,
) -> None:
    path = output_dir / "metadata" / "run_protocol.json"
    value = {
        "schema": "saner_run_protocol",
        "track": str(config["track"]),
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_hash(),
        "config": config,
    }
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("output directory belongs to a different protocol")
        return
    if not create:
        raise ValueError("run is not prepared")
    artifact_patterns = (
        "*.output.json",
        "*.labels.json",
        "*.features.safetensors",
        "*.done.json",
    )
    if any(any(output_dir.rglob(pattern)) for pattern in artifact_patterns):
        raise ValueError("sample artifacts exist without a run protocol")
    atomic_write_json(path, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_done(
    paths: SamplePaths,
    sample_id: str,
    protocol_sha256: str,
    run_id: str,
    source_row_sha256: str,
    *,
    track: str,
) -> None:
    files = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (paths.output, paths.labels, paths.features)
    }
    atomic_write_json(
        paths.done,
        {
            "schema": "saner_sample_done",
            "run_id": run_id,
            "track": track,
            "sample_id": sample_id,
            "protocol_sha256": protocol_sha256,
            "source_row_sha256": source_row_sha256,
            "files": files,
        },
    )


def write_sample_artifacts(
    output_dir: Path,
    run_id: str,
    sample: Any,
    feature_set: Any,
    protocol_sha256: str,
    response_start: int,
    *,
    track: str,
) -> SamplePaths:
    row = sample.row
    sample_id = str(sample.sample_id)
    paths = sample_paths(output_dir, str(row["model"]), str(row["dataset"]), sample_id)
    output_fields = (
        "idx",
        "model",
        "dataset",
        "task_id",
        "meta",
        "hallucinated",
        "templated_question",
        "question",
        "answer",
        "reply",
        "claims",
        "input_tokens",
        "input_ids",
    )
    output = {name: row[name] for name in output_fields if name in row}
    source_claims = list(row.get("claims", []))
    output.update(
        {
            "schema": "saner_paper_output",
            "run_id": run_id,
            "track": track,
            "sample_id": sample_id,
            "protocol_sha256": protocol_sha256,
            "view": "model_output" if bool(int(row["hallucinated"])) else "closest_gt",
            "response_start": response_start,
            "source_row_sha256": sample.source_row_sha256,
            "occurrences": list(sample.occurrences),
        }
    )
    labels = {
        "schema": "saner_paper_labels",
        "run_id": run_id,
        "track": track,
        "sample_id": sample_id,
        "protocol_sha256": protocol_sha256,
        "source_row_sha256": sample.source_row_sha256,
        "hallucinated": int(row["hallucinated"]),
        "uncertainty_labels": row["uncertainty_labels"],
        "claims": source_claims,
        "verified": row.get("verified", []),
    }
    if "diff" in row:
        labels["diff"] = row["diff"]
    metadata = {str(key): str(value) for key, value in feature_set.metadata.items()}
    metadata["run_id"] = run_id
    metadata["track"] = track
    metadata["sample_id"] = sample_id
    metadata["protocol_sha256"] = protocol_sha256
    metadata["source_row_sha256"] = sample.source_row_sha256

    atomic_write_json(paths.output, output)
    atomic_write_json(paths.labels, labels)
    atomic_save_safetensors(paths.features, feature_set.tensors, metadata)
    return paths
