from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import canonical_bytes
from .features import response_alignment
from .types import PaperSample


def stable_sample_id(source_model: str, source_index: int, view: str, input_ids: Sequence[int]) -> str:
    ids_hash = hashlib.sha256(
        json.dumps([int(value) for value in input_ids], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    raw = f"{source_model}|{int(source_index)}|{view}|{ids_hash}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _row_identity(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    view = "model_output" if bool(int(row["hallucinated"])) else "closest_gt"
    ids_hash = hashlib.sha256(
        json.dumps(
            [int(value) for value in row["input_ids"]], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return str(row["model"]), int(row["idx"]), view, ids_hash


def build_view_manifest(
    run_id: str,
    main_samples: Sequence[PaperSample],
    view_name: str,
    auxiliary_rows: Iterable[tuple[str, int, Mapping[str, Any]]],
    *,
    track: str,
    extra_label_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Join released subset views to main samples without another model forward."""
    main_by_identity: dict[tuple[str, int, str, str], PaperSample] = {}
    for sample in main_samples:
        identity = _row_identity(sample.row)
        if identity in main_by_identity:
            raise ValueError("main samples contain a duplicate view identity")
        main_by_identity[identity] = sample

    result = []
    seen: set[tuple[str, int, str, str]] = set()
    for row_index, (source_split, source_row_index, row) in enumerate(auxiliary_rows):
        identity = _row_identity(row)
        if identity in seen:
            raise ValueError(f"view {view_name} contains a duplicate row identity")
        seen.add(identity)
        sample = main_by_identity.get(identity)
        if sample is None:
            raise ValueError(
                f"view {view_name} row {row_index} is not present in main model-forward data"
            )
        missing_labels = [field for field in extra_label_fields if field not in row]
        if missing_labels:
            raise ValueError(
                f"view {view_name} row {row_index} is missing required labels: "
                f"{missing_labels}"
            )
        if not any(
            str(occurrence.get("split")) == str(source_split)
            for occurrence in sample.occurrences
        ):
            raise ValueError(
                f"view {view_name} row {row_index} split {source_split} is absent "
                "from the matching main sample"
            )
        labels = {field: row[field] for field in extra_label_fields}
        result.append(
            {
                "schema": "paper_view_member",
                "run_id": str(run_id),
                "track": str(track),
                "view": str(view_name),
                "row_index": row_index,
                "source_split": str(source_split),
                "source_row_index": int(source_row_index),
                "sample_id": sample.sample_id,
                "occurrences": [
                    {"split": str(source_split), "row_index": int(source_row_index)}
                ],
                "source_row_sha256": hashlib.sha256(canonical_bytes(row)).hexdigest(),
                "labels": labels,
            }
        )
    return result


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.filename}")
        handle.extractall(destination)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def download_archive(url: str, expected_sha256: str, destination: Path) -> Path:
    destination = destination.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _file_sha256(destination) == expected_sha256:
        return destination

    temporary = tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    digest = hashlib.sha256()
    try:
        with temporary, urllib.request.urlopen(url) as response:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                temporary.write(chunk)
                digest.update(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded archive checksum does not match config")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _source_metadata(
    source: Mapping[str, Any],
    archive_path: Path,
    release_dir: Path,
    dataset_path: Path,
    release_tree_sha256: str,
) -> dict[str, Any]:
    del archive_path, release_dir, dataset_path
    revision = str(source["revision"])
    archive_sha256 = str(source["archive_sha256"])
    return {
        "schema": "paper_release_source",
        "repository": str(source["repository"]),
        "revision": revision,
        "archive_url": str(source["archive_url"]),
        "archive_sha256": archive_sha256,
        "release_tree_sha256": release_tree_sha256,
        "logical_cache_id": f"paper_release/{revision}/{archive_sha256[:12]}",
        "dataset_dir": str(source["dataset_dir"]),
        "source_model": str(source["source_model"]),
        "splits": list(map(str, source["splits"])),
        "views": [
            {
                "name": str(view["name"]),
                "dataset_dir": str(view["dataset_dir"]),
                "role": str(view["role"]),
                "expected_rows": int(view["expected_rows"]),
                "expected_split_counts": {
                    str(name): int(count)
                    for name, count in view.get("expected_split_counts", {}).items()
                },
            }
            for view in source["views"]
        ],
        "public_gaps": [dict(gap) for gap in source.get("public_gaps", [])],
    }


def prepare_source(
    config: Mapping[str, Any], cache_dir: Path
) -> tuple[Path, dict[str, Any]]:
    source = config["source"]
    revision = str(source["revision"])
    checksum = str(source["archive_sha256"])
    cache_root = cache_dir.expanduser() / "paper_release"
    archive_path = cache_root / "archives" / f"{revision}-{checksum[:12]}.zip"
    release_dir = cache_root / revision
    dataset_path = release_dir / str(source["data_subdir"]) / str(source["dataset_dir"])
    ready_path = release_dir / ".ready.json"
    ready_base = {
        "schema": "paper_release",
        "repository": str(source["repository"]),
        "revision": revision,
        "archive_sha256": checksum,
    }

    if ready_path.is_file():
        observed = json.loads(ready_path.read_text(encoding="utf-8"))
        if any(observed.get(name) != value for name, value in ready_base.items()):
            raise ValueError(f"cached release provenance differs: {release_dir}")
        if not dataset_path.is_dir():
            raise ValueError(f"released dataset is missing: {dataset_path}")
        tree_hash = _tree_sha256(release_dir / str(source["data_subdir"]))
        if observed.get("release_tree_sha256") != tree_hash:
            raise ValueError(f"release cache checksum differs: {release_dir}")
        metadata = _source_metadata(
            source, archive_path, release_dir, dataset_path, tree_hash
        )
        return dataset_path.resolve(), metadata

    download_archive(str(source["archive_url"]), checksum, archive_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{revision}.", dir=cache_root))
    try:
        safe_extract_zip(archive_path, temporary_dir)
        temporary_dataset = (
            temporary_dir / str(source["data_subdir"]) / str(source["dataset_dir"])
        )
        if not temporary_dataset.is_dir():
            raise ValueError(f"dataset is absent from release: {source['dataset_dir']}")
        tree_hash = _tree_sha256(temporary_dir / str(source["data_subdir"]))
        ready_value = {**ready_base, "release_tree_sha256": tree_hash}
        (temporary_dir / ".ready.json").write_bytes(canonical_bytes(ready_value) + b"\n")
        if release_dir.exists():
            shutil.rmtree(release_dir)
        os.replace(temporary_dir, release_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
    metadata = _source_metadata(source, archive_path, release_dir, dataset_path, tree_hash)
    return dataset_path.resolve(), metadata


def load_released_splits(
    dataset_path: Path, split_names: Sequence[str]
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to read the paper release") from exc
    return {
        str(name): load_from_disk(str(dataset_path / str(name)))
        for name in split_names
    }


def _load_released_view(
    path: Path, expected_split_counts: Mapping[str, Any]
) -> list[tuple[str, int, Mapping[str, Any]]]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required to read the paper release") from exc
    loaded = load_from_disk(str(path))
    if isinstance(loaded, Mapping):
        return [
            (str(split_name), int(row_index), row)
            for split_name, split_rows in loaded.items()
            for row_index, row in enumerate(split_rows)
        ]
    declared_splits = [str(name) for name in expected_split_counts]
    if len(declared_splits) != 1:
        raise ValueError(
            f"view {path.name} is a Dataset without an unambiguous declared split"
        )
    split_name = declared_splits[0]
    return [(split_name, int(row_index), row) for row_index, row in enumerate(loaded)]


def _annotation_records(
    source: Mapping[str, Any], data_root: Path
) -> list[dict[str, Any]]:
    records = []
    for annotation in source.get("annotations", []):
        relative = Path(str(annotation["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source annotation path must be release-relative")
        path = data_root / relative
        if not path.is_file():
            raise ValueError(f"released annotation file is missing: {relative.as_posix()}")
        observed = _file_sha256(path)
        if observed != str(annotation["sha256"]):
            raise ValueError(f"released annotation checksum differs: {relative.as_posix()}")
        records.append(
            {
                "schema": "paper_annotation_source",
                "name": str(annotation.get("name", relative.stem)),
                "path": relative.as_posix(),
                "sha256": observed,
                "bytes": path.stat().st_size,
                "source_file": path,
            }
        )
    return records


def prepare_samples(
    config: Mapping[str, Any], cache_dir: Path
) -> tuple[
    list[PaperSample],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    dataset_path, metadata = prepare_source(config, cache_dir)
    samples, manifest = build_manifest(
        config,
        load_released_splits(dataset_path, config["source"]["splits"]),
    )
    view_manifests: dict[str, list[dict[str, Any]]] = {}
    for view in config["source"]["views"]:
        if view["role"] == "model_forward":
            continue
        rows = _load_released_view(
            dataset_path.parent / str(view["dataset_dir"]),
            view.get("expected_split_counts", {}),
        )
        if len(rows) != int(view["expected_rows"]):
            raise ValueError(
                f"view {view['name']} expected {view['expected_rows']} rows, found {len(rows)}"
            )
        view_manifests[str(view["name"])] = build_view_manifest(
            str(config["run_id"]),
            samples,
            str(view["name"]),
            rows,
            track=str(config["track"]),
            extra_label_fields=list(view.get("extra_label_fields", [])),
        )
        expected_split_counts = view.get("expected_split_counts")
        if expected_split_counts is not None:
            observed = Counter(
                str(member["source_split"])
                for member in view_manifests[str(view["name"])]
            )
            expected = {
                str(name): int(count) for name, count in expected_split_counts.items()
            }
            if dict(observed) != expected:
                raise ValueError(
                    f"view {view['name']} split counts differ: expected {expected}, "
                    f"found {dict(observed)}"
                )
    annotations = _annotation_records(config["source"], dataset_path.parent)
    metadata["annotations"] = [
        {key: value for key, value in record.items() if key != "source_file"}
        for record in annotations
    ]
    auxiliary = {"views": view_manifests, "annotations": annotations}
    return samples, manifest, auxiliary, metadata


def build_manifest(
    config: Mapping[str, Any], splits: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[PaperSample], list[dict[str, Any]]]:
    source = config["source"]
    expected_model = str(source["source_model"])
    unique: dict[str, dict[str, Any]] = {}
    physical_rows = 0
    observed_datasets: set[str] = set()

    for split, rows in splits.items():
        for row_index, row in enumerate(rows):
            physical_rows += 1
            if str(row["model"]) != expected_model:
                raise ValueError(f"unexpected source model: {row['model']}")
            observed_datasets.add(str(row["dataset"]))
            alignment = response_alignment(row["input_ids"], row["uncertainty_labels"])
            view = "model_output" if bool(int(row["hallucinated"])) else "closest_gt"
            sample_id = stable_sample_id(expected_model, int(row["idx"]), view, row["input_ids"])
            row_hash = hashlib.sha256(canonical_bytes(row)).hexdigest()
            occurrence = {"split": str(split), "row_index": row_index}
            if sample_id in unique:
                if unique[sample_id]["source_row_sha256"] != row_hash:
                    raise ValueError(f"duplicate sample has different content: {sample_id}")
                unique[sample_id]["occurrences"].append(occurrence)
                continue
            unique[sample_id] = {
                "sample_id": sample_id,
                "row": row,
                "source_row_sha256": row_hash,
                "source_idx": int(row["idx"]),
                "dataset": str(row["dataset"]),
                "task_id": str(row["task_id"]),
                "view": view,
                "response_start": alignment.response_start,
                "total_tokens": len(row["input_ids"]),
                "occurrences": [occurrence],
            }

    if physical_rows != int(source["expected_physical_rows"]):
        raise ValueError(
            f"expected {source['expected_physical_rows']} physical rows, found {physical_rows}"
        )
    if len(unique) != int(source["expected_unique_samples"]):
        raise ValueError(
            f"expected {source['expected_unique_samples']} unique samples, found {len(unique)}"
        )
    expected_datasets = set(map(str, source["datasets"]))
    if observed_datasets != expected_datasets:
        raise ValueError(
            f"released datasets differ: expected {sorted(expected_datasets)}, "
            f"found {sorted(observed_datasets)}"
        )

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item["source_idx"],
            0 if item["view"] == "model_output" else 1,
            item["sample_id"],
        ),
    )
    samples = [
        PaperSample(
            sample_id=item["sample_id"],
            row=item["row"],
            occurrences=tuple(item["occurrences"]),
            source_row_sha256=item["source_row_sha256"],
        )
        for item in ordered
    ]
    manifest = [
        {
            "schema": "paper_manifest_sample",
            "run_id": str(config["run_id"]),
            "track": str(config["track"]),
            "sample_id": item["sample_id"],
            "source_idx": item["source_idx"],
            "dataset": item["dataset"],
            "task_id": item["task_id"],
            "view": item["view"],
            "response_start": item["response_start"],
            "total_tokens": item["total_tokens"],
            "source_row_sha256": item["source_row_sha256"],
            "occurrences": item["occurrences"],
        }
        for item in ordered
    ]
    return samples, manifest
