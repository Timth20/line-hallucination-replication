from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

from .artifacts import (
    atomic_copy_file,
    atomic_write_json,
    atomic_write_jsonl,
    ensure_protocol,
    model_output_dir,
    sample_paths,
    sha256_file,
    write_done,
    write_sample_artifacts,
)
from .config import canonical_bytes, protocol_hash, validate_output_root
from .features import response_alignment
from .runtime import (
    validate_execution_context,
    validate_frozen_environment,
    validate_model_against_config,
    validate_runtime_environment,
)
from .smoke import (
    smoke_sample_start,
    validate_smoke_marker,
    validate_smoke_runtime,
    write_smoke_marker,
)
from .validation import (
    _environment_sha256,
    _execution_inventory_sha256,
    _open_safetensors,
    _validate_completion_marker,
    sample_complete,
    status as validation_status,
    validate_written_sample,
    verify as validation_verify,
)


SampleT = TypeVar("SampleT")


def _restore_smoke_marker(
    run_dir: Path,
    config: Mapping[str, Any],
    protocol_sha256: str,
    sample_start: int,
    sample: Any,
    paths: Any,
) -> bool:
    marker = run_dir / "state" / "SMOKE_PASSED.json"
    if marker.is_file():
        if not paths.done.is_file():
            raise RuntimeError("smoke gate marker exists without its completed artifact")
        validate_smoke_marker(
            run_dir / "state",
            run_id=str(config["run_id"]),
            track=str(config["track"]),
            protocol_sha256=protocol_sha256,
            sample_start=sample_start,
            sample_id=str(sample.sample_id),
            done_path=paths.done,
            environment_sha256=_environment_sha256(run_dir, config),
        )
        return True
    if not paths.done.is_file():
        return False
    with _open_safetensors(paths.features) as handle:
        metadata = handle.metadata() or {}
    runtime_profile = {
        name: metadata.get(name)
        for name in (
            "gpu_total_memory_bytes",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
            "attention_dtype",
        )
    }
    validate_smoke_runtime(str(config["run_id"]), runtime_profile)
    write_smoke_marker(
        run_dir / "state",
        run_id=str(config["run_id"]),
        track=str(config["track"]),
        protocol_sha256=protocol_sha256,
        sample_start=sample_start,
        sample_id=str(sample.sample_id),
        done_path=paths.done,
        environment_sha256=_environment_sha256(run_dir, config),
        runtime_profile=runtime_profile,
    )
    return True


def prepare_samples(config: Mapping[str, Any], cache_dir: Path):
    from .datasets import prepare_samples as implementation

    return implementation(config, cache_dir)


def load_model(config: Mapping[str, Any], cache_dir: Path, device: str = "cuda:0"):
    from .models import load_model as implementation

    return implementation(config, cache_dir, device=device)


def extract_features(config: Mapping[str, Any], sample: Any, bundle: Any, protocol_sha256: str):
    from .features import extract_features as implementation

    return implementation(config, sample, bundle, protocol_sha256)


def pending_samples(
    samples: Iterable[SampleT],
    output_dir: Path,
    protocol_sha256: str,
    *,
    run_id: str,
    config: Mapping[str, Any] | None = None,
) -> list[SampleT]:
    pending = []
    for sample in samples:
        sample_id = str(getattr(sample, "sample_id"))
        paths = sample_paths(
            output_dir,
            str(getattr(sample, "row")["model"]),
            str(getattr(sample, "row")["dataset"]),
            sample_id,
        )
        if not paths.done.is_file():
            pending.append(sample)
            continue
        if not sample_complete(paths, sample_id, protocol_sha256):
            raise ValueError(f"completed sample {sample_id} failed checksum validation")
        try:
            validate_written_sample(
                paths,
                run_id,
                sample_id,
                protocol_sha256,
                config=config,
                expected_source_row_sha256=str(
                    getattr(sample, "source_row_sha256")
                ),
            )
        except ValueError as exc:
            raise ValueError(f"completed sample {sample_id} failed schema validation") from exc
    return pending


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path
) -> tuple[list[Any], list[dict[str, Any]], str, dict[str, Any]]:
    validate_output_root(output_dir, config)
    sha256 = protocol_hash(config)
    run_dir = model_output_dir(output_dir, config)
    ensure_protocol(run_dir, config, sha256)
    samples, manifest, auxiliary, source_metadata = prepare_samples(config, cache_dir)
    source_metadata = {**source_metadata, "track": str(config["track"])}
    manifests = run_dir / "manifests"
    samples_path = manifests / "samples.jsonl"
    if samples_path.is_file():
        existing = [
            json.loads(line)
            for line in samples_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if existing != manifest:
            raise ValueError("prepared manifest differs from the existing immutable manifest")
    else:
        atomic_write_jsonl(samples_path, manifest)
    manifest_path = manifests / "manifest.json"
    view_files: dict[str, dict[str, Any]] = {}
    for view_name, rows in auxiliary["views"].items():
        view_path = manifests / "views" / f"{view_name}.jsonl"
        if view_path.is_file():
            existing = [
                json.loads(line)
                for line in view_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            if existing != rows:
                raise ValueError(f"prepared {view_name} view differs from its immutable manifest")
        else:
            atomic_write_jsonl(view_path, rows)
        view_files[view_name] = {
            "rows": len(rows),
            "sha256": sha256_file(view_path),
            "path": f"views/{view_name}.jsonl",
        }

    annotation_rows = []
    for record in auxiliary["annotations"]:
        filename = Path(str(record["path"])).name
        destination = manifests / "source_files" / filename
        if destination.is_file():
            if sha256_file(destination) != record["sha256"]:
                raise ValueError(f"copied annotation checksum differs: {filename}")
        else:
            atomic_copy_file(Path(record["source_file"]), destination)
        annotation_rows.append(
            {
                key: value
                for key, value in record.items()
                if key != "source_file"
            }
            | {"copied_path": f"source_files/{filename}"}
        )
    annotations_path = manifests / "annotations.json"
    annotation_value = {
        "schema": "paper_annotation_manifest",
        "track": str(config["track"]),
        "files": annotation_rows,
    }
    if annotations_path.is_file():
        if json.loads(annotations_path.read_text(encoding="utf-8")) != annotation_value:
            raise ValueError("prepared annotation manifest differs from the immutable manifest")
    else:
        atomic_write_json(annotations_path, annotation_value)

    manifest_value = {
        "schema": "saner_paper_manifest",
        "run_id": str(config["run_id"]),
        "track": str(config["track"]),
        "protocol_sha256": sha256,
        "total_samples": len(samples),
        "samples_sha256": sha256_file(samples_path),
        "views": view_files,
        "annotations_sha256": sha256_file(annotations_path),
    }
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest_value:
            raise ValueError("prepared manifest metadata differs from the existing manifest metadata")
    else:
        atomic_write_json(manifest_path, manifest_value)
    source_path = run_dir / "metadata" / "source.json"
    if source_path.is_file():
        if json.loads(source_path.read_text(encoding="utf-8")) != source_metadata:
            raise ValueError("prepared source metadata differs from the immutable source record")
    else:
        atomic_write_json(source_path, source_metadata)
    return samples, manifest, sha256, source_metadata


def prepare(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path
) -> dict[str, Any]:
    samples, _, sha256, source_metadata = _prepare(config, output_dir, cache_dir)
    return {
        "run_id": str(config["run_id"]),
        "track": str(config["track"]),
        "protocol_sha256": sha256,
        "total_samples": len(samples),
        "source": source_metadata,
    }


def _stable_runtime(runtime: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_runtime_environment(runtime)
    return {
        key: value
        for key, value in checked.items()
        if key != "execution_context"
    }


def _environment(bundle: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        **_stable_runtime(runtime),
        "model": dict(getattr(bundle, "provenance", {})),
    }
    if bundle is not None:
        return validate_frozen_environment(value)
    return value


def _write_execution_environment(
    output_dir: Path,
    bundle: Any,
    runtime: Mapping[str, Any],
) -> None:
    frozen = _environment(bundle, runtime)
    context = validate_execution_context(
        runtime.get(
            "execution_context",
            {"slurm": None, "container": None},
        )
    )
    value = {
        "schema": "saner_execution_environment",
        "runtime": {
            key: item for key, item in frozen.items() if key != "model"
        },
        "execution_context": context,
        "model": frozen["model"],
    }
    identity = hashlib.sha256(canonical_bytes(value)).hexdigest()
    path = output_dir / "metadata" / "executions" / f"{identity}.json"
    if path.is_file():
        try:
            if json.loads(path.read_text(encoding="utf-8")) == value:
                return
        except (OSError, json.JSONDecodeError):
            pass
        raise ValueError("execution environment record is invalid")
    atomic_write_json(path, value)


def _record_environment(
    output_dir: Path,
    bundle: Any | None,
    runtime: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    existing_complete_samples: int = 0,
) -> None:
    path = output_dir / "metadata" / "environment.json"
    lock_path = output_dir / "metadata" / "environment_lock.json"
    value = _environment(bundle, runtime)
    model = value.get("model")
    if bundle is not None:
        validate_model_against_config(model, config)
    if existing_complete_samples:
        if not path.is_file():
            raise ValueError("runtime environment metadata is missing for existing samples")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("runtime environment metadata is invalid") from exc
        try:
            existing = validate_frozen_environment(existing)
            validate_runtime_environment(runtime)
        except RuntimeError as exc:
            raise ValueError("recorded runtime environment violates runtime_lock.json") from exc
        recorded_runtime = {
            key: item for key, item in existing.items() if key != "model"
        }
        if recorded_runtime != _stable_runtime(runtime):
            raise ValueError("resume runtime environment differs from environment.json")
        if bundle is not None and existing != value:
            raise ValueError("loaded model/GPU environment differs from environment.json")
        validate_model_against_config(existing["model"], config)
        _environment_sha256(output_dir, config)
        if bundle is not None:
            _write_execution_environment(output_dir, bundle, runtime)
        return

    if path.is_file() and lock_path.is_file():
        try:
            if json.loads(path.read_text(encoding="utf-8")) == value:
                if bundle is not None:
                    _write_execution_environment(output_dir, bundle, runtime)
                _environment_sha256(output_dir, config)
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    atomic_write_json(path, value)
    atomic_write_json(
        lock_path,
        {
            "schema": "saner_runtime_environment_lock",
            "environment_sha256": sha256_file(path),
            "requirements_sha256": runtime["requirements_sha256"],
        },
    )
    if bundle is not None:
        _write_execution_environment(output_dir, bundle, runtime)


def run(
    config: Mapping[str, Any],
    output_dir: Path,
    cache_dir: Path,
    max_samples: int | None = None,
    sample_start: int = 0,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if sample_start < 0:
        raise ValueError("sample_start must be non-negative")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")

    samples, _, sha256, _ = _prepare(config, output_dir, cache_dir)
    runtime = validate_runtime_environment()
    run_dir = model_output_dir(output_dir, config)
    locked_smoke_start = smoke_sample_start(str(config["run_id"]))
    if locked_smoke_start >= len(samples):
        raise RuntimeError("locked smoke sample is outside the prepared manifest")
    smoke_sample = samples[locked_smoke_start]
    all_pending = pending_samples(
        samples,
        output_dir,
        sha256,
        run_id=str(config["run_id"]),
        config=config,
    )
    eligible = {str(sample.sample_id) for sample in samples[sample_start:]}
    candidates = [sample for sample in all_pending if str(sample.sample_id) in eligible]
    if max_samples is not None:
        candidates = candidates[:max_samples]
    complete_count = len(samples) - len(all_pending)
    if complete_count:
        _record_environment(
            run_dir,
            None,
            runtime,
            existing_complete_samples=complete_count,
            config=config,
        )
    smoke_paths = sample_paths(
        output_dir,
        str(smoke_sample.row["model"]),
        str(smoke_sample.row["dataset"]),
        str(smoke_sample.sample_id),
    )
    smoke_passed = _restore_smoke_marker(
        run_dir,
        config,
        sha256,
        locked_smoke_start,
        smoke_sample,
        smoke_paths,
    )
    exact_smoke = (
        sample_start == locked_smoke_start and max_samples == 1
    )
    if not smoke_passed and not exact_smoke:
        raise RuntimeError(
            "smoke gate is incomplete; before any other run execute "
            f"--sample-start {locked_smoke_start} --max-samples 1"
        )
    current = {
        "total_samples": len(samples),
        "complete_samples": complete_count,
        "pending_samples": len(samples) - complete_count,
    }
    _sync_run_markers(run_dir, config, sha256, current)
    if not candidates:
        return {
            "run_id": str(config["run_id"]),
            "track": str(config["track"]),
            "written": 0,
            **current,
        }

    bundle = load_model(config, cache_dir, device=device)
    _record_environment(
        run_dir,
        bundle,
        runtime,
        existing_complete_samples=complete_count,
        config=config,
    )
    written = 0
    last_runtime: dict[str, Any] | None = None
    sample_indices = {
        str(prepared.sample_id): index for index, prepared in enumerate(samples)
    }
    for sample in candidates:
        paths = sample_paths(
            output_dir,
            str(sample.row["model"]),
            str(sample.row["dataset"]),
            str(sample.sample_id),
        )
        paths.done.unlink(missing_ok=True)
        alignment = response_alignment(sample.row["input_ids"], sample.row["uncertainty_labels"])
        feature_set = extract_features(config, sample, bundle, sha256)
        last_runtime = {
            "gpu_total_memory_bytes": int(feature_set.metadata["gpu_total_memory_bytes"]),
            "gpu_peak_allocated_bytes": int(
                feature_set.metadata["gpu_peak_allocated_bytes"]
            ),
            "gpu_peak_reserved_bytes": int(
                feature_set.metadata["gpu_peak_reserved_bytes"]
            ),
            "attention_dtype": str(feature_set.metadata["attention_dtype"]),
        }
        if sample_indices[str(sample.sample_id)] == locked_smoke_start:
            validate_smoke_runtime(str(config["run_id"]), last_runtime)
        paths = write_sample_artifacts(
            output_dir,
            str(config["run_id"]),
            sample,
            feature_set,
            sha256,
            alignment.response_start,
            track=str(config["track"]),
        )
        validate_written_sample(
            paths,
            str(config["run_id"]),
            str(sample.sample_id),
            sha256,
            config=config,
            expected_source_row_sha256=str(sample.source_row_sha256),
        )
        write_done(
            paths,
            str(sample.sample_id),
            sha256,
            str(config["run_id"]),
            str(sample.source_row_sha256),
            track=str(config["track"]),
        )
        if not sample_complete(paths, str(sample.sample_id), sha256):
            raise ValueError(f"sample completion check failed for {sample.sample_id}")
        written += 1
        complete_count += 1
        current = {
            "total_samples": len(samples),
            "complete_samples": complete_count,
            "pending_samples": len(samples) - complete_count,
        }
        _sync_run_markers(
            run_dir,
            config,
            sha256,
            current,
            last_sample_id=str(sample.sample_id),
            runtime_profile=last_runtime,
        )
        if sample_indices[str(sample.sample_id)] == locked_smoke_start:
            write_smoke_marker(
                run_dir / "state",
                run_id=str(config["run_id"]),
                track=str(config["track"]),
                protocol_sha256=sha256,
                sample_start=locked_smoke_start,
                sample_id=str(sample.sample_id),
                done_path=paths.done,
                environment_sha256=_environment_sha256(run_dir, config),
                runtime_profile=last_runtime,
            )

    result = {
        "run_id": str(config["run_id"]),
        "track": str(config["track"]),
        "written": written,
        **current,
    }
    return result


def _sync_run_markers(
    output_dir: Path,
    config: Mapping[str, Any],
    protocol_sha256: str,
    counts: Mapping[str, int],
    last_sample_id: str | None = None,
    runtime_profile: Mapping[str, Any] | None = None,
) -> None:
    completed_path = output_dir / "state" / "COMPLETED.json"
    run_complete = counts["complete_samples"] == counts["total_samples"]
    if not run_complete:
        completed_path.unlink(missing_ok=True)
    environment_sha256 = None
    if counts["complete_samples"]:
        environment_sha256 = _environment_sha256(output_dir, config)

    state_path = output_dir / "state" / "run_state.json"
    previous: Mapping[str, Any] = {}
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                isinstance(loaded, Mapping)
                and loaded.get("schema") == "saner_run_state"
                and loaded.get("run_id") == str(config["run_id"])
                and loaded.get("track") == str(config["track"])
                and loaded.get("protocol_sha256") == protocol_sha256
            ):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = {}
    state = {
        "schema": "saner_run_state",
        "run_id": str(config["run_id"]),
        "track": str(config["track"]),
        "protocol_sha256": protocol_sha256,
        "updated_at": _utc_now(),
        **counts,
    }
    if last_sample_id is not None:
        state["last_sample_id"] = last_sample_id
    elif "last_sample_id" in previous:
        state["last_sample_id"] = previous["last_sample_id"]
    if runtime_profile is not None:
        state["last_sample_runtime"] = dict(runtime_profile)
    elif "last_sample_runtime" in previous:
        state["last_sample_runtime"] = previous["last_sample_runtime"]
    atomic_write_json(state_path, state)

    if not run_complete:
        return
    expected = {
        "schema": "saner_run_completed",
        "run_id": str(config["run_id"]),
        "track": str(config["track"]),
        "protocol_sha256": protocol_sha256,
        "total_samples": counts["total_samples"],
        "manifest_sha256": sha256_file(output_dir / "manifests" / "samples.jsonl"),
        "environment_sha256": environment_sha256,
        "execution_inventory_sha256": _execution_inventory_sha256(output_dir),
    }
    if completed_path.is_file():
        _validate_completion_marker(completed_path, expected)
        return
    atomic_write_json(completed_path, {**expected, "completed_at": _utc_now()})


def status(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path | None = None
) -> dict[str, Any]:
    return validation_status(config, output_dir, cache_dir)


def verify(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path | None = None
) -> dict[str, Any]:
    return validation_verify(config, output_dir, cache_dir)
