from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifacts import atomic_write_json, sha256_file


_LOCK_PATH = Path(__file__).with_name("smoke_lock.json")


def _lock() -> tuple[dict[str, Any], str]:
    raw = _LOCK_PATH.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("smoke_lock.json is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "saner_longest_sample_smoke"
        or not isinstance(value.get("runs"), dict)
    ):
        raise RuntimeError("smoke_lock.json is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def smoke_sample_start(run_id: str) -> int:
    start = _smoke_spec(run_id)["sample_start"]
    if not isinstance(start, int) or start < 0:
        raise RuntimeError(f"smoke sample is not locked for run {run_id}")
    return start


def _smoke_spec(run_id: str) -> dict[str, Any]:
    value, _ = _lock()
    spec = value["runs"].get(run_id)
    if not isinstance(spec, dict):
        raise RuntimeError(f"smoke criteria are not locked for run {run_id}")
    required = {
        "sample_start",
        "allowed_attention_dtypes",
        "minimum_gpu_memory_bytes",
        "minimum_headroom_bytes",
        "maximum_reserved_fraction",
    }
    if set(spec) != required:
        raise RuntimeError(f"smoke criteria are invalid for run {run_id}")
    return spec


def _runtime_profile(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "gpu_total_memory_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "attention_dtype",
    }
    if set(value) != required:
        raise RuntimeError("smoke runtime profile is invalid")
    try:
        profile = {
            "gpu_total_memory_bytes": int(value["gpu_total_memory_bytes"]),
            "gpu_peak_allocated_bytes": int(value["gpu_peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(value["gpu_peak_reserved_bytes"]),
            "attention_dtype": str(value["attention_dtype"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("smoke runtime profile is invalid") from exc
    if (
        profile["gpu_total_memory_bytes"] <= 0
        or profile["gpu_peak_allocated_bytes"] < 0
        or profile["gpu_peak_reserved_bytes"] <= 0
        or profile["gpu_peak_allocated_bytes"]
        > profile["gpu_peak_reserved_bytes"]
        or profile["gpu_peak_reserved_bytes"] > profile["gpu_total_memory_bytes"]
        or profile["attention_dtype"] not in {"float16", "bfloat16", "float32"}
    ):
        raise RuntimeError("smoke runtime profile is invalid")
    return profile


def validate_smoke_runtime(
    run_id: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    profile = _runtime_profile(value)
    spec = _smoke_spec(run_id)
    total = profile["gpu_total_memory_bytes"]
    reserved = profile["gpu_peak_reserved_bytes"]
    failures = []
    if profile["attention_dtype"] not in spec["allowed_attention_dtypes"]:
        failures.append("attention dtype")
    if total < int(spec["minimum_gpu_memory_bytes"]):
        failures.append("GPU memory")
    if total - reserved < int(spec["minimum_headroom_bytes"]):
        failures.append("memory headroom")
    if reserved / total > float(spec["maximum_reserved_fraction"]):
        failures.append("reserved-memory fraction")
    if failures:
        raise RuntimeError(
            "smoke go/no-go failed: " + ", ".join(failures)
        )
    return profile


def _criteria(run_id: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in _smoke_spec(run_id).items()
        if key != "sample_start"
    }


def write_smoke_marker(
    state_dir: Path,
    *,
    run_id: str,
    track: str,
    protocol_sha256: str,
    sample_start: int,
    sample_id: str,
    done_path: Path,
    environment_sha256: str,
    runtime_profile: Mapping[str, Any],
) -> None:
    _, lock_sha256 = _lock()
    expected = {
        "schema": "saner_smoke_passed",
        "run_id": run_id,
        "track": track,
        "protocol_sha256": protocol_sha256,
        "smoke_lock_sha256": lock_sha256,
        "sample_start": sample_start,
        "sample_id": sample_id,
        "done_sha256": sha256_file(done_path),
        "environment_sha256": environment_sha256,
        "criteria": _criteria(run_id),
        "runtime_profile": validate_smoke_runtime(run_id, runtime_profile),
    }
    path = state_dir / "SMOKE_PASSED.json"
    if path.is_file():
        validate_smoke_marker(
            state_dir,
            run_id=run_id,
            track=track,
            protocol_sha256=protocol_sha256,
            sample_start=sample_start,
            sample_id=sample_id,
            done_path=done_path,
            environment_sha256=environment_sha256,
        )
        return
    atomic_write_json(
        path,
        {**expected, "passed_at": datetime.now(timezone.utc).isoformat()},
    )


def validate_smoke_marker(
    state_dir: Path,
    *,
    run_id: str,
    track: str,
    protocol_sha256: str,
    sample_start: int,
    sample_id: str,
    done_path: Path,
    environment_sha256: str,
) -> dict[str, Any]:
    path = state_dir / "SMOKE_PASSED.json"
    if not path.is_file():
        raise RuntimeError(
            "smoke gate is incomplete; run the locked longest sample with "
            f"--sample-start {sample_start} --max-samples 1"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("smoke gate marker is invalid") from exc
    _, lock_sha256 = _lock()
    expected = {
        "schema": "saner_smoke_passed",
        "run_id": run_id,
        "track": track,
        "protocol_sha256": protocol_sha256,
        "smoke_lock_sha256": lock_sha256,
        "sample_start": sample_start,
        "sample_id": sample_id,
        "done_sha256": sha256_file(done_path),
        "environment_sha256": environment_sha256,
        "criteria": _criteria(run_id),
    }
    if (
        not isinstance(value, dict)
        or set(value) != set(expected) | {"runtime_profile", "passed_at"}
        or any(value.get(key) != expected_value for key, expected_value in expected.items())
        or not isinstance(value.get("passed_at"), str)
    ):
        raise RuntimeError("smoke gate marker is invalid")
    validate_smoke_runtime(run_id, value.get("runtime_profile", {}))
    return value
