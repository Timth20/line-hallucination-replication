from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping


_LOCK_PATH = Path(__file__).with_name("runtime_lock.json")

_RUNTIME_SCHEMA = "saner_runtime_environment"
_RUNTIME_BASE_FIELDS = {
    "schema",
    "python",
    "python_implementation",
    "cuda_runtime",
    "packages",
    "platform",
}
_STABLE_RUNTIME_FIELDS = _RUNTIME_BASE_FIELDS | {"requirements_sha256"}
_MODEL_FIELDS = {
    "model_id",
    "requested_revision",
    "resolved_revision",
    "tokenizer_revision",
    "dtype",
    "attention_backend",
    "load_in_4bit",
    "torch_version",
    "transformers_version",
    "cuda_version",
    "gpu_name",
    "gpu_total_memory_bytes",
    "gpu_capability",
    "cuda_driver_version",
    "layers",
    "hidden_size",
    "attention_heads",
}
_QUANTIZED_MODEL_FIELDS = {"quantization_config", "bitsandbytes_version"}
_SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-f]{64})\Z", re.IGNORECASE)

_SLURM_ALIASES = {
    "SANER_SLURM_JOB_ALIAS": "job_alias",
    "SANER_SLURM_CLUSTER_ALIAS": "cluster_alias",
    "SANER_SLURM_PARTITION_ALIAS": "partition_alias",
}

_SLURM_COUNTS = {
    "SLURM_CPUS_PER_TASK": "cpus_per_task",
    "SLURM_NTASKS": "ntasks",
    "SLURM_JOB_NUM_NODES": "nodes",
    "SLURM_ARRAY_TASK_COUNT": "array_task_count",
}


def _logical_value(value: object, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or len(text) > 256
        or any(character in text for character in ("/", "\\", "\0", "\r", "\n"))
    ):
        raise RuntimeError(f"{label} is not a safe logical value")
    return text


def _container_revision(value: object) -> str:
    match = _SHA256_RE.fullmatch(str(value).strip())
    if match is None:
        raise RuntimeError("container revision must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value or "\n" in value:
        raise RuntimeError(f"{label} is invalid")
    return value


def _positive_count(value: object, label: str) -> int:
    try:
        count = int(str(value).strip())
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a valid resource count") from exc
    if count <= 0:
        raise RuntimeError(f"{label} is not a valid resource count")
    return count


def _gpu_count(value: object) -> int:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise RuntimeError("Slurm GPU allocation is invalid")
    count = 0
    for token in tokens:
        endpoints = token.split("-", 1)
        if len(endpoints) == 2 and all(endpoint.isdigit() for endpoint in endpoints):
            start, end = map(int, endpoints)
            if end < start:
                raise RuntimeError("Slurm GPU allocation is invalid")
            count += end - start + 1
        else:
            count += 1
    return count


def validate_execution_context(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"slurm", "container"}:
        raise RuntimeError("execution context schema is invalid")
    slurm = value["slurm"]
    if slurm is not None:
        allowed = {
            "scheduler",
            *_SLURM_ALIASES.values(),
            *_SLURM_COUNTS.values(),
            "gpu_count",
        }
        if (
            not isinstance(slurm, Mapping)
            or not set(slurm).issubset(allowed)
            or slurm.get("scheduler") != "slurm"
        ):
            raise RuntimeError("Slurm execution context is invalid")
        normalized_slurm: dict[str, Any] = {"scheduler": "slurm"}
        for key in _SLURM_ALIASES.values():
            if key in slurm:
                normalized_slurm[key] = _logical_value(slurm[key], f"Slurm {key}")
        for key in {*_SLURM_COUNTS.values(), "gpu_count"}:
            if key in slurm:
                normalized_slurm[key] = _positive_count(slurm[key], f"Slurm {key}")
        slurm = normalized_slurm
    container = value["container"]
    if container is not None:
        if not isinstance(container, Mapping) or set(container) != {
            "engine",
            "revision",
        }:
            raise RuntimeError("container execution context is invalid")
        container = {
            "engine": _logical_value(container["engine"], "container engine"),
            "revision": _container_revision(container["revision"]),
        }
    return {"slurm": slurm, "container": container}


def execution_context(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    values = os.environ if environ is None else environ
    slurm_detected = any(
        name.startswith("SLURM_") and item for name, item in values.items()
    )
    slurm: dict[str, Any] = {"scheduler": "slurm"} if slurm_detected else {}
    for source_name, output_name in _SLURM_ALIASES.items():
        if values.get(source_name):
            slurm[output_name] = _logical_value(
                values[source_name], f"Slurm {output_name}"
            )
    for source_name, output_name in _SLURM_COUNTS.items():
        if values.get(source_name):
            slurm[output_name] = _positive_count(
                values[source_name], f"Slurm {output_name}"
            )
    if values.get("SLURM_GPUS_ON_NODE"):
        slurm["gpu_count"] = _positive_count(
            values["SLURM_GPUS_ON_NODE"], "Slurm gpu_count"
        )
    elif values.get("SLURM_JOB_GPUS"):
        slurm["gpu_count"] = _gpu_count(values["SLURM_JOB_GPUS"])

    engine = None
    for source_name, candidate in (
        ("APPTAINER_CONTAINER", "apptainer"),
        ("SINGULARITY_CONTAINER", "singularity"),
    ):
        if values.get(source_name):
            engine = candidate
            break
    revision = values.get("SANER_CONTAINER_REVISION")
    container = None
    if engine is not None or revision:
        if not revision:
            raise RuntimeError(
                "container revision is required via SANER_CONTAINER_REVISION"
            )
        container = {
            "engine": engine or "unspecified",
            "revision": _container_revision(revision),
        }
    return validate_execution_context(
        {"slurm": slurm or None, "container": container}
    )


def _requirements() -> tuple[dict[str, Any], str]:
    raw = _LOCK_PATH.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("runtime_lock.json is invalid") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "saner_runtime_requirements"
        or not isinstance(value.get("python"), dict)
        or not isinstance(value.get("packages"), dict)
    ):
        raise RuntimeError("runtime_lock.json is invalid")
    return value, hashlib.sha256(raw).hexdigest()


def collect_runtime_environment() -> dict[str, Any]:
    import torch

    requirements, _ = _requirements()
    packages: dict[str, str | None] = {}
    for name in requirements["packages"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema": "saner_runtime_environment",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "packages": packages,
        "platform": platform.platform(),
        "execution_context": execution_context(),
    }


def _package_matches(name: str, observed: object, expected: str) -> bool:
    if not isinstance(observed, str):
        return False
    if name == "torch":
        return observed.split("+", 1)[0] == expected
    return observed == expected


def validate_runtime_environment(
    observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    requirements, requirements_sha256 = _requirements()
    value = dict(collect_runtime_environment() if observed is None else observed)
    errors: list[str] = []

    allowed_fields = _RUNTIME_BASE_FIELDS | {
        "execution_context",
        "requirements_sha256",
    }
    if not _RUNTIME_BASE_FIELDS.issubset(value) or not set(value).issubset(
        allowed_fields
    ):
        raise RuntimeError("runtime environment schema is invalid")
    if value.get("schema") != _RUNTIME_SCHEMA:
        raise RuntimeError("runtime environment schema is invalid")
    if not isinstance(value.get("platform"), str) or not value["platform"]:
        errors.append("platform metadata is missing")
    if "execution_context" in value:
        try:
            value["execution_context"] = validate_execution_context(
                value["execution_context"]
            )
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise RuntimeError("runtime environment schema is invalid") from exc

    recorded_lock = value.get("requirements_sha256")
    if recorded_lock is not None and recorded_lock != requirements_sha256:
        errors.append("runtime requirements checksum differs from runtime_lock.json")

    python_requirement = requirements["python"]
    python_version = str(value.get("python", ""))
    observed_major_minor = ".".join(python_version.split(".")[:2])
    if observed_major_minor != str(python_requirement["major_minor"]):
        errors.append(
            f"Python must be {python_requirement['major_minor']}.x; found {python_version or 'missing'}"
        )
    implementation = value.get("python_implementation")
    if implementation != python_requirement["implementation"]:
        errors.append(
            "Python implementation must be "
            f"{python_requirement['implementation']}; found {implementation or 'missing'}"
        )

    cuda_runtime = value.get("cuda_runtime")
    if cuda_runtime != requirements["cuda_runtime"]:
        errors.append(
            f"CUDA runtime must be {requirements['cuda_runtime']}; "
            f"found {cuda_runtime or 'missing'}"
        )

    packages = value.get("packages")
    if not isinstance(packages, Mapping):
        errors.append("packages metadata is missing")
        packages = {}
    elif set(packages) != set(requirements["packages"]):
        errors.append("package inventory differs from runtime_lock.json")
    for name, expected in requirements["packages"].items():
        installed = packages.get(name)
        if not _package_matches(str(name), installed, str(expected)):
            errors.append(f"{name} must be {expected}; found {installed or 'missing'}")

    if errors:
        raise RuntimeError("runtime environment is incompatible: " + "; ".join(errors))
    return {
        **value,
        "schema": _RUNTIME_SCHEMA,
        "requirements_sha256": requirements_sha256,
    }


def validate_model_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("model provenance schema is invalid")
    quantized = value.get("load_in_4bit")
    expected_fields = _MODEL_FIELDS | (
        _QUANTIZED_MODEL_FIELDS if quantized is True else set()
    )
    if set(value) != expected_fields or not isinstance(quantized, bool):
        raise RuntimeError("model provenance schema is invalid")

    normalized = dict(value)
    for name in (
        "model_id",
        "requested_revision",
        "resolved_revision",
        "tokenizer_revision",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "gpu_name",
    ):
        normalized[name] = _nonempty_string(value[name], f"model provenance {name}")
    if value["dtype"] not in {"float16", "bfloat16"}:
        raise RuntimeError("model provenance dtype is invalid")
    if value["attention_backend"] != "eager":
        raise RuntimeError("model provenance attention backend is invalid")
    driver = value["cuda_driver_version"]
    if driver is not None:
        normalized["cuda_driver_version"] = _nonempty_string(
            driver, "model provenance cuda_driver_version"
        )
    for name in (
        "gpu_total_memory_bytes",
        "layers",
        "hidden_size",
        "attention_heads",
    ):
        if not isinstance(value[name], int) or isinstance(value[name], bool) or value[name] <= 0:
            raise RuntimeError(f"model provenance {name} is invalid")
    capability = value["gpu_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in capability
        )
    ):
        raise RuntimeError("model provenance gpu_capability is invalid")
    if quantized:
        if not isinstance(value["quantization_config"], Mapping):
            raise RuntimeError("model provenance quantization_config is invalid")
        normalized["quantization_config"] = dict(value["quantization_config"])
        normalized["bitsandbytes_version"] = _nonempty_string(
            value["bitsandbytes_version"],
            "model provenance bitsandbytes_version",
        )
    return normalized


def validate_frozen_environment(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STABLE_RUNTIME_FIELDS | {
        "model"
    }:
        raise RuntimeError("frozen runtime environment schema is invalid")
    runtime_value = {key: value[key] for key in _STABLE_RUNTIME_FIELDS}
    checked_runtime = validate_runtime_environment(runtime_value)
    if set(checked_runtime) != _STABLE_RUNTIME_FIELDS:
        raise RuntimeError("frozen runtime environment schema is invalid")
    model = validate_model_provenance(value["model"])
    packages = checked_runtime["packages"]
    if (
        model["cuda_version"] != checked_runtime["cuda_runtime"]
        or model["torch_version"].split("+", 1)[0]
        != str(packages["torch"]).split("+", 1)[0]
        or model["transformers_version"] != packages["transformers"]
        or (
            model["load_in_4bit"]
            and model["bitsandbytes_version"] != packages["bitsandbytes"]
        )
    ):
        raise RuntimeError("model provenance differs from runtime package inventory")
    return {**checked_runtime, "model": model}


def validate_model_against_config(
    value: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    model = validate_model_provenance(value)
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        raise RuntimeError("experiment model config is invalid")
    revision = str(model_config.get("revision", ""))
    quantization = model_config.get("quantization", {})
    if not isinstance(quantization, Mapping):
        raise RuntimeError("experiment model config is invalid")
    expected_shape = model_config.get("expected_shape")
    if not isinstance(expected_shape, Mapping):
        raise RuntimeError("experiment model config is invalid")
    expected = {
        "model_id": str(model_config.get("id", "")),
        "requested_revision": revision,
        "resolved_revision": revision,
        "tokenizer_revision": revision,
        "dtype": str(model_config.get("dtype", "")),
        "attention_backend": str(
            model_config.get("attention_implementation", "")
        ),
        "load_in_4bit": bool(quantization.get("load_in_4bit", False)),
        "layers": int(expected_shape.get("layers", -1)),
        "hidden_size": int(expected_shape.get("hidden_size", -1)),
        "attention_heads": int(expected_shape.get("attention_heads", -1)),
    }
    mismatches = [
        name for name, expected_value in expected.items()
        if model.get(name) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            "model provenance differs from experiment config: "
            + ", ".join(sorted(mismatches))
        )
    return model
