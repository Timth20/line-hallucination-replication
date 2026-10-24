from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

from .config import canonical_bytes, digest
from .sources import file_sha256


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name+".", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, payload):
    def write(temporary):
        with open(temporary, "wb") as stream:
            stream.write(canonical_bytes({"sha256": digest(payload), "data": payload})+b"\n")
    _atomic(path, write)


def checked_json(path):
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(envelope) != {"sha256", "data"} or envelope["sha256"] != digest(envelope["data"]):
            raise ValueError("checksum mismatch")
        return envelope["data"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON/checksum envelope: {Path(path).name}") from error


def write_checkpoint(path, payload):
    if Path(path).exists():
        if checked_json(path) != payload:
            raise ValueError(f"refusing to overwrite immutable checkpoint: {Path(path).name}")
        return
    atomic_json(path, payload)


def read_checkpoint(path, protocol_sha256, task_uid):
    record = checked_json(path)
    if (record.get("schema") != "extension_generation_v1" or
        record.get("protocol_sha256") != protocol_sha256 or record.get("task_uid") != task_uid):
        raise ValueError("generation checkpoint protocol/task mismatch")
    return record


@contextmanager
def writer_lock(root):
    """OS releases this advisory lock on interruption; the file itself is harmless."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/".writer.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if stream.read(1) == b"":
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("another writer is using this run") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("another writer is using this run") from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def sample_paths(root, sample_id):
    if not sample_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in sample_id):
        raise ValueError("unsafe sample id")
    return {key: Path(root)/f"{sample_id}.{suffix}" for key, suffix in
            (("output", "output.json"), ("labels", "labels.json"),
             ("features", "features.safetensors"), ("done", "done.json"))}


def write_sample(root, record, labels, features, config):
    from safetensors.torch import save_file
    from .features import validate_tensors

    paths = sample_paths(root, record["sample_id"])
    if paths["done"].exists():
        verify_sample(root, record["sample_id"], config, record["protocol_sha256"])
        return
    validate_tensors(features.tensors, features.metadata, record, config)
    output = {**record, "schema": "extension_output_v1", "claims": labels["claims"]}
    output["generation_record_sha256"] = digest(record)
    labels = {**labels, "schema": "extension_labels_v1", "track": "extension",
              "protocol_sha256": record["protocol_sha256"], "sample_id": record["sample_id"],
              "task_uid": record["task_uid"], "view": record["view"], "split": record["split"]}
    atomic_json(paths["output"], output)
    atomic_json(paths["labels"], labels)
    _atomic(paths["features"], lambda temporary: save_file(dict(features.tensors), temporary,
                                                         metadata=dict(features.metadata)))
    marker = {"schema": "extension_sample_done_v1", "track": "extension",
              "sample_id": record["sample_id"], "protocol_sha256": record["protocol_sha256"],
              "task_uid": record["task_uid"], "view": record["view"],
              "files": {key: {"sha256": file_sha256(paths[key]), "bytes": paths[key].stat().st_size}
                        for key in ("output", "labels", "features")}}
    # Validate the files before writing the completion marker.
    _validate_sample_files(paths, config, marker)
    write_checkpoint(paths["done"], marker)


def _validate_sample_files(paths, config, marker):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from .features import validate_tensors
    from .labels import validate_labels

    if set(marker["files"]) != {"output", "labels", "features"}:
        raise ValueError("sample file inventory mismatch")
    for key, expected in marker["files"].items():
        if not paths[key].is_file() or paths[key].stat().st_size != expected["bytes"] or file_sha256(paths[key]) != expected["sha256"]:
            raise ValueError(f"sample checksum mismatch: {paths[key].name}")
    record, labels = checked_json(paths["output"]), checked_json(paths["labels"])
    for item in (record, labels):
        if any(item.get(key) != marker[key] for key in ("track", "sample_id", "protocol_sha256", "task_uid", "view")):
            raise ValueError("sample identity mismatch")
    if record.get("schema") != "extension_output_v1" or labels.get("schema") != "extension_labels_v1":
        raise ValueError("unexpected sample schema")
    validate_labels(labels, record)
    with safe_open(paths["features"], framework="pt", device="cpu") as archive:
        metadata = archive.metadata()
    validate_tensors(load_file(paths["features"], device="cpu"), metadata, record, config)
    return record, labels, metadata


def verify_sample(root, sample_id, config, protocol_sha256):
    paths = sample_paths(root, sample_id)
    if not paths["done"].exists():
        return None
    marker = checked_json(paths["done"])
    if (marker.get("schema") != "extension_sample_done_v1" or marker.get("track") != "extension" or
        marker.get("sample_id") != sample_id or marker.get("protocol_sha256") != protocol_sha256):
        raise ValueError("completion marker identity/protocol mismatch")
    return _validate_sample_files(paths, config, marker)
