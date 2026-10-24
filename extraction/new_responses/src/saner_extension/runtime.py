from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

from .config import digest


def code_sha256():
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
             for p in sorted(root.rglob("*")) if p.suffix in (".py", ".json")}
    return digest(files)


def load_tokenizer(config, cache_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config["model"]["id"], revision=config["model"]["revision"],
        cache_dir=str(Path(cache_dir)), trust_remote_code=False)


def tokenizer_provenance(tokenizer, config):
    template = tokenizer.get_chat_template()
    return {"model_id": config["model"]["id"], "revision": config["model"]["revision"],
            "resolved_revision": tokenizer.init_kwargs.get("_commit_hash") or config["model"]["revision"],
            "chat_template": template, "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
            "vocabulary_sha256": digest(tokenizer.get_vocab()),
            "special_tokens": tokenizer.special_tokens_map,
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "transformers": metadata.version("transformers"), "tokenizers": metadata.version("tokenizers")}


def check_runtime(config, device):
    import torch
    from ._replay.models import _driver_version

    lock = json.loads((Path(__file__).parent/"runtime_lock.json").read_text(encoding="utf-8"))
    observed = {name: metadata.version(name).split("+")[0] for name in lock["packages"]}
    if platform.python_implementation() != "CPython" or platform.python_version_tuple()[:2] != tuple(lock["python"].split(".")):
        raise ValueError("run requires the pinned Python version in environment.yml")
    if observed != lock["packages"] or torch.version.cuda != lock["cuda_runtime"]:
        raise ValueError("run dependency/CUDA mismatch; use the pinned environment.yml")
    if not device.startswith("cuda:") or not device[5:].isdigit() or not torch.cuda.is_available():
        raise ValueError("run requires one CUDA device in cuda:N form")
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < config["runtime"]["minimum_gpu_gib"]*2**30:
        raise ValueError("GPU capacity is below this configuration's allocation gate")
    if torch.cuda.get_device_capability(device)[0] not in (8,9):
        raise ValueError("this CUDA 12.1 environment needs a GPU of compute capability 8.x or 9.x")
    return {"schema": "extension_environment_v1", "track": "extension", "python": platform.python_version(),
            "platform": platform.system(), "machine": platform.machine(), "packages": observed,
            "torch_build": str(torch.__version__), "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
            "gpu_capability": list(torch.cuda.get_device_capability(device)), "driver": _driver_version()}


def gpu_gate(profile, config):
    total, reserved, allocated = (int(profile[key]) for key in
        ("gpu_total_memory_bytes", "gpu_peak_reserved_bytes", "gpu_peak_allocated_bytes"))
    gate = config["runtime"]
    if (profile.get("attention_dtype", "float16") != "float16" or not 0 < allocated <= reserved <= total or
        total < gate["minimum_gpu_gib"]*2**30 or reserved/total > gate["peak_reserved_fraction"] or
        total-reserved < gate["minimum_free_gib"]*2**30):
        raise ValueError("GPU smoke gate failed: inspect capacity, peak reserved and attention dtype")


def compact_profile(metadata):
    return {key: metadata[key] for key in ("gpu_total_memory_bytes", "gpu_peak_allocated_bytes",
                                          "gpu_peak_reserved_bytes", "attention_dtype")}
