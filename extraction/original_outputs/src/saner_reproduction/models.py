from __future__ import annotations

from importlib import metadata
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .types import ModelBundle


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if result.returncode == 0 and versions else None


def validate_released_tokens(
    tokenizer: Any, input_ids: Sequence[int], released_tokens: Sequence[str]
) -> None:
    converted = tokenizer.convert_ids_to_tokens([int(value) for value in input_ids])
    if len(converted) != len(released_tokens):
        raise ValueError("released input_tokens length does not match input_ids")
    for position, (observed, released) in enumerate(zip(converted, released_tokens)):
        if observed != released:
            raise ValueError(
                f"released input_tokens differ from tokenizer at position {position}: "
                f"{released!r} != {observed!r}"
            )


def _validate_shape(model: Any, expected: Mapping[str, Any]) -> dict[str, int]:
    actual = {
        "layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "attention_heads": int(model.config.num_attention_heads),
    }
    for name, observed in actual.items():
        if name in expected and observed != int(expected[name]):
            raise ValueError(
                f"model {name} differs from config: expected {expected[name]}, found {observed}"
            )
    return actual


def load_model(
    config: Mapping[str, Any], cache_dir: Path, device: str = "cuda:0"
) -> ModelBundle:
    import torch
    import transformers

    device = str(device)
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("device must use cuda:N form")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for paper feature extraction")

    model_config = config["model"]
    model_id = str(model_config["id"])
    revision = str(model_config["revision"])
    trust_remote_code = bool(model_config.get("trust_remote_code", False))
    dtype_name = str(model_config["dtype"])
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]
    capability = torch.cuda.get_device_capability(device)
    device_properties = torch.cuda.get_device_properties(device)
    if dtype_name == "bfloat16" and int(capability[0]) < 8:
        raise RuntimeError("BF16 extraction requires an Ampere-or-newer GPU")
    seed = int(config.get("runtime", {}).get("seed", 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    common = {
        "revision": revision,
        "cache_dir": str(Path(cache_dir).expanduser()),
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **common)
    model_kwargs = {
        **common,
        "torch_dtype": dtype,
        "attn_implementation": "eager",
        "device_map": {"": str(device)},
        "low_cpu_mem_usage": True,
    }
    quantization = model_config.get("quantization", {})
    quantization_config = None
    if bool(quantization.get("load_in_4bit", False)):
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["quantization_config"] = quantization_config

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    shape = _validate_shape(model, model_config.get("expected_shape", {}))
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    provenance = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": str(resolved_revision),
        "tokenizer_revision": str(tokenizer.init_kwargs.get("_commit_hash") or revision),
        "dtype": str(model_config["dtype"]),
        "attention_backend": "eager",
        "load_in_4bit": "quantization_config" in model_kwargs,
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(device_properties.name),
        "gpu_total_memory_bytes": int(device_properties.total_memory),
        "gpu_capability": list(capability),
        "cuda_driver_version": _driver_version(),
        **shape,
    }
    if quantization_config is not None:
        provenance["quantization_config"] = quantization_config.to_dict()
        provenance["bitsandbytes_version"] = metadata.version("bitsandbytes")
    return ModelBundle(tokenizer, model, str(device), provenance)
