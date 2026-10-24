"""LoRA modules on every decoder layer of the Student, and the Student's hidden states on a token sequence."""
import math

import numpy as np
import torch
import torch.nn as nn

LORA = {"rank": 16, "alpha": 32, "dropout": 0.05}
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        delta = (self.dropout(x).float() @ self.lora_A.t()) @ self.lora_B.t()
        return out + (delta * self.scaling).to(out.dtype)


def inject_lora(model) -> list:
    params = []
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGET_MODULES:
                if hasattr(parent, name) and isinstance(getattr(parent, name), nn.Linear):
                    wrapped = LoRALinear(getattr(parent, name), LORA["rank"], LORA["alpha"], LORA["dropout"])
                    setattr(parent, name, wrapped)
                    params += [wrapped.lora_A, wrapped.lora_B]
    return params


def lora_state(model) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def forward_hidden(model, ids: np.ndarray):
    inp = torch.as_tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)
    out = model.model(input_ids=inp, output_hidden_states=True, use_cache=False)
    return out.hidden_states


def load_student(repo, revision, adapter=None, lora=False, cache_dir=None):
    """The frozen Student; LoRA modules are added with lora and loaded from adapter if given."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    path = snapshot_download(repo, revision=revision, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to("cuda")
    params = inject_lora(model) if (lora or adapter is not None) else []
    if adapter is not None:
        from safetensors.torch import load_file
        result = model.load_state_dict(load_file(str(adapter)), strict=False)
        if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
            raise ValueError("LoRA weights do not match the model")
    return model, params
