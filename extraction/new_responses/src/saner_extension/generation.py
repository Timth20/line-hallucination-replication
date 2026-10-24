from __future__ import annotations

import copy
import time

from .config import digest


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def eos_ids(tokenizer, model=None):
    result = set()
    for value in (getattr(tokenizer, "eos_token_id", None),
                  getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if value is not None:
            result.update(value if isinstance(value, (tuple, list)) else [value])
    return sorted(result)


def make_record(task, prompt_ids, output_ids, tokenizer, config, protocol_sha256,
                gpu_profile, elapsed_seconds, stop_ids=None):
    prompt_ids, output_ids = list(prompt_ids), list(output_ids)
    if not prompt_ids or output_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generated prefix does not equal frozen prompt ids")
    response = output_ids[len(prompt_ids):]
    if not response or len(response) > config["generation"]["max_new_tokens"]:
        raise ValueError("empty generation or exceeded generation ceiling")
    if len(output_ids) > config["features"]["max_total_tokens"]:
        raise ValueError("generation exceeds total-token ceiling")
    stops = eos_ids(tokenizer) if stop_ids is None else list(stop_ids)
    if response[-1] in stops:
        reason = "eos"
    elif len(response) == config["generation"]["max_new_tokens"]:
        reason = "length"
    else:
        raise ValueError("generation ended without EOS or the configured length limit")
    return {
        "schema": "extension_generation_v1", "track": "extension",
        "protocol_sha256": protocol_sha256, "run_id": config["run_id"],
        "task_uid": task["task_uid"], "task_id": task["task_id"],
        "dataset": config["dataset"], "group_id": task["group_id"], "split": task["split"],
        "source_task_sha256": task["source_task_sha256"],
        "sample_id": task["task_uid"]+"-model_output", "view": "model_output_only",
        "precision_class": config["precision_class"], "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"], "question": task["question"],
        "scenario": task.get("scenario", "generation"),
        "templated_question": tokenizer.decode(prompt_ids, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),
        "input_ids": output_ids, "input_tokens": tokenizer.convert_ids_to_tokens(output_ids),
        "generated_ids": response, "response_start": len(prompt_ids),
        "response_end_exclusive": len(output_ids), "response_feature_row_start": len(prompt_ids)-1,
        "raw_reply": _decode(tokenizer, response), "reply": _decode(tokenizer, response),
        "finish_reason": reason, "generation_parameters": dict(config["generation"]),
        "eos_token_ids": stops, "generation_seconds": float(elapsed_seconds),
        "generation_gpu_profile": gpu_profile,
    }


def generate(task, prompt_ids, bundle, config, protocol_sha256):
    import torch
    from transformers import GenerationConfig

    if len(prompt_ids)+config["generation"]["max_new_tokens"] > config["features"]["max_total_tokens"]:
        raise ValueError("prompt plus generation budget exceeds total-token ceiling")
    stops = eos_ids(bundle.tokenizer, bundle.model)
    if not stops:
        raise ValueError("tokenizer/model must declare EOS")
    parameters = GenerationConfig(**config["generation"], eos_token_id=stops,
        pad_token_id=getattr(bundle.tokenizer, "pad_token_id", None) or stops[0],
        bos_token_id=getattr(bundle.tokenizer, "bos_token_id", None), use_cache=True,
        return_dict_in_generate=False, output_attentions=False, output_hidden_states=False)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
    cuda = str(bundle.device).startswith("cuda:")
    if cuda:
        torch.cuda.synchronize(bundle.device)
        torch.cuda.reset_peak_memory_stats(bundle.device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = bundle.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       generation_config=parameters)
    profile = {}
    if cuda:
        torch.cuda.synchronize(bundle.device)
        profile = {"gpu_total_memory_bytes": torch.cuda.get_device_properties(bundle.device).total_memory,
                   "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(bundle.device),
                   "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(bundle.device)}
    elapsed = time.perf_counter()-start
    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError("generation must return exactly one sequence")
    return make_record(task, prompt_ids, output[0].tolist(), bundle.tokenizer, config,
                       protocol_sha256, profile, elapsed, stops)


def reference_record(record, reference_text, tokenizer):
    result = copy.deepcopy(record)
    result.update(view="reference_control", sample_id=record["task_uid"]+"-reference_control",
                  finish_reason="teacher_forced_reference", generation_seconds=0.0,
                  generation_gpu_profile={}, parent_generation_sha256=digest(record))
    prompt = record["input_ids"][:record["response_start"]]
    response = tokenizer.encode(reference_text, add_special_tokens=False)
    if not response:
        raise ValueError("empty reference control")
    result.update(input_ids=prompt+response, generated_ids=response,
                  input_tokens=tokenizer.convert_ids_to_tokens(prompt+response),
                  response_end_exclusive=len(prompt)+len(response),
                  raw_reply=_decode(tokenizer, response), reply=_decode(tokenizer, response))
    return result
