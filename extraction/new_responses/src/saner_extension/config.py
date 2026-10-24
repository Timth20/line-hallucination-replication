from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

SCHEMA = "saner_extension_v1"
OUTPUT_ROOT = "outputs_new"
ORIGINAL_DATASETS = ("mbpp", "humaneval", "humaneval_java", "defects4j")
MODEL_IDS = {
    "dsc_1_3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "dsc_6_7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "dsc_33b": "deepseek-ai/deepseek-coder-33b-instruct",
    "codellama_7b": "meta-llama/CodeLlama-7b-Instruct-hf",
    "qwen2_5_coder_7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
}


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def validate_config(config):
    if config.get("schema") != SCHEMA or config.get("track") != "extension":
        raise ValueError("extension schema and track are required")
    key, dataset = config["model_key"], config["dataset"]
    if key not in MODEL_IDS or not (dataset == "bigcodebench" or key == "qwen2_5_coder_7b" and dataset in ORIGINAL_DATASETS):
        raise ValueError("model/dataset combination is not one of the configured settings")
    if config["run_id"] != f"extension_{key}_{dataset}_v1":
        raise ValueError("run_id must identify this exact extension cell")
    if config["model"]["id"] != MODEL_IDS[key]:
        raise ValueError("model id differs from model_key")
    if config["runtime"]["output_root_name"] != OUTPUT_ROOT:
        raise ValueError("the output root must be named " + OUTPUT_ROOT)
    for revision in (config["model"]["revision"], config["source"]["revision"], config["features"]["reference_revision"]):
        if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
            raise ValueError("all source/model revisions must be full commit hashes")
    if not re.fullmatch(r"[0-9a-f]{64}", str(config["source"]["archive_sha256"])):
        raise ValueError("source checksum is required")
    if config["source"]["kind"] != ("bigcodebench_parquet" if dataset == "bigcodebench" else "paper_task_union"):
        raise ValueError("source kind does not match dataset")
    if config["labels"] != {
        "method": "paper_diff_proxy", "reference_selection": "highest_difflib_ratio_then_sha256",
        "reference_role": "separate_control",
    }:
        raise ValueError("the label/reference settings differ")
    if config["split"] != {"seed": 20260907, "train": 0.64, "val": 0.16, "test": 0.20,
                           "grouping": "task_id_humaneval_family_defects4j_project_bcb_exact_prompt",
                           "defects4j_allocation": "hash_order_11_train_3_val_3_test_projects"}:
        raise ValueError("frozen group split contract differs")
    if config["model"]["dtype"] != "float16" or config["model"]["attention_implementation"] != "eager":
        raise ValueError("this protocol requires float16 and eager attention")
    if bool(config["model"].get("quantization", {}).get("load_in_4bit", False)) != (key == "dsc_33b"):
        raise ValueError("only DSC-33B is loaded in 4-bit")
    if config["precision_class"] != ("quantized" if key == "dsc_33b" else "unquantized"):
        raise ValueError("precision class is missing or inconsistent")
    features = config["features"]
    wanted = ["TP", "HS", "AW", "TS", "OR", "LR", "text"]
    if features["feature_families"] != wanted or features["predictor_scope"] != "full_sequence":
        raise ValueError("feature family/scope contract differs")
    shape = config["model"]["expected_shape"]
    for field, length in (("hidden_state_layers", shape["layers"]+1), ("attention_layers", shape["layers"])):
        values = features[field]
        if not values or len(set(values)) != len(values) or any(type(i) is not int or not -length <= i < length for i in values):
            raise ValueError(f"invalid {field}")
    for field in ("layer_top_logits_layers", "token_similarity_layers"):
        if "TS" in wanted:
            values = features[field]
            length = shape["layers"] + 1
            if len(values) < 2 or len(set(values)) != len(values) or any(type(i) is not int or not -length <= i < length for i in values):
                raise ValueError(f"invalid {field}")
    for field in ("probability_top_k", "attention_history", "probability_chunk_tokens", "max_total_tokens"):
        if type(features[field]) is not int or features[field] <= 0:
            raise ValueError(f"invalid {field}")
    gen = config["generation"]
    if gen["do_sample"] is not False or gen["num_beams"] != 1 or gen["max_new_tokens"] <= 0:
        raise ValueError("one greedy output per task is required")
    if gen["max_new_tokens"] >= features["max_total_tokens"]:
        raise ValueError("generation ceiling leaves no prompt room")
    return config


def load_config(path):
    return validate_config(yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader))
