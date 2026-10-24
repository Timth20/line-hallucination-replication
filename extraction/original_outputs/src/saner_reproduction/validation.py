from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .artifacts import (
    SamplePaths,
    ensure_protocol,
    model_output_dir,
    sample_paths,
    sha256_file,
)
from .config import canonical_bytes, protocol_hash, validate_output_root
from .runtime import (
    validate_execution_context,
    validate_frozen_environment,
    validate_model_against_config,
)
from .smoke import smoke_sample_start, validate_smoke_marker


_BASE_TENSOR_KEYS = {
    "input_ids",
    "next_token_ids",
    "response_token_ids",
    "response_token_positions",
    "predictor_positions",
    "topk_probabilities",
    "topk_token_ids",
    "hidden_states",
    "attention_window",
}

_EXTENDED_TENSOR_KEYS = {
    "layer_top_logits",
    "layer_top_token_ids",
    "token_similarities",
    "output_ranks",
    "lookback_lens",
    "attention_prompt_mass",
    "attention_generated_mass",
}

_EXPECTED_DTYPES = {
    "input_ids": {"I32"},
    "next_token_ids": {"I32"},
    "response_token_ids": {"I32"},
    "response_token_positions": {"I32"},
    "predictor_positions": {"I32"},
    "topk_probabilities": {"BF16", "F16", "F32"},
    "topk_token_ids": {"I32"},
    "hidden_states": {"BF16", "F16"},
    "attention_window": {"BF16", "F16", "F32"},
    "layer_top_logits": {"F32"},
    "layer_top_token_ids": {"I32"},
    "token_similarities": {"F32"},
    "output_ranks": {"I32"},
    "lookback_lens": {"F32"},
    "attention_prompt_mass": {"F32"},
    "attention_generated_mass": {"F32"},
}

_BASE_METADATA_KEYS = {
    "schema",
    "run_id",
    "track",
    "sample_id",
    "protocol_sha256",
    "feature_families",
    "hidden_state_layers",
    "attention_layers",
    "attention_history",
    "probability_top_k",
    "predictor_scope",
    "prediction_alignment",
    "attention_axis_order",
    "model_id",
    "model_revision",
    "resolved_model_revision",
    "source_row_sha256",
    "source_revision",
    "source_archive_sha256",
    "reference_repository",
    "reference_revision",
    "decoder_layers",
    "hidden_size",
    "attention_heads",
    "max_total_tokens",
    "probability_chunk_tokens",
    "response_start",
    "response_tokens",
    "predictor_rows",
    "response_feature_row_start",
    "hidden_state_tuple_indices",
    "attention_tuple_indices",
    "vocabulary_size",
    "gpu_total_memory_bytes",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "attention_dtype",
    "gpu_peak_scope",
    "tensor_dtypes",
}

_EXTENDED_METADATA_KEYS = {
    "layer_top_logits_layers",
    "layer_top_logits_tuple_indices",
    "token_similarity_layers",
    "token_similarity_tuple_indices",
    "output_rank_tuple_indices",
    "output_rank_layers",
    "output_rank_index_semantics",
    "lookback_layers",
    "lookback_index_semantics",
    "intermediate_top_k",
    "intermediate_chunk_tokens",
    "intermediate_projection",
    "configuration_repository",
    "configuration_revision",
    "configuration_path",
    "uhead_configuration_path",
    "lookback_configuration_path",
    "output_rank_behavior",
    "output_rank_alignment",
    "lookback_behavior",
    "layer_top_axis_order",
    "token_similarity_axis_order",
    "output_rank_axis_order",
    "lookback_axis_order",
    "attention_mass_axis_order",
}

_SOURCE_OUTPUT_FIELDS = (
    "idx",
    "model",
    "dataset",
    "task_id",
    "meta",
    "hallucinated",
    "templated_question",
    "question",
    "answer",
    "reply",
    "claims",
    "input_tokens",
    "input_ids",
)
_SOURCE_LABEL_FIELDS = ("uncertainty_labels", "verified", "diff")

_OUTPUT_FIELDS = set(_SOURCE_OUTPUT_FIELDS) | {
    "schema",
    "run_id",
    "track",
    "sample_id",
    "protocol_sha256",
    "view",
    "response_start",
    "source_row_sha256",
    "occurrences",
}

_LABEL_FIELDS = set(_SOURCE_LABEL_FIELDS) | {
    "schema",
    "run_id",
    "track",
    "sample_id",
    "protocol_sha256",
    "source_row_sha256",
    "hallucinated",
    "claims",
}


def _read_json_mapping(path: Path, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} is invalid")
    return value


def _require_document_identity(
    value: Mapping[str, Any],
    description: str,
    schema: str,
    run_id: str,
    sample_id: str,
) -> None:
    if value.get("schema") != schema:
        raise ValueError(f"{description} schema is invalid")
    if value.get("run_id") != run_id:
        raise ValueError(f"{description} run_id differs from the requested run")
    if value.get("sample_id") != sample_id:
        raise ValueError(f"{description} sample_id differs from the requested sample")


def _metadata_int(metadata: Mapping[str, str], name: str) -> int:
    try:
        return int(metadata[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"safetensors metadata {name} is invalid") from exc


def _metadata_list(metadata: Mapping[str, str], name: str) -> list[Any]:
    try:
        value = json.loads(metadata[name])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors metadata {name} is invalid") from exc
    if not isinstance(value, list):
        raise ValueError(f"safetensors metadata {name} is invalid")
    return value


def _integer_list(value: Any, description: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{description} is invalid")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} is invalid") from exc


def _resolved_layers(layers: list[int], tuple_length: int) -> list[int]:
    resolved = [tuple_length + layer if layer < 0 else layer for layer in layers]
    if any(index < 0 or index >= tuple_length for index in resolved):
        raise ValueError("configured feature layer is unavailable")
    return resolved


def _validate_config_contract(
    metadata: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...]],
    config: Mapping[str, Any],
    input_length: int,
) -> None:
    model = config["model"]
    source = config["source"]
    features = config["features"]
    architecture = model["expected_shape"]
    decoder_layers = int(architecture["layers"])
    hidden_size = int(architecture["hidden_size"])
    attention_heads = int(architecture["attention_heads"])

    if config.get("run_id") != metadata.get("run_id"):
        raise ValueError("safetensors run_id differs from config")
    if config.get("track") != metadata.get("track"):
        raise ValueError("safetensors track differs from config")
    if input_length > int(features["max_total_tokens"]):
        raise ValueError("sample exceeds the configured token ceiling")
    for name, expected in (
        ("model_id", model["id"]),
        ("model_revision", model["revision"]),
        ("resolved_model_revision", model["revision"]),
        ("source_revision", source["revision"]),
        ("source_archive_sha256", source["archive_sha256"]),
        ("reference_repository", features["reference_repository"]),
        ("reference_revision", features["reference_revision"]),
        ("gpu_peak_scope", "teacher_forcing_through_compact_reductions"),
    ):
        if metadata.get(name) != str(expected):
            raise ValueError(f"safetensors metadata {name} differs from config")
    for name, expected in (
        ("decoder_layers", decoder_layers),
        ("hidden_size", hidden_size),
        ("attention_heads", attention_heads),
        ("max_total_tokens", features["max_total_tokens"]),
        ("probability_chunk_tokens", features.get("probability_chunk_tokens", 128)),
        ("probability_top_k", features["probability_top_k"]),
        ("attention_history", features["attention_history"]),
    ):
        if _metadata_int(metadata, name) != int(expected):
            raise ValueError(f"safetensors metadata {name} differs from config")

    families = _metadata_list(metadata, "feature_families")
    if families != list(features["feature_families"]):
        raise ValueError("safetensors feature_families differs from config")
    hidden_layers = [int(value) for value in features["hidden_state_layers"]]
    attention_layers = [int(value) for value in features["attention_layers"]]
    if _metadata_list(metadata, "hidden_state_layers") != hidden_layers:
        raise ValueError("safetensors hidden-state layers differ from config")
    if _metadata_list(metadata, "hidden_state_tuple_indices") != _resolved_layers(
        hidden_layers, decoder_layers + 1
    ):
        raise ValueError("safetensors hidden-state tuple indices differ from config")
    if _metadata_list(metadata, "attention_layers") != attention_layers:
        raise ValueError("safetensors attention layers differ from config")
    if _metadata_list(metadata, "attention_tuple_indices") != _resolved_layers(
        attention_layers, decoder_layers
    ):
        raise ValueError("safetensors attention tuple indices differ from config")
    if shapes["hidden_states"][-1] != hidden_size:
        raise ValueError("safetensors hidden size differs from config")
    if shapes["attention_window"][2] != attention_heads:
        raise ValueError("safetensors attention head count differs from config")

    extended = {"TS", "OR", "LR"}.issubset(features["feature_families"])
    if not extended:
        return
    top_layers = [int(value) for value in features["layer_top_logits_layers"]]
    similarity_layers = [int(value) for value in features["token_similarity_layers"]]
    for name, expected in (
        ("layer_top_logits_layers", top_layers),
        (
            "layer_top_logits_tuple_indices",
            _resolved_layers(top_layers, decoder_layers + 1),
        ),
        ("token_similarity_layers", similarity_layers),
        (
            "token_similarity_tuple_indices",
            _resolved_layers(similarity_layers, decoder_layers + 1),
        ),
        ("output_rank_tuple_indices", list(range(decoder_layers))),
        ("output_rank_layers", list(range(decoder_layers))),
        ("lookback_layers", list(range(decoder_layers))),
    ):
        if _metadata_list(metadata, name) != expected:
            raise ValueError(f"safetensors metadata {name} differs from config")
    for name in ("intermediate_top_k", "intermediate_chunk_tokens"):
        if _metadata_int(metadata, name) != int(features[name]):
            raise ValueError(f"safetensors metadata {name} differs from config")
    for name in (
        "intermediate_projection",
        "configuration_repository",
        "configuration_revision",
        "configuration_path",
        "uhead_configuration_path",
        "lookback_configuration_path",
        "output_rank_behavior",
        "output_rank_alignment",
        "lookback_behavior",
    ):
        if metadata.get(name) != str(features[name]):
            raise ValueError(f"safetensors metadata {name} differs from config")
    if (
        metadata.get("output_rank_index_semantics")
        != "public_hidden_tuple_indices_includes_embedding_omits_final_hidden"
        or metadata.get("lookback_index_semantics")
        != "attention_tuple_decoder_layer_indices"
    ):
        raise ValueError("safetensors public layer index semantics are invalid")
    for name in (
        "output_ranks",
        "lookback_lens",
        "attention_prompt_mass",
        "attention_generated_mass",
    ):
        expected_tail = (
            (decoder_layers,)
            if name == "output_ranks"
            else (decoder_layers, attention_heads)
        )
        if shapes[name][1:] != expected_tail:
            raise ValueError(f"safetensors {name} architecture differs from config")


@contextmanager
def _open_safetensors(path: Path):
    try:
        import safetensors
    except ImportError as exc:
        raise RuntimeError("safetensors is required to validate sample artifacts") from exc
    safetensor_error = getattr(safetensors, "SafetensorError", ())
    try:
        with safetensors.safe_open(str(path), framework="pt", device="cpu") as handle:
            yield handle
    except safetensor_error as exc:
        raise ValueError("safetensors file is invalid") from exc


def validate_written_sample(
    paths: SamplePaths,
    run_id: str,
    sample_id: str,
    protocol_sha256: str,
    *,
    config: Mapping[str, Any] | None = None,
    expected_source_row_sha256: str | None = None,
    manifest_row: Mapping[str, Any] | None = None,
) -> None:
    output = _read_json_mapping(paths.output, "output")
    labels = _read_json_mapping(paths.labels, "labels")
    _require_document_identity(
        output, "output", "saner_paper_output", run_id, sample_id
    )
    _require_document_identity(
        labels, "labels", "saner_paper_labels", run_id, sample_id
    )
    unexpected_output_fields = set(output) - _OUTPUT_FIELDS
    unexpected_label_fields = set(labels) - _LABEL_FIELDS
    if unexpected_output_fields:
        raise ValueError(
            f"output fields are invalid: unexpected={sorted(unexpected_output_fields)}"
        )
    if unexpected_label_fields:
        raise ValueError(
            f"labels fields are invalid: unexpected={sorted(unexpected_label_fields)}"
        )
    track = output.get("track")
    if track not in {"original"}:
        raise ValueError("output track is invalid")
    if labels.get("track") != track:
        raise ValueError("output and labels track values differ")
    source_row_sha256 = output.get("source_row_sha256")
    if (
        not isinstance(source_row_sha256, str)
        or len(source_row_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_row_sha256)
    ):
        raise ValueError("output source_row_sha256 is invalid")
    for document, description in ((output, "output"), (labels, "labels")):
        if document.get("protocol_sha256") != protocol_sha256:
            raise ValueError(f"{description} protocol_sha256 differs from the requested sample")
        if document.get("source_row_sha256") != source_row_sha256:
            raise ValueError(f"{description} source_row_sha256 differs from output")
    output_claims = output.get("claims")
    label_claims = labels.get("claims")
    if (
        not isinstance(output_claims, list)
        or not isinstance(label_claims, list)
        or output_claims != label_claims
    ):
        raise ValueError("output and labels claims are invalid or differ")
    reconstructed_source_row = {
        name: output[name]
        for name in _SOURCE_OUTPUT_FIELDS
        if name in output
    }
    reconstructed_source_row.update(
        {name: labels[name] for name in _SOURCE_LABEL_FIELDS if name in labels}
    )
    reconstructed_source_sha256 = hashlib.sha256(
        canonical_bytes(reconstructed_source_row)
    ).hexdigest()
    if reconstructed_source_sha256 != source_row_sha256:
        raise ValueError("artifact contents differ from output source_row_sha256")
    if (
        expected_source_row_sha256 is not None
        and source_row_sha256 != expected_source_row_sha256
    ):
        raise ValueError("output source_row_sha256 differs from the expected source row")

    input_ids = _integer_list(output.get("input_ids"), "output input_ids")
    input_tokens = output.get("input_tokens")
    if (
        not isinstance(input_tokens, list)
        or len(input_tokens) != len(input_ids)
        or any(not isinstance(token, str) for token in input_tokens)
    ):
        raise ValueError("output input_tokens are invalid")
    for name in ("templated_question", "question", "reply"):
        if not isinstance(output.get(name), str):
            raise ValueError(f"output {name} is invalid")
    if not isinstance(labels.get("verified"), list):
        raise ValueError("labels verified is invalid")
    try:
        output_hallucinated = int(output["hallucinated"])
        labels_hallucinated = int(labels["hallucinated"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("hallucinated label is invalid") from exc
    if output_hallucinated not in (0, 1) or labels_hallucinated != output_hallucinated:
        raise ValueError("output and labels hallucinated values differ")
    uncertainty_labels = _integer_list(
        labels.get("uncertainty_labels"), "labels uncertainty_labels"
    )
    if len(input_ids) != len(uncertainty_labels) or len(input_ids) < 2:
        raise ValueError("output input_ids and labels uncertainty_labels are misaligned")
    response_start = output.get("response_start")
    if not isinstance(response_start, int) or not 0 < response_start < len(input_ids):
        raise ValueError("output response_start is invalid")
    if any(label != -100 for label in uncertainty_labels[:response_start]) or any(
        label == -100 for label in uncertainty_labels[response_start:]
    ):
        raise ValueError("labels uncertainty_labels do not define the configured response suffix")
    if config is not None:
        if track != config["track"]:
            raise ValueError("output track differs from config")
        if output.get("model") != config["source"]["source_model"]:
            raise ValueError("output model differs from config")
        if output.get("dataset") not in config["source"]["datasets"]:
            raise ValueError("output dataset differs from config")
    if manifest_row is not None:
        for name in (
            "sample_id",
            "track",
            "dataset",
            "task_id",
            "view",
            "response_start",
            "source_row_sha256",
            "occurrences",
        ):
            if output.get(name) != manifest_row.get(name):
                raise ValueError(f"output {name} differs from manifest")
        if len(input_ids) != manifest_row.get("total_tokens"):
            raise ValueError("output token count differs from manifest")

    if not paths.features.is_file():
        raise ValueError("features.safetensors is missing")
    with _open_safetensors(paths.features) as handle:
        metadata = handle.metadata()
        if not isinstance(metadata, Mapping):
            raise ValueError("safetensors metadata is invalid")
        if metadata.get("schema") != "saner_paper_features":
            raise ValueError("safetensors metadata schema is invalid")
        for name, expected in (
            ("run_id", run_id),
            ("track", str(track)),
            ("sample_id", sample_id),
            ("protocol_sha256", protocol_sha256),
        ):
            if metadata.get(name) != expected:
                raise ValueError(f"safetensors metadata {name} differs from the requested sample")
        if metadata.get("source_row_sha256") != source_row_sha256:
            raise ValueError("safetensors source_row_sha256 differs from output")
        if (
            metadata.get("prediction_alignment")
            != "row p predicts input_ids[p+1]"
        ):
            raise ValueError("safetensors metadata prediction_alignment is invalid")
        if metadata.get("predictor_scope") != "full_sequence":
            raise ValueError("safetensors metadata predictor_scope is invalid")
        if metadata.get("attention_axis_order") != "predictor,history,head,layer":
            raise ValueError("safetensors metadata attention_axis_order is invalid")
        families = _metadata_list(metadata, "feature_families")
        base_families = {"TP", "HS", "AW", "text"}
        extended_families = {"TS", "OR", "LR"}
        family_set = set(families)
        if (
            len(family_set) != len(families)
            or family_set not in (base_families, base_families | extended_families)
        ):
            raise ValueError("safetensors feature_families is invalid")
        extended_enabled = family_set == base_families | extended_families
        allowed_metadata_keys = _BASE_METADATA_KEYS | (
            _EXTENDED_METADATA_KEYS if extended_enabled else set()
        )
        unexpected_metadata_keys = set(metadata) - allowed_metadata_keys
        if unexpected_metadata_keys:
            raise ValueError(
                "safetensors metadata keys are invalid: "
                f"unexpected={sorted(unexpected_metadata_keys)}"
            )
        expected_keys = _BASE_TENSOR_KEYS | (
            _EXTENDED_TENSOR_KEYS if extended_enabled else set()
        )
        observed_keys = set(handle.keys())
        if observed_keys != expected_keys:
            raise ValueError(
                "safetensors tensor keys are invalid: "
                f"missing={sorted(expected_keys - observed_keys)}, "
                f"unexpected={sorted(observed_keys - expected_keys)}"
            )
        slices = {name: handle.get_slice(name) for name in expected_keys}
        shapes = {
            name: tuple(int(size) for size in tensor_slice.get_shape())
            for name, tensor_slice in slices.items()
        }
        dtypes = {
            name: str(tensor_slice.get_dtype()).upper()
            for name, tensor_slice in slices.items()
        }
        for name in expected_keys:
            allowed = _EXPECTED_DTYPES[name]
            if dtypes[name] not in allowed:
                raise ValueError(
                    f"safetensors {name} dtype {dtypes[name]} is invalid; "
                    f"expected one of {sorted(allowed)}"
                )

        response_tokens = len(input_ids) - response_start
        predictor_rows = len(input_ids) - 1
        top_k = _metadata_int(metadata, "probability_top_k")
        attention_history = _metadata_int(metadata, "attention_history")
        if _metadata_int(metadata, "response_start") != response_start:
            raise ValueError("safetensors metadata response_start is misaligned")
        if _metadata_int(metadata, "response_tokens") != response_tokens:
            raise ValueError("safetensors metadata response_tokens is misaligned")
        if _metadata_int(metadata, "predictor_rows") != predictor_rows:
            raise ValueError("safetensors metadata predictor_rows is misaligned")
        if _metadata_int(metadata, "response_feature_row_start") != response_start - 1:
            raise ValueError("safetensors response feature boundary is misaligned")
        if top_k <= 0 or attention_history <= 0:
            raise ValueError("safetensors configured dimensions must be positive")

        expected_shapes = {
            "input_ids": (len(input_ids),),
            "next_token_ids": (predictor_rows,),
            "response_token_ids": (response_tokens,),
            "response_token_positions": (response_tokens,),
            "predictor_positions": (predictor_rows,),
            "topk_probabilities": (predictor_rows, top_k),
            "topk_token_ids": (predictor_rows, top_k),
        }
        for name, expected_shape in expected_shapes.items():
            if shapes[name] != expected_shape:
                raise ValueError(
                    f"safetensors {name} shape {shapes[name]} differs from {expected_shape}"
                )

        decoder_layers = _metadata_int(metadata, "decoder_layers")
        hidden_size = _metadata_int(metadata, "hidden_size")
        attention_heads = _metadata_int(metadata, "attention_heads")
        max_total_tokens = _metadata_int(metadata, "max_total_tokens")
        probability_chunk_tokens = _metadata_int(metadata, "probability_chunk_tokens")
        if min(
            decoder_layers,
            hidden_size,
            attention_heads,
            max_total_tokens,
            probability_chunk_tokens,
        ) <= 0 or len(input_ids) > max_total_tokens:
            raise ValueError("safetensors architecture or token ceiling is invalid")
        hidden_layers = _integer_list(
            _metadata_list(metadata, "hidden_state_layers"),
            "safetensors hidden_state_layers",
        )
        hidden_indices = _integer_list(
            _metadata_list(metadata, "hidden_state_tuple_indices"),
            "safetensors hidden_state_tuple_indices",
        )
        attention_layers = _integer_list(
            _metadata_list(metadata, "attention_layers"),
            "safetensors attention_layers",
        )
        attention_indices = _integer_list(
            _metadata_list(metadata, "attention_tuple_indices"),
            "safetensors attention_tuple_indices",
        )
        if hidden_indices != _resolved_layers(hidden_layers, decoder_layers + 1):
            raise ValueError("safetensors hidden-state tuple indices are invalid")
        if attention_indices != _resolved_layers(attention_layers, decoder_layers):
            raise ValueError("safetensors attention tuple indices are invalid")
        hidden_shape = shapes["hidden_states"]
        if (
            len(hidden_shape) != 3
            or hidden_shape[0] != predictor_rows
            or hidden_shape[1] != len(hidden_layers)
            or hidden_shape[1] != len(hidden_indices)
            or hidden_shape[2] != hidden_size
        ):
            raise ValueError("safetensors hidden_states shape is inconsistent with metadata")
        attention_shape = shapes["attention_window"]
        if (
            len(attention_shape) != 4
            or attention_shape[0] != predictor_rows
            or attention_shape[1] != attention_history
            or attention_shape[2] != attention_heads
            or attention_shape[3] != len(attention_layers)
            or attention_shape[3] != len(attention_indices)
        ):
            raise ValueError("safetensors attention_window shape is inconsistent with metadata")
        vocabulary_size = _metadata_int(metadata, "vocabulary_size")
        if vocabulary_size <= 0:
            raise ValueError("safetensors vocabulary_size is invalid")
        gpu_total = _metadata_int(metadata, "gpu_total_memory_bytes")
        gpu_allocated = _metadata_int(metadata, "gpu_peak_allocated_bytes")
        gpu_reserved = _metadata_int(metadata, "gpu_peak_reserved_bytes")
        if not (0 <= gpu_allocated <= gpu_reserved <= gpu_total):
            raise ValueError("safetensors GPU memory metadata is invalid")
        if metadata.get("gpu_peak_scope") != "teacher_forcing_through_compact_reductions":
            raise ValueError("safetensors gpu_peak_scope metadata is invalid")
        attention_dtype = metadata.get("attention_dtype")
        expected_attention_dtype = {
            "float16": "F16",
            "bfloat16": "BF16",
            "float32": "F32",
        }.get(attention_dtype)
        if expected_attention_dtype is None or dtypes["attention_window"] != expected_attention_dtype:
            raise ValueError("safetensors attention_dtype metadata is invalid")

        if extended_enabled:
            for name, expected in (
                ("layer_top_axis_order", "predictor,layer,rank"),
                (
                    "token_similarity_axis_order",
                    "predictor,adjacent_layer_pair,rank",
                ),
                (
                    "output_rank_axis_order",
                    "predictor,public_hidden_tuple_index",
                ),
                ("lookback_axis_order", "predictor,decoder_layer,head"),
                ("attention_mass_axis_order", "predictor,decoder_layer,head"),
            ):
                if metadata.get(name) != expected:
                    raise ValueError(f"safetensors metadata {name} is invalid")
            if metadata.get("intermediate_projection") != "lm_head_raw_logits":
                raise ValueError("safetensors intermediate projection is invalid")
            if (
                metadata.get("output_rank_behavior")
                != "loader_ignores_config_uses_all_decoder_layers"
            ):
                raise ValueError("safetensors output rank behavior is invalid")
            if (
                metadata.get("output_rank_alignment")
                != "public_training_row_lag_first_row_one"
            ):
                raise ValueError("safetensors output rank alignment is invalid")
            if metadata.get("lookback_behavior") != "public_training_forward_all_heads":
                raise ValueError("safetensors lookback behavior is invalid")
            if _metadata_int(metadata, "intermediate_chunk_tokens") != 32:
                raise ValueError("safetensors intermediate chunk size is invalid")

            intermediate_top_k = _metadata_int(metadata, "intermediate_top_k")
            layer_top_layers = _integer_list(
                _metadata_list(metadata, "layer_top_logits_layers"),
                "safetensors layer_top_logits_layers",
            )
            layer_top_indices = _integer_list(
                _metadata_list(metadata, "layer_top_logits_tuple_indices"),
                "safetensors layer_top_logits_tuple_indices",
            )
            similarity_layers = _integer_list(
                _metadata_list(metadata, "token_similarity_layers"),
                "safetensors token_similarity_layers",
            )
            similarity_indices = _integer_list(
                _metadata_list(metadata, "token_similarity_tuple_indices"),
                "safetensors token_similarity_tuple_indices",
            )
            rank_indices = _metadata_list(metadata, "output_rank_tuple_indices")
            output_rank_layers = _metadata_list(metadata, "output_rank_layers")
            lookback_layers = _metadata_list(metadata, "lookback_layers")
            if layer_top_indices != _resolved_layers(
                layer_top_layers, decoder_layers + 1
            ):
                raise ValueError("safetensors layer top tuple indices are invalid")
            if similarity_indices != _resolved_layers(
                similarity_layers, decoder_layers + 1
            ):
                raise ValueError("safetensors similarity tuple indices are invalid")
            if rank_indices != list(range(decoder_layers)):
                raise ValueError("safetensors output ranks do not preserve public all-layer behavior")
            if output_rank_layers != rank_indices or lookback_layers != list(
                range(decoder_layers)
            ):
                raise ValueError("safetensors public layer index lists are invalid")
            if (
                metadata.get("output_rank_index_semantics")
                != "public_hidden_tuple_indices_includes_embedding_omits_final_hidden"
                or metadata.get("lookback_index_semantics")
                != "attention_tuple_decoder_layer_indices"
            ):
                raise ValueError("safetensors public layer index semantics are invalid")
            extended_shapes = {
                "layer_top_logits": (
                    predictor_rows,
                    len(layer_top_layers),
                    intermediate_top_k,
                ),
                "layer_top_token_ids": (
                    predictor_rows,
                    len(layer_top_layers),
                    intermediate_top_k,
                ),
                "token_similarities": (
                    predictor_rows,
                    len(similarity_layers) - 1,
                    intermediate_top_k,
                ),
                "output_ranks": (predictor_rows, len(rank_indices)),
                "lookback_lens": (
                    predictor_rows,
                    len(rank_indices),
                    attention_shape[2],
                ),
                "attention_prompt_mass": (
                    predictor_rows,
                    len(rank_indices),
                    attention_shape[2],
                ),
                "attention_generated_mass": (
                    predictor_rows,
                    len(rank_indices),
                    attention_shape[2],
                ),
            }
            if len(layer_top_layers) != len(layer_top_indices):
                raise ValueError("safetensors layer top metadata is inconsistent")
            if len(similarity_layers) < 2 or len(similarity_layers) != len(
                similarity_indices
            ):
                raise ValueError("safetensors similarity metadata is inconsistent")
            for name, expected_shape in extended_shapes.items():
                if shapes[name] != expected_shape:
                    raise ValueError(
                        f"safetensors {name} shape {shapes[name]} differs from {expected_shape}"
                    )

        if config is not None:
            _validate_config_contract(metadata, shapes, config, len(input_ids))

        import torch

        for name in expected_keys:
            tensor = handle.get_tensor(name)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"safetensors {name} contains non-finite values")
        top_ids = handle.get_tensor("topk_token_ids")
        if bool((top_ids < 0).any()) or bool((top_ids >= vocabulary_size).any()):
            raise ValueError("safetensors topk_token_ids are outside the vocabulary")
        probabilities = handle.get_tensor("topk_probabilities").float()
        if bool((probabilities < 0).any()) or bool((probabilities > 1).any()):
            raise ValueError("safetensors topk_probabilities are outside [0, 1]")
        if bool((handle.get_tensor("attention_window").float() < 0).any()):
            raise ValueError("safetensors attention_window contains negative values")
        if extended_enabled:
            layer_top_ids = handle.get_tensor("layer_top_token_ids")
            if bool((layer_top_ids < 0).any()) or bool(
                (layer_top_ids >= vocabulary_size).any()
            ):
                raise ValueError("safetensors layer_top_token_ids are outside the vocabulary")
            ranks = handle.get_tensor("output_ranks")
            if bool((ranks < 1).any()) or bool((ranks > vocabulary_size).any()):
                raise ValueError("safetensors output_ranks are outside the vocabulary rank range")
            similarities = handle.get_tensor("token_similarities")
            if bool((similarities < -1.0001).any()) or bool((similarities > 1.0001).any()):
                raise ValueError("safetensors token_similarities are outside [-1, 1]")
            lookback = handle.get_tensor("lookback_lens")
            if bool((lookback < 0).any()) or bool((lookback > 1.0001).any()):
                raise ValueError("safetensors lookback_lens are outside [0, 1]")
            prompt_mass = handle.get_tensor("attention_prompt_mass")
            generated_mass = handle.get_tensor("attention_generated_mass")
            if bool((prompt_mass < 0).any()) or bool((generated_mass < 0).any()):
                raise ValueError("safetensors attention masses contain negative values")
            total_mass = prompt_mass + generated_mass
            if bool((total_mass < 0.9).any()) or bool((total_mass > 1.1).any()):
                raise ValueError("safetensors attention masses do not sum to one")

        tensor_values = {
            name: _integer_list(handle.get_tensor(name).tolist(), f"safetensors {name}")
            for name in (
                "input_ids",
                "next_token_ids",
                "response_token_ids",
                "response_token_positions",
                "predictor_positions",
            )
        }

    expected_positions = list(range(response_start, len(input_ids)))
    if tensor_values["input_ids"] != input_ids:
        raise ValueError("safetensors input_ids differ from output input_ids")
    if tensor_values["next_token_ids"] != input_ids[1:]:
        raise ValueError("safetensors next_token_ids are misaligned")
    if tensor_values["response_token_positions"] != expected_positions:
        raise ValueError("safetensors response_token_positions are misaligned")
    if tensor_values["predictor_positions"] != list(range(len(input_ids) - 1)):
        raise ValueError("safetensors predictor_positions are misaligned")
    if tensor_values["response_token_ids"] != input_ids[response_start:]:
        raise ValueError("safetensors response_token_ids are misaligned")


def sample_complete(paths: SamplePaths, sample_id: str, protocol_sha256: str) -> bool:
    if not paths.done.is_file():
        return False
    try:
        value = json.loads(paths.done.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("schema") != "saner_sample_done":
            return False
        if value.get("sample_id") != sample_id or value.get("protocol_sha256") != protocol_sha256:
            return False
        if value.get("track") not in {"original"}:
            return False
        output = _read_json_mapping(paths.output, "output")
        labels = _read_json_mapping(paths.labels, "labels")
        for document in (output, labels):
            if (
                document.get("run_id") != value.get("run_id")
                or document.get("track") != value.get("track")
                or document.get("protocol_sha256") != protocol_sha256
                or document.get("source_row_sha256") != value.get("source_row_sha256")
            ):
                return False
        for path in (paths.output, paths.labels, paths.features):
            expected = value["files"][path.name]
            if not path.is_file() or path.stat().st_size != expected["bytes"]:
                return False
            if sha256_file(path) != expected["sha256"]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _manifest_rows(
    output_dir: Path,
    protocol_sha256: str,
    run_id: str,
    track: str,
    expected_total: int | None = None,
    expected_views: Mapping[str, Mapping[str, Any]] | None = None,
    expected_annotations: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    path = output_dir / "manifests" / "samples.jsonl"
    index_path = output_dir / "manifests" / "manifest.json"
    if not path.is_file() or not index_path.is_file():
        raise ValueError("run is not prepared")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest metadata is invalid") from exc
    if not isinstance(index, Mapping) or index.get("schema") != "saner_paper_manifest":
        raise ValueError("manifest schema is invalid")
    if index.get("run_id") != run_id:
        raise ValueError("manifest run_id differs from config")
    if index.get("track") != track:
        raise ValueError("manifest track differs from config")
    if index.get("protocol_sha256") != protocol_sha256:
        raise ValueError("manifest belongs to a different protocol")
    if index.get("samples_sha256") != sha256_file(path):
        raise ValueError("manifest checksum does not match samples.jsonl")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("samples manifest is invalid") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("samples manifest rows must be JSON objects")
    if any(
        row.get("schema") != "paper_manifest_sample"
        or row.get("run_id") != run_id
        or row.get("track") != track
        for row in rows
    ):
        raise ValueError("samples manifest rows have an invalid identity")
    if index.get("total_samples") != len(rows):
        raise ValueError("manifest total_samples does not match samples.jsonl")
    if expected_total is not None and len(rows) != expected_total:
        raise ValueError("manifest total_samples differs from config")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("manifest sample_id values must be non-empty and unique")
    view_index = index.get("views")
    if not isinstance(view_index, Mapping):
        raise ValueError("manifest views index is invalid")
    if expected_views is not None and set(map(str, view_index)) != set(expected_views):
        raise ValueError("manifest views differ from config")
    sample_id_set = set(sample_ids)
    sample_splits = {
        str(row["sample_id"]): {
            str(occurrence.get("split"))
            for occurrence in row.get("occurrences", [])
            if isinstance(occurrence, Mapping)
        }
        for row in rows
    }
    for view_name, descriptor in view_index.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError("manifest view descriptor is invalid")
        relative = Path(str(descriptor.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest view path is invalid")
        view_path = output_dir / "manifests" / relative
        if not view_path.is_file() or sha256_file(view_path) != descriptor.get("sha256"):
            raise ValueError(f"manifest view checksum differs: {view_name}")
        try:
            view_rows = [
                json.loads(line)
                for line in view_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest view is invalid: {view_name}") from exc
        expected_view = expected_views.get(str(view_name), {}) if expected_views else {}
        required_labels = set(map(str, expected_view.get("extra_label_fields", [])))
        split_counts = Counter(str(row.get("source_split")) for row in view_rows)
        expected_split_counts = {
            str(name): int(count)
            for name, count in expected_view.get("expected_split_counts", {}).items()
        }
        source_locations = [
            (str(row.get("source_split")), row.get("source_row_index"))
            for row in view_rows
        ]
        if (
            len(view_rows) != descriptor.get("rows")
            or ("expected_rows" in expected_view and len(view_rows) != int(expected_view["expected_rows"]))
            or (expected_split_counts and dict(split_counts) != expected_split_counts)
            or len(source_locations) != len(set(source_locations))
            or any(
            not isinstance(row, Mapping)
            or row.get("schema") != "paper_view_member"
            or row.get("run_id") != run_id
            or row.get("track") != track
            or row.get("view") != view_name
            or row.get("sample_id") not in sample_id_set
            or not isinstance(row.get("source_split"), str)
            or not isinstance(row.get("source_row_index"), int)
            or int(row.get("source_row_index", -1)) < 0
            or row.get("source_split") not in sample_splits.get(str(row.get("sample_id")), set())
            or row.get("occurrences") != [{
                "split": row.get("source_split"),
                "row_index": row.get("source_row_index"),
            }]
            or not isinstance(row.get("labels"), Mapping)
            or not required_labels.issubset(set(map(str, row.get("labels", {}))))
            for row in view_rows
            )
        ):
            raise ValueError(f"manifest view rows are invalid: {view_name}")

    annotations_path = output_dir / "manifests" / "annotations.json"
    if (
        not annotations_path.is_file()
        or index.get("annotations_sha256") != sha256_file(annotations_path)
    ):
        raise ValueError("annotation manifest checksum differs")
    annotations = _read_json_mapping(annotations_path, "annotation manifest")
    if (
        annotations.get("schema") != "paper_annotation_manifest"
        or annotations.get("track") != track
        or not isinstance(annotations.get("files"), list)
    ):
        raise ValueError("annotation manifest is invalid")
    if expected_annotations is not None and {
        str(record.get("path")): str(record.get("sha256"))
        for record in annotations["files"]
        if isinstance(record, Mapping)
    } != dict(expected_annotations):
        raise ValueError("annotation manifest differs from config")
    for record in annotations["files"]:
        relative = Path(str(record.get("copied_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("annotation copied path is invalid")
        copied = output_dir / "manifests" / relative
        if not copied.is_file() or sha256_file(copied) != record.get("sha256"):
            raise ValueError("copied annotation checksum differs")
    return [dict(row) for row in rows]


def _expected_total(config: Mapping[str, Any]) -> int | None:
    source = config.get("source")
    if isinstance(source, Mapping) and "expected_unique_samples" in source:
        return int(source["expected_unique_samples"])
    return None


def _expected_views(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(view["name"]): view
        for view in config["source"].get("views", [])
        if view.get("role") == "subset"
    }


def _expected_annotations(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(annotation["path"]): str(annotation["sha256"])
        for annotation in config["source"].get("annotations", [])
    }


def _validate_source_metadata(
    output_dir: Path, config: Mapping[str, Any]
) -> None:
    source = _read_json_mapping(
        output_dir / "metadata" / "source.json", "source metadata"
    )
    if source.get("schema") != "paper_release_source":
        raise ValueError("source metadata schema is invalid")
    if source.get("track") != str(config["track"]):
        raise ValueError("source metadata track differs from config")
    for name in ("revision", "archive_sha256", "source_model"):
        if source.get(name) != str(config["source"][name]):
            raise ValueError(f"source metadata {name} differs from config")


def _validate_run_state(
    output_dir: Path,
    config: Mapping[str, Any],
    protocol_sha256: str,
    *,
    required: bool,
) -> None:
    path = output_dir / "state" / "run_state.json"
    if not path.is_file():
        if required:
            raise ValueError("run state is missing")
        return
    state = _read_json_mapping(path, "run state")
    if state.get("schema") != "saner_run_state":
        raise ValueError("run state schema is invalid")
    if state.get("run_id") != str(config["run_id"]):
        raise ValueError("run state run_id differs from config")
    if state.get("track") != str(config["track"]):
        raise ValueError("run state track differs from config")
    if state.get("protocol_sha256") != protocol_sha256:
        raise ValueError("run state protocol_sha256 differs from config")


def _environment_sha256(
    output_dir: Path, config: Mapping[str, Any]
) -> str:
    path = output_dir / "metadata" / "environment.json"
    lock_path = output_dir / "metadata" / "environment_lock.json"
    if not path.is_file() or not lock_path.is_file():
        raise ValueError("runtime environment metadata or checksum is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("runtime environment metadata is invalid") from exc
    try:
        checked = validate_frozen_environment(value)
    except RuntimeError as exc:
        raise ValueError("runtime environment violates runtime_lock.json") from exc
    if checked != value:
        raise ValueError("runtime environment metadata is not canonical")
    try:
        validate_model_against_config(checked["model"], config)
    except RuntimeError as exc:
        raise ValueError(
            "runtime environment model differs from experiment config"
        ) from exc
    observed = sha256_file(path)
    if (
        not isinstance(lock, Mapping)
        or set(lock)
        != {"schema", "environment_sha256", "requirements_sha256"}
        or lock.get("schema") != "saner_runtime_environment_lock"
        or lock.get("environment_sha256") != observed
        or lock.get("requirements_sha256") != checked["requirements_sha256"]
    ):
        raise ValueError("runtime environment checksum does not match environment.json")
    _execution_inventory_sha256(output_dir, checked)
    return observed


def _execution_inventory_sha256(
    output_dir: Path, environment: Mapping[str, Any] | None = None
) -> str:
    if environment is None:
        environment_path = output_dir / "metadata" / "environment.json"
        try:
            loaded = json.loads(environment_path.read_text(encoding="utf-8"))
            environment = validate_frozen_environment(loaded)
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise ValueError("runtime environment metadata is invalid") from exc
    execution_dir = output_dir / "metadata" / "executions"
    execution_paths = sorted(execution_dir.glob("*.json"))
    if not execution_paths:
        raise ValueError("execution environment provenance is missing")
    stable_runtime = {
        key: item for key, item in environment.items() if key != "model"
    }
    inventory: list[dict[str, Any]] = []
    for execution_path in execution_paths:
        record = _read_json_mapping(execution_path, "execution environment")
        if (
            set(record)
            != {"schema", "runtime", "execution_context", "model"}
            or record.get("schema") != "saner_execution_environment"
            or record.get("runtime") != stable_runtime
            or record.get("model") != environment.get("model")
        ):
            raise ValueError("execution environment record is invalid")
        try:
            execution_frozen = validate_frozen_environment(
                {**record["runtime"], "model": record["model"]}
            )
            checked_context = validate_execution_context(
                record["execution_context"]
            )
        except (TypeError, RuntimeError) as exc:
            raise ValueError("execution environment record is invalid") from exc
        if (
            execution_frozen != environment
            or checked_context != record["execution_context"]
        ):
            raise ValueError("execution environment record is invalid")
        identity = hashlib.sha256(canonical_bytes(record)).hexdigest()
        if execution_path.name != f"{identity}.json":
            raise ValueError("execution environment checksum is invalid")
        inventory.append(
            {
                "path": execution_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(execution_path),
                "bytes": execution_path.stat().st_size,
                "identity_sha256": identity,
            }
        )
    return hashlib.sha256(canonical_bytes(inventory)).hexdigest()


def _validate_completion_marker(path: Path, expected: Mapping[str, Any]) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("completion marker is invalid") from exc
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("completion marker differs from verified run metadata")
    if not isinstance(value.get("completed_at"), str):
        raise ValueError("completion marker is invalid")


def status(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path | None = None
) -> dict[str, Any]:
    del cache_dir
    validate_output_root(output_dir, config)
    protocol_sha256 = protocol_hash(config)
    run_dir = model_output_dir(output_dir, config)
    model = str(config["source"]["source_model"])
    ensure_protocol(run_dir, config, protocol_sha256, create=False)
    rows = _manifest_rows(
        run_dir,
        protocol_sha256,
        str(config["run_id"]),
        str(config["track"]),
        _expected_total(config),
        _expected_views(config),
        _expected_annotations(config),
    )
    _validate_source_metadata(run_dir, config)
    _validate_run_state(run_dir, config, protocol_sha256, required=False)
    complete = sum(
        sample_complete(
            sample_paths(
                output_dir,
                model,
                str(row["dataset"]),
                str(row["sample_id"]),
            ),
            str(row["sample_id"]),
            protocol_sha256,
        )
        for row in rows
    )
    return {
        "track": str(config["track"]),
        "total_samples": len(rows),
        "complete_samples": complete,
        "pending_samples": len(rows) - complete,
    }


def verify(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path | None = None
) -> dict[str, Any]:
    del cache_dir
    validate_output_root(output_dir, config)
    protocol_sha256 = protocol_hash(config)
    run_dir = model_output_dir(output_dir, config)
    model = str(config["source"]["source_model"])
    ensure_protocol(run_dir, config, protocol_sha256, create=False)
    rows = _manifest_rows(
        run_dir,
        protocol_sha256,
        str(config["run_id"]),
        str(config["track"]),
        _expected_total(config),
        _expected_views(config),
        _expected_annotations(config),
    )
    _validate_source_metadata(run_dir, config)
    _validate_run_state(run_dir, config, protocol_sha256, required=False)
    invalid = []
    for row in rows:
        sample_id = str(row["sample_id"])
        paths = sample_paths(output_dir, model, str(row["dataset"]), sample_id)
        if not sample_complete(paths, sample_id, protocol_sha256):
            invalid.append(sample_id)
            continue
        try:
            validate_written_sample(
                paths,
                str(config["run_id"]),
                sample_id,
                protocol_sha256,
                config=config,
                expected_source_row_sha256=str(row["source_row_sha256"]),
                manifest_row=row,
            )
        except ValueError:
            invalid.append(sample_id)
    valid_samples = len(rows) - len(invalid)
    environment_sha256 = (
        _environment_sha256(run_dir, config) if valid_samples else None
    )
    if not invalid:
        locked_smoke_start = smoke_sample_start(str(config["run_id"]))
        if locked_smoke_start >= len(rows):
            raise ValueError("locked smoke sample is outside the prepared manifest")
        smoke_row = rows[locked_smoke_start]
        smoke_paths = sample_paths(
            output_dir,
            model,
            str(smoke_row["dataset"]),
            str(smoke_row["sample_id"]),
        )
        try:
            validate_smoke_marker(
                run_dir / "state",
                run_id=str(config["run_id"]),
                track=str(config["track"]),
                protocol_sha256=protocol_sha256,
                sample_start=locked_smoke_start,
                sample_id=str(smoke_row["sample_id"]),
                done_path=smoke_paths.done,
                environment_sha256=str(environment_sha256),
            )
        except RuntimeError as exc:
            raise ValueError("smoke gate verification failed") from exc
        _validate_run_state(run_dir, config, protocol_sha256, required=True)
        completed_path = run_dir / "state" / "COMPLETED.json"
        if not completed_path.is_file():
            raise ValueError("completion marker is missing; run once to restore it")
        _validate_completion_marker(
            completed_path,
            {
                "schema": "saner_run_completed",
                "run_id": str(config["run_id"]),
                "track": str(config["track"]),
                "protocol_sha256": protocol_sha256,
                "total_samples": len(rows),
                "manifest_sha256": sha256_file(
                    run_dir / "manifests" / "samples.jsonl"
                ),
                "environment_sha256": environment_sha256,
                "execution_inventory_sha256": _execution_inventory_sha256(
                    run_dir
                ),
            },
        )
    return {
        "track": str(config["track"]),
        "total_samples": len(rows),
        "valid_samples": valid_samples,
        "invalid_samples": invalid,
    }
