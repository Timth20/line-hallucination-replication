from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_BASE_FEATURE_FIELDS = {
    "feature_families",
    "reference_repository",
    "reference_revision",
    "reference_extractors",
    "published_protocols",
    "hidden_state_layers",
    "probability_top_k",
    "attention_layers",
    "attention_history",
    "max_total_tokens",
    "probability_chunk_tokens",
    "predictor_scope",
}

_EXTENDED_FEATURE_FIELDS = {
    "configuration_repository",
    "configuration_revision",
    "configuration_path",
    "uhead_configuration_path",
    "lookback_configuration_path",
    "layer_top_logits_layers",
    "token_similarity_layers",
    "intermediate_top_k",
    "intermediate_chunk_tokens",
    "intermediate_projection",
    "output_rank_layers",
    "output_rank_behavior",
    "output_rank_alignment",
    "lookback_layers",
    "lookback_behavior",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def implementation_hash(package_dir: Path | None = None) -> str:
    root = package_dir or Path(__file__).resolve().parent
    digest = hashlib.sha256()
    paths = set(root.rglob("*.py"))
    paths.update(
        path for name in ("runtime_lock.json", "smoke_lock.json")
        if (path := root / name).is_file()
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def _layer_list(
    features: Mapping[str, Any], name: str, tuple_length: int, *, minimum: int = 1
) -> list[int]:
    values = features.get(name)
    if (
        not isinstance(values, list)
        or len(values) < minimum
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"features.{name} must be a list of unique integer layers")
    for value in values:
        resolved = tuple_length + value if value < 0 else value
        if resolved < 0 or resolved >= tuple_length:
            raise ValueError(f"features.{name} contains unavailable layer {value}")
    return values


def validate_config(value: Mapping[str, Any]) -> None:
    required = {"schema", "run_id", "track", "runtime", "source", "model", "features"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"config is missing sections: {sorted(missing)}")
    if value["schema"] != "saner_paper_extraction":
        raise ValueError("schema must be saner_paper_extraction")
    if not str(value["run_id"]).startswith("paper_"):
        raise ValueError("run_id must start with paper_")
    if value["track"] not in {"original"}:
        raise ValueError(
            "track must be original"
        )

    source = value["source"]
    source_required = {
        "repository",
        "revision",
        "archive_url",
        "archive_sha256",
        "data_subdir",
        "dataset_dir",
        "source_model",
        "splits",
        "datasets",
        "expected_unique_samples",
        "expected_physical_rows",
    }
    if source.get("kind") != "paper_release" or source_required - set(source):
        raise ValueError("source must describe the complete paper release")
    if not _is_hex(source["revision"], 40):
        raise ValueError("source.revision must be an immutable 40-character commit SHA")
    if not _is_hex(source["archive_sha256"], 64):
        raise ValueError("source.archive_sha256 must be a 64-character SHA-256")
    if not source["datasets"] or min(
        int(source[name]) for name in ("expected_unique_samples", "expected_physical_rows")
    ) <= 0:
        raise ValueError("source datasets and row counts must be non-empty and positive")
    track_by_source_model = {
        "deepseekcoder_1.3b": "original",
        "deepseekcoder_6.7b": "original",
        "codellama_7b": "original",
        "deepseekcoder_33b": "original",
    }
    source_model = source.get("source_model")
    if source_model not in track_by_source_model:
        raise ValueError(f"unsupported source model: {source_model}")
    expected_track = track_by_source_model[source_model]
    if value["track"] != expected_track:
        raise ValueError(
            f"track must be {expected_track} for source model {source.get('source_model')}"
        )
    views = source.get("views")
    if not isinstance(views, list) or not views or views[0].get("role") != "model_forward":
        raise ValueError("source.views must start with the model_forward view")
    if (
        views[0].get("dataset_dir") != source.get("dataset_dir")
        or int(views[0].get("expected_rows", 0))
        != int(source.get("expected_unique_samples", 0))
    ):
        raise ValueError("source main view must match the deduplicated model-forward source")
    view_names = [str(view.get("name", "")) for view in views]
    if any(not name for name in view_names) or len(set(view_names)) != len(view_names):
        raise ValueError("source.views names must be non-empty and unique")
    for view in views:
        if not isinstance(view, Mapping) or not str(view.get("dataset_dir", "")):
            raise ValueError("each source view must name its released dataset directory")
        if view.get("role") not in {"model_forward", "subset"}:
            raise ValueError("source view role must be model_forward or subset")
        if int(view.get("expected_rows", 0)) <= 0:
            raise ValueError("source view expected_rows must be positive")
        fields = view.get("extra_label_fields", [])
        if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
            raise ValueError("source view extra_label_fields must be a string list")
        split_counts = view.get("expected_split_counts")
        if split_counts is not None and (
            not isinstance(split_counts, Mapping)
            or any(not isinstance(name, str) or int(count) <= 0 for name, count in split_counts.items())
            or sum(int(count) for count in split_counts.values()) != int(view["expected_rows"])
        ):
            raise ValueError("source view expected_split_counts must sum to expected_rows")
    public_gaps = source.get("public_gaps", [])
    if not isinstance(public_gaps, list) or any(
        not isinstance(gap, Mapping)
        or not str(gap.get("name", ""))
        or gap.get("status") != "not_in_pinned_release"
        for gap in public_gaps
    ):
        raise ValueError("source.public_gaps must explicitly describe absent published inputs")
    annotations = source.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("source.annotations must be a list")
    for annotation in annotations:
        if (
            not isinstance(annotation, Mapping)
            or not str(annotation.get("path", ""))
            or not _is_hex(annotation.get("sha256", ""), 64)
        ):
            raise ValueError("source annotation paths and checksums must be pinned")
    splits = source["splits"]
    if (
        not isinstance(splits, list)
        or not splits
        or any(not isinstance(name, str) or not name for name in splits)
        or len(set(splits)) != len(splits)
    ):
        raise ValueError("source.splits must be a non-empty list of unique names")

    model = value["model"]
    if not _is_hex(model.get("revision", ""), 40):
        raise ValueError("model.revision must be an immutable 40-character commit SHA")
    if model.get("dtype") not in {"bfloat16", "float16"}:
        raise ValueError("model.dtype must be bfloat16 or float16")
    if model.get("attention_implementation") != "eager":
        raise ValueError("model.attention_implementation must be eager")
    expected_shape = model.get("expected_shape", {})
    if any(int(expected_shape.get(name, 0)) <= 0 for name in ("layers", "hidden_size", "attention_heads")):
        raise ValueError("model.expected_shape must contain positive architecture dimensions")

    features = value["features"]
    layer_count = int(expected_shape["layers"])
    _layer_list(features, "hidden_state_layers", layer_count + 1)
    _layer_list(features, "attention_layers", layer_count)
    for name in (
        "probability_top_k",
        "attention_history",
        "max_total_tokens",
        "probability_chunk_tokens",
    ):
        if int(features.get(name, 0)) <= 0:
            raise ValueError(f"features.{name} must be positive")
    if not _is_hex(features.get("reference_revision", ""), 40):
        raise ValueError("features.reference_revision must be a 40-character commit SHA")
    if features.get("predictor_scope") != "full_sequence":
        raise ValueError("features.predictor_scope must be full_sequence")
    protocols = features.get("published_protocols")
    if not isinstance(protocols, Mapping) or "main" not in protocols:
        raise ValueError("features.published_protocols must record the main paper protocol")
    main_protocol = protocols["main"]
    if not isinstance(main_protocol, Mapping):
        raise ValueError("features.published_protocols.main must be a mapping")
    for name in ("probability_top_k", "hidden_state_layers", "attention_layers", "attention_history"):
        if name not in main_protocol:
            raise ValueError(f"features.published_protocols.main is missing {name}")
    if (
        int(main_protocol["probability_top_k"]) > int(features["probability_top_k"])
        or int(main_protocol["attention_history"]) > int(features["attention_history"])
        or not set(main_protocol["hidden_state_layers"]).issubset(
            features["hidden_state_layers"]
        )
        or not set(main_protocol["attention_layers"]).issubset(
            features["attention_layers"]
        )
    ):
        raise ValueError("features extraction selectors do not cover the main protocol")
    transformer_attention_union = protocols.get(
        "published_transformer_attention_union"
    )
    if not isinstance(transformer_attention_union, Mapping):
        raise ValueError(
            "features.published_protocols must record the published Transformer attention union"
        )
    union_layers = transformer_attention_union.get("attention_layers")
    union_history = transformer_attention_union.get("attention_history")
    if (
        not isinstance(union_layers, list)
        or not union_layers
        or not set(union_layers).issubset(features["attention_layers"])
        or int(union_history or 0) > int(features["attention_history"])
    ):
        raise ValueError(
            "features extraction selectors do not cover the published Transformer attention union"
        )
    families = features.get("feature_families")
    allowed_families = {"TP", "HS", "AW", "TS", "OR", "LR", "text"}
    required_families = {"TP", "HS", "AW", "text"}
    if (
        not isinstance(families, list)
        or not families
        or len(set(families)) != len(families)
        or not required_families.issubset(families)
        or not set(families).issubset(allowed_families)
    ):
        raise ValueError("features.feature_families is invalid")

    extended = {"TS", "OR", "LR"}
    enabled_extended = extended.intersection(families)
    if enabled_extended and enabled_extended != extended:
        raise ValueError("TS, OR, and LR must be enabled together")
    allowed_feature_fields = _BASE_FEATURE_FIELDS | (
        _EXTENDED_FEATURE_FIELDS if enabled_extended else set()
    )
    unexpected_feature_fields = set(features) - allowed_feature_fields
    if unexpected_feature_fields:
        raise ValueError(
            "features contains unexpected fields: "
            f"{sorted(unexpected_feature_fields)}"
        )
    expected_protocol_names = {"main", "published_transformer_attention_union"}
    if enabled_extended:
        expected_protocol_names.update({"factoscope", "lookback"})
    if set(protocols) != expected_protocol_names:
        raise ValueError("features.published_protocols has invalid protocol names")
    expected_reference_extractors = {
        "token_probabilities.py",
        "basic_hidden_states.py",
        "basic_attention.py",
    }
    if enabled_extended:
        expected_reference_extractors.update(
            {
                "token_similarities_adjacent_layers.py",
                "token_probabilities_from_layers.py",
                "final_output_ranks.py",
                "lookback_lens.py",
            }
        )
    reference_extractors = features.get("reference_extractors")
    if (
        not isinstance(reference_extractors, list)
        or len(reference_extractors) != len(set(reference_extractors))
        or set(reference_extractors) != expected_reference_extractors
    ):
        raise ValueError("features.reference_extractors is invalid")
    extended_names = {
        "layer_top_logits_layers",
        "token_similarity_layers",
        "intermediate_top_k",
        "intermediate_chunk_tokens",
        "output_rank_layers",
        "output_rank_alignment",
        "lookback_layers",
        "uhead_configuration_path",
    }
    if enabled_extended:
        if not isinstance(protocols.get("factoscope"), Mapping) or not isinstance(
            protocols.get("lookback"), Mapping
        ):
            raise ValueError("extended features must record FactoScope and Lookback protocols")
        if not _is_hex(features.get("configuration_revision", ""), 40):
            raise ValueError("features.configuration_revision must be a 40-character commit SHA")
        if not all(
            isinstance(features.get(name), str) and features[name]
            for name in (
                "configuration_repository",
                "configuration_path",
                "uhead_configuration_path",
                "lookback_configuration_path",
            )
        ):
            raise ValueError("features must pin the published configuration source")
        layer_top_logits_layers = _layer_list(
            features, "layer_top_logits_layers", layer_count + 1
        )
        token_similarity_layers = _layer_list(
            features, "token_similarity_layers", layer_count + 1, minimum=2
        )
        factoscope = protocols["factoscope"]
        stored_similarity_indices = {
            layer_count + 1 + layer if layer < 0 else layer
            for layer in token_similarity_layers
        }
        linear_similarity = factoscope.get("linear_token_similarity_layers")
        uhead_similarity = factoscope.get("uhead_token_similarity_layers")
        if not isinstance(linear_similarity, list) or (
            uhead_similarity != "all" and not isinstance(uhead_similarity, list)
        ):
            raise ValueError("FactoScope similarity selectors are invalid")
        required_similarity = list(linear_similarity)
        if uhead_similarity == "all":
            required_similarity.extend(range(layer_count + 1))
        else:
            required_similarity.extend(uhead_similarity)
        required_similarity_indices = {
            layer_count + 1 + int(layer) if int(layer) < 0 else int(layer)
            for layer in required_similarity
        }
        if (
            not set(factoscope.get("hidden_state_layers", [])).issubset(
                features["hidden_state_layers"]
            )
            or factoscope.get("layer_top_logits_layers") != layer_top_logits_layers
            or not required_similarity_indices.issubset(stored_similarity_indices)
            or factoscope.get("implemented_output_rank_layers") != "all"
        ):
            raise ValueError("features extraction selectors do not cover FactoScope")
        lookback = protocols["lookback"]
        if (
            lookback.get("attention_layers") != "all"
            or lookback.get("attention_heads") != "all"
            or lookback.get("behavior") != "public_training_forward_all_heads"
        ):
            raise ValueError("features.published_protocols.lookback is invalid")
        if features.get("intermediate_top_k") != 10 or isinstance(
            features.get("intermediate_top_k"), bool
        ):
            raise ValueError("features.intermediate_top_k must be 10")
        if features.get("intermediate_chunk_tokens") != 32 or isinstance(
            features.get("intermediate_chunk_tokens"), bool
        ):
            raise ValueError("features.intermediate_chunk_tokens must be 32")
        if features.get("output_rank_layers") != "all":
            raise ValueError("features.output_rank_layers must be all")
        if features.get("lookback_layers") != "all":
            raise ValueError("features.lookback_layers must be all")
        if features.get("intermediate_projection") != "lm_head_raw_logits":
            raise ValueError("features.intermediate_projection must be lm_head_raw_logits")
        if (
            features.get("output_rank_behavior")
            != "loader_ignores_config_uses_all_decoder_layers"
        ):
            raise ValueError("features.output_rank_behavior is invalid")
        if features.get("output_rank_alignment") != "public_training_row_lag_first_row_one":
            raise ValueError("features.output_rank_alignment is invalid")
        if features.get("lookback_behavior") != "public_training_forward_all_heads":
            raise ValueError("features.lookback_behavior is invalid")
        required_extractors = {
            "token_similarities_adjacent_layers.py",
            "token_probabilities_from_layers.py",
            "final_output_ranks.py",
            "lookback_lens.py",
        }
        if not required_extractors.issubset(features.get("reference_extractors", [])):
            raise ValueError("features.reference_extractors is missing extractors")
    elif extended_names.intersection(features):
        raise ValueError("unpublished extended features must be absent")

    runtime = value["runtime"]
    if runtime.get("output_root_name") != "outputs_original":
        raise ValueError("runtime.output_root_name must be outputs_original")
    int(runtime["seed"])


def validate_output_root(output_dir: Path, config: Mapping[str, Any]) -> None:
    expected = str(config["runtime"]["output_root_name"])
    if output_dir.expanduser().name != expected:
        raise ValueError(f"--output-dir must end with {expected}")


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    value = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    validate_config(value)
    return value


def protocol_hash(value: Mapping[str, Any]) -> str:
    validate_config(value)
    payload = {"config": value, "implementation_sha256": implementation_hash()}
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()
