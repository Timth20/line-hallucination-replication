from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from .models import validate_released_tokens
from .types import Alignment, FeatureSet, ModelBundle, PaperSample


def response_alignment(input_ids: Sequence[int], labels: Sequence[int]) -> Alignment:
    if len(input_ids) != len(labels) or len(input_ids) < 2:
        raise ValueError("input_ids and labels must have the same non-trivial length")
    response = [index for index, label in enumerate(labels) if int(label) != -100]
    if not response or response[0] == 0:
        raise ValueError("the released row must contain prompt and response tokens")
    if response != list(range(response[0], len(labels))):
        raise ValueError("response labels must form one contiguous suffix")
    targets = tuple(response)
    return Alignment(
        response_start=targets[0],
        target_positions=targets,
        predictor_positions=tuple(position - 1 for position in targets),
    )


def attention_window_indices(query_position: int, history: int) -> tuple[int, ...]:
    if query_position < 0 or history <= 0:
        raise ValueError("query_position must be non-negative and history must be positive")
    return tuple(query_position - offset if query_position >= offset else -1 for offset in range(history))


def resolve_tuple_layers(tuple_length: int, layers: Iterable[int]) -> tuple[int, ...]:
    resolved = []
    for layer in layers:
        index = tuple_length + int(layer) if int(layer) < 0 else int(layer)
        if index < 0 or index >= tuple_length:
            raise ValueError(f"layer {layer} is unavailable for tuple length {tuple_length}")
        resolved.append(index)
    return tuple(resolved)


def feature_metadata(
    run_id: str,
    track: str,
    sample_id: str,
    protocol_sha256: str,
    hidden_layers: Sequence[int],
    attention_layers: Sequence[int],
    attention_history: int,
    probability_top_k: int,
    feature_families: Sequence[str],
) -> dict[str, str]:
    return {
        "schema": "saner_paper_features",
        "run_id": run_id,
        "track": track,
        "sample_id": sample_id,
        "protocol_sha256": protocol_sha256,
        "feature_families": json.dumps(list(feature_families)),
        "hidden_state_layers": json.dumps(list(hidden_layers)),
        "attention_layers": json.dumps(list(attention_layers)),
        "attention_history": str(attention_history),
        "probability_top_k": str(probability_top_k),
        "predictor_scope": "full_sequence",
        "prediction_alignment": "row p predicts input_ids[p+1]",
        "attention_axis_order": "predictor,history,head,layer",
    }


def _on_cpu(tensor: Any, dtype: Any = None) -> Any:
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu().contiguous()


def _probability_features(
    torch: Any,
    logits: Any,
    predictor_positions: Sequence[int],
    target_ids: Sequence[int],
    top_k: int,
    chunk_size: int,
) -> dict[str, Any]:
    if len(target_ids) != len(predictor_positions):
        raise ValueError("target ids must align with predictor positions")
    if top_k > int(logits.shape[-1]):
        raise ValueError(f"probability_top_k {top_k} exceeds vocabulary size {logits.shape[-1]}")

    result: dict[str, list[Any]] = {
        "topk_probabilities": [],
        "topk_token_ids": [],
    }
    for start in range(0, len(predictor_positions), chunk_size):
        positions = predictor_positions[start : start + chunk_size]
        chunk_logits = torch.stack([logits[0, position, :] for position in positions])
        probabilities = torch.softmax(chunk_logits, dim=-1)
        top_probabilities, top_ids = torch.topk(probabilities, top_k, dim=-1)
        result["topk_probabilities"].append(top_probabilities)
        result["topk_token_ids"].append(top_ids)

    return {
        "topk_probabilities": _on_cpu(torch.cat(result["topk_probabilities"])),
        "topk_token_ids": _on_cpu(torch.cat(result["topk_token_ids"]), torch.int32),
    }


def _hidden_state_features(
    torch: Any,
    hidden_states: Sequence[Any],
    predictor_positions: Sequence[int],
    layers: Sequence[int],
) -> tuple[Any, tuple[int, ...]]:
    resolved = resolve_tuple_layers(len(hidden_states), layers)
    selected = torch.stack(
        [
            torch.stack([hidden_states[layer][0, position, :] for layer in resolved])
            for position in predictor_positions
        ]
    )
    return _on_cpu(selected), resolved


def _attention_features(
    torch: Any,
    attentions: Sequence[Any],
    predictor_positions: Sequence[int],
    layers: Sequence[int],
    history: int,
) -> tuple[Any, tuple[int, ...]]:
    resolved = resolve_tuple_layers(len(attentions), layers)
    heads = int(attentions[resolved[0]].shape[1])
    for layer in resolved[1:]:
        if int(attentions[layer].shape[1]) != heads:
            raise ValueError("selected attention layers have different head counts")
    reference = attentions[resolved[0]]
    zero = torch.zeros(heads, dtype=reference.dtype, device=reference.device)
    response_rows = []
    for predictor in predictor_positions:
        history_rows = []
        for key in attention_window_indices(predictor, history):
            layer_rows = (
                [zero for _ in resolved]
                if key < 0
                else [attentions[layer][0, :, predictor, key] for layer in resolved]
            )
            history_rows.append(torch.stack(layer_rows, dim=1))
        response_rows.append(torch.stack(history_rows))
    return _on_cpu(torch.stack(response_rows)), resolved


def _intermediate_projection_features(
    torch: Any,
    hidden_states: Sequence[Any],
    lm_head: Any,
    predictor_positions: Sequence[int],
    sequence_token_ids: Sequence[int],
    *,
    layer_top_logits_layers: Sequence[int],
    token_similarity_layers: Sequence[int],
    top_k: int,
    chunk_size: int,
    include_output_ranks: bool,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    if top_k <= 0 or chunk_size <= 0:
        raise ValueError("intermediate top_k and chunk_size must be positive")
    if not predictor_positions or any(
        int(position) < 0 or int(position) >= len(sequence_token_ids) - 1
        for position in predictor_positions
    ):
        raise ValueError("intermediate projections require valid predictor positions")
    if not hasattr(lm_head, "weight"):
        raise ValueError("model lm_head must expose its token embedding weight")
    vocabulary_size = int(lm_head.weight.shape[0])
    if top_k > vocabulary_size:
        raise ValueError(f"intermediate top_k {top_k} exceeds vocabulary size {vocabulary_size}")
    if any(
        int(sequence_token_ids[position]) < 0
        or int(sequence_token_ids[position]) >= vocabulary_size
        for position in predictor_positions
        if position > 0
    ):
        raise ValueError("public output-rank token id is outside the model vocabulary")

    top_indices = resolve_tuple_layers(len(hidden_states), layer_top_logits_layers)
    similarity_indices = resolve_tuple_layers(len(hidden_states), token_similarity_layers)
    if len(similarity_indices) < 2:
        raise ValueError("token similarities require at least two layers")
    rank_indices = tuple(range(len(hidden_states) - 1)) if include_output_ranks else ()
    required_indices = tuple(sorted(set(top_indices + similarity_indices + rank_indices)))

    top_logits_chunks = []
    top_ids_chunks = []
    similarity_chunks = []
    rank_chunks = []
    for start in range(0, len(predictor_positions), chunk_size):
        positions = list(predictor_positions[start : start + chunk_size])
        rankable = [(offset, position) for offset, position in enumerate(positions) if position > 0]
        rank_positions = [position - 1 for _, position in rankable]
        rank_targets = torch.tensor(
            [sequence_token_ids[position] for _, position in rankable],
            dtype=torch.long,
            device=hidden_states[0].device,
        )
        layer_top_logits: dict[int, Any] = {}
        layer_top_ids: dict[int, Any] = {}
        similarity_embeddings: dict[int, Any] = {}
        layer_ranks: dict[int, Any] = {}

        for layer in required_indices:
            if layer in top_indices or layer in similarity_indices:
                projected = lm_head(hidden_states[layer][0, positions, :])
                if layer in top_indices:
                    sorted_logits, sorted_ids = torch.sort(
                        projected, dim=-1, descending=True
                    )
                    layer_top_logits[layer] = sorted_logits[:, :top_k].clone()
                else:
                    sorted_ids = torch.argsort(projected, dim=-1, descending=True)
                compact_ids = sorted_ids[:, :top_k].clone()
                if layer in top_indices:
                    layer_top_ids[layer] = compact_ids
                if layer in similarity_indices:
                    similarity_embeddings[layer] = lm_head.weight[compact_ids]
            if layer in rank_indices:
                ranks = torch.ones(
                    len(positions), dtype=torch.int32, device=hidden_states[0].device
                )
                if rankable:
                    public_logits = lm_head(hidden_states[layer][0, rank_positions, :])
                    public_order = torch.argsort(public_logits, dim=-1, descending=True)
                    matches = public_order == rank_targets.unsqueeze(-1)
                    rank_values = (
                        matches.to(dtype=torch.int32).argmax(dim=-1) + 1
                    ).to(dtype=torch.int32)
                    ranks[torch.tensor(
                        [offset for offset, _ in rankable],
                        dtype=torch.long,
                        device=ranks.device,
                    )] = rank_values
                layer_ranks[layer] = ranks

        top_logits_chunks.append(
            _on_cpu(
                torch.stack([layer_top_logits[layer] for layer in top_indices], dim=1),
                torch.float32,
            )
        )
        top_ids_chunks.append(
            _on_cpu(
                torch.stack([layer_top_ids[layer] for layer in top_indices], dim=1),
                torch.int32,
            )
        )
        similarities = [
            torch.cosine_similarity(
                similarity_embeddings[current], similarity_embeddings[previous], dim=-1
            )
            for previous, current in zip(similarity_indices, similarity_indices[1:])
        ]
        similarity_chunks.append(
            _on_cpu(torch.stack(similarities, dim=1), torch.float32)
        )
        if include_output_ranks:
            rank_chunks.append(
                _on_cpu(
                    torch.stack([layer_ranks[layer] for layer in rank_indices], dim=1),
                    torch.int32,
                )
            )

    tensors = {
        "layer_top_logits": torch.cat(top_logits_chunks, dim=0),
        "layer_top_token_ids": torch.cat(top_ids_chunks, dim=0),
        "token_similarities": torch.cat(similarity_chunks, dim=0),
    }
    if include_output_ranks:
        tensors["output_ranks"] = torch.cat(rank_chunks, dim=0)
    return tensors, {
        "layer_top_logits_tuple_indices": top_indices,
        "token_similarity_tuple_indices": similarity_indices,
        "output_rank_tuple_indices": rank_indices,
    }


def _lookback_features(
    torch: Any,
    attentions: Sequence[Any],
    predictor_positions: Sequence[int],
    response_start: int,
) -> dict[str, Any]:
    if response_start <= 0:
        raise ValueError("lookback features require a non-empty prompt")
    ratios = []
    prompt_masses = []
    generated_masses = []
    for predictor in predictor_positions:
        prompt_layers = []
        generated_layers = []
        for attention in attentions:
            row = attention[0, :, predictor, : predictor + 1]
            prompt_layers.append(row[:, :response_start].sum(dim=-1))
            if predictor >= response_start:
                generated_layers.append(row[:, response_start:].sum(dim=-1))
            else:
                generated_layers.append(torch.zeros_like(prompt_layers[-1]))
        prompt_mass = torch.stack(prompt_layers, dim=0)
        generated_mass = torch.stack(generated_layers, dim=0)
        if predictor > response_start:
            prompt_mean = prompt_mass / response_start
            generated_mean = generated_mass / (predictor - response_start)
            denominator = prompt_mean + generated_mean
            ratio = generated_mean / denominator
        else:
            ratio = torch.ones_like(prompt_mass)
        prompt_masses.append(prompt_mass)
        generated_masses.append(generated_mass)
        ratios.append(ratio)
    return {
        "lookback_lens": _on_cpu(torch.stack(ratios), torch.float32),
        "attention_prompt_mass": _on_cpu(torch.stack(prompt_masses), torch.float32),
        "attention_generated_mass": _on_cpu(
            torch.stack(generated_masses), torch.float32
        ),
    }


def _validate_output_shapes(config: Mapping[str, Any], outputs: Any, sequence_length: int) -> None:
    expected = config["model"]["expected_shape"]
    layers = int(expected["layers"])
    if len(outputs.hidden_states) != layers + 1 or len(outputs.attentions) != layers:
        raise ValueError("model output layer tuples differ from the configured architecture")
    expected_hidden = (1, sequence_length, int(expected["hidden_size"]))
    if any(tuple(map(int, tensor.shape)) != expected_hidden for tensor in outputs.hidden_states):
        raise ValueError("model hidden-state shapes differ from the configured architecture")
    expected_attention = (
        1,
        int(expected["attention_heads"]),
        sequence_length,
        sequence_length,
    )
    if any(tuple(map(int, tensor.shape)) != expected_attention for tensor in outputs.attentions):
        raise ValueError("model did not return configured full teacher-forcing attentions")
    if int(outputs.logits.shape[0]) != 1 or int(outputs.logits.shape[1]) != sequence_length:
        raise ValueError("model output sequence length differs from released input_ids")


def _gpu_profile(torch: Any, device: str, attentions: Sequence[Any]) -> dict[str, str]:
    torch.cuda.synchronize(device)
    dtypes = {str(tensor.dtype).replace("torch.", "") for tensor in attentions}
    if len(dtypes) != 1:
        raise ValueError("attention layers have different dtypes")
    return {
        "gpu_total_memory_bytes": str(torch.cuda.get_device_properties(device).total_memory),
        "gpu_peak_allocated_bytes": str(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": str(torch.cuda.max_memory_reserved(device)),
        "attention_dtype": next(iter(dtypes)),
    }


def _validate_compact_tensors(torch: Any, tensors: Mapping[str, Any]) -> None:
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"feature tensor {name} contains non-finite values")


def extract_features(
    config: Mapping[str, Any],
    sample: PaperSample,
    bundle: ModelBundle,
    protocol_sha256: str,
) -> FeatureSet:
    import torch

    row = sample.row
    input_ids = [int(value) for value in row["input_ids"]]
    validate_released_tokens(bundle.tokenizer, input_ids, row["input_tokens"])
    alignment = response_alignment(input_ids, row["uncertainty_labels"])
    feature_config = config["features"]
    max_tokens = int(feature_config["max_total_tokens"])
    if len(input_ids) > max_tokens:
        raise ValueError(f"sample has {len(input_ids)} tokens, exceeding limit {max_tokens}")

    model_input = torch.tensor([input_ids], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(model_input)
    torch.cuda.synchronize(bundle.device)
    torch.cuda.reset_peak_memory_stats(bundle.device)
    with torch.inference_mode():
        outputs = bundle.model(
            input_ids=model_input,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    if outputs.hidden_states is None or outputs.attentions is None:
        raise ValueError("model did not return hidden states and eager attentions")
    _validate_output_shapes(config, outputs, len(input_ids))

    predictor_positions = tuple(range(len(input_ids) - 1))
    next_token_ids = input_ids[1:]
    response_token_ids = [input_ids[position] for position in alignment.target_positions]
    tensors = _probability_features(
        torch,
        outputs.logits,
        predictor_positions,
        next_token_ids,
        int(feature_config["probability_top_k"]),
        int(feature_config.get("probability_chunk_tokens", 128)),
    )
    hidden_states, hidden_indices = _hidden_state_features(
        torch,
        outputs.hidden_states,
        predictor_positions,
        feature_config["hidden_state_layers"],
    )
    attention, attention_indices = _attention_features(
        torch,
        outputs.attentions,
        predictor_positions,
        feature_config["attention_layers"],
        int(feature_config["attention_history"]),
    )
    tensors.update(
        {
            "input_ids": torch.tensor(input_ids, dtype=torch.int32),
            "next_token_ids": torch.tensor(next_token_ids, dtype=torch.int32),
            "response_token_ids": torch.tensor(response_token_ids, dtype=torch.int32),
            "response_token_positions": torch.tensor(
                alignment.target_positions, dtype=torch.int32
            ),
            "predictor_positions": torch.tensor(
                predictor_positions, dtype=torch.int32
            ),
            "hidden_states": hidden_states,
            "attention_window": attention,
        }
    )
    families = list(map(str, feature_config["feature_families"]))
    projection_indices: dict[str, tuple[int, ...]] = {}
    if {"TS", "OR", "LR"}.issubset(families):
        lm_head = getattr(bundle.model, "lm_head", None)
        if lm_head is None:
            raise ValueError("model does not expose lm_head required by the public extractors")
        projected, projection_indices = _intermediate_projection_features(
            torch,
            outputs.hidden_states,
            lm_head,
            predictor_positions,
            input_ids,
            layer_top_logits_layers=feature_config["layer_top_logits_layers"],
            token_similarity_layers=feature_config["token_similarity_layers"],
            top_k=int(feature_config["intermediate_top_k"]),
            chunk_size=int(feature_config["intermediate_chunk_tokens"]),
            include_output_ranks=True,
        )
        tensors.update(projected)
        tensors.update(
            _lookback_features(
                torch,
                outputs.attentions,
                predictor_positions,
                alignment.response_start,
            )
        )
    _validate_compact_tensors(torch, tensors)
    gpu_profile = _gpu_profile(torch, bundle.device, outputs.attentions)
    metadata = feature_metadata(
        str(config["run_id"]),
        str(config["track"]),
        sample.sample_id,
        protocol_sha256,
        feature_config["hidden_state_layers"],
        feature_config["attention_layers"],
        int(feature_config["attention_history"]),
        int(feature_config["probability_top_k"]),
        families,
    )
    metadata.update(
        {
            "model_id": str(config["model"]["id"]),
            "model_revision": str(config["model"]["revision"]),
            "resolved_model_revision": str(
                bundle.provenance.get("resolved_revision", config["model"]["revision"])
            ),
            "source_row_sha256": sample.source_row_sha256,
            "source_revision": str(config["source"]["revision"]),
            "source_archive_sha256": str(config["source"]["archive_sha256"]),
            "reference_repository": str(feature_config["reference_repository"]),
            "reference_revision": str(feature_config["reference_revision"]),
            "decoder_layers": str(config["model"]["expected_shape"]["layers"]),
            "hidden_size": str(config["model"]["expected_shape"]["hidden_size"]),
            "attention_heads": str(
                config["model"]["expected_shape"]["attention_heads"]
            ),
            "max_total_tokens": str(feature_config["max_total_tokens"]),
            "probability_chunk_tokens": str(
                feature_config.get("probability_chunk_tokens", 128)
            ),
            "response_start": str(alignment.response_start),
            "response_tokens": str(len(response_token_ids)),
            "predictor_rows": str(len(predictor_positions)),
            "response_feature_row_start": str(alignment.response_start - 1),
            "hidden_state_tuple_indices": json.dumps(list(hidden_indices)),
            "attention_tuple_indices": json.dumps(list(attention_indices)),
            "vocabulary_size": str(outputs.logits.shape[-1]),
            "gpu_total_memory_bytes": gpu_profile["gpu_total_memory_bytes"],
            "gpu_peak_allocated_bytes": gpu_profile["gpu_peak_allocated_bytes"],
            "gpu_peak_reserved_bytes": gpu_profile["gpu_peak_reserved_bytes"],
            "attention_dtype": gpu_profile["attention_dtype"],
            "gpu_peak_scope": "teacher_forcing_through_compact_reductions",
            "tensor_dtypes": json.dumps(
                {name: str(tensor.dtype).replace("torch.", "") for name, tensor in tensors.items()},
                sort_keys=True,
            ),
        }
    )
    if projection_indices:
        metadata.update(
            {
                "layer_top_logits_layers": json.dumps(
                    list(feature_config["layer_top_logits_layers"])
                ),
                "layer_top_logits_tuple_indices": json.dumps(
                    list(projection_indices["layer_top_logits_tuple_indices"])
                ),
                "token_similarity_layers": json.dumps(
                    list(feature_config["token_similarity_layers"])
                ),
                "token_similarity_tuple_indices": json.dumps(
                    list(projection_indices["token_similarity_tuple_indices"])
                ),
                "output_rank_tuple_indices": json.dumps(
                    list(projection_indices["output_rank_tuple_indices"])
                ),
                "output_rank_layers": json.dumps(
                    list(projection_indices["output_rank_tuple_indices"])
                ),
                "output_rank_index_semantics": (
                    "public_hidden_tuple_indices_includes_embedding_omits_final_hidden"
                ),
                "lookback_layers": json.dumps(list(range(len(outputs.attentions)))),
                "lookback_index_semantics": "attention_tuple_decoder_layer_indices",
                "intermediate_top_k": str(feature_config["intermediate_top_k"]),
                "intermediate_chunk_tokens": str(
                    feature_config["intermediate_chunk_tokens"]
                ),
                "intermediate_projection": str(
                    feature_config["intermediate_projection"]
                ),
                "configuration_repository": str(
                    feature_config["configuration_repository"]
                ),
                "configuration_revision": str(
                    feature_config["configuration_revision"]
                ),
                "configuration_path": str(feature_config["configuration_path"]),
                "uhead_configuration_path": str(
                    feature_config["uhead_configuration_path"]
                ),
                "lookback_configuration_path": str(
                    feature_config["lookback_configuration_path"]
                ),
                "output_rank_behavior": str(feature_config["output_rank_behavior"]),
                "output_rank_alignment": str(
                    feature_config["output_rank_alignment"]
                ),
                "lookback_behavior": str(feature_config["lookback_behavior"]),
                "layer_top_axis_order": "predictor,layer,rank",
                "token_similarity_axis_order": "predictor,adjacent_layer_pair,rank",
                "output_rank_axis_order": "predictor,public_hidden_tuple_index",
                "lookback_axis_order": "predictor,decoder_layer,head",
                "attention_mass_axis_order": "predictor,decoder_layer,head",
            }
        )
    return FeatureSet(tensors, metadata)
