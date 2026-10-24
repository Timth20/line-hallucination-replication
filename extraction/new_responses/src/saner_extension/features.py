from __future__ import annotations

import json

from ._replay.features import extract_features as replay_extract, resolve_tuple_layers
from ._replay.types import FeatureSet, PaperSample


def extract_features(config, record, bundle, protocol_sha256):
    # The zeros only mark the response start; no label enters the model.
    boundary = record["response_start"]
    row = {"input_ids": record["input_ids"], "input_tokens": record["input_tokens"],
           "uncertainty_labels": [-100]*boundary+[0]*(len(record["input_ids"])-boundary)}
    sample = PaperSample(record["sample_id"], row, (), record["source_task_sha256"])
    features = replay_extract(config, sample, bundle, protocol_sha256)
    metadata = dict(features.metadata)
    metadata.pop("source_row_sha256")
    metadata.update(schema="saner_extension_features_v1", task_uid=record["task_uid"],
                    source_task_sha256=record["source_task_sha256"], view=record["view"],
                    precision_class=config["precision_class"],
                    feature_formulas="extraction/original_outputs",
                    feature_adaptation=config["features"].get("adaptation", "same_model_selectors"))
    validate_tensors(features.tensors, metadata, record, config)
    return FeatureSet(features.tensors, metadata)


def validate_tensors(tensors, metadata, record, config):
    import torch

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(metadata.get("schema") == "saner_extension_features_v1", "feature schema mismatch")
    require(metadata.get("track") == record.get("track") == "extension", "feature track mismatch")
    for key in ("sample_id", "task_uid", "protocol_sha256", "source_task_sha256", "view", "precision_class"):
        require(metadata.get(key) == record[key], f"feature identity mismatch: {key}")
    require(metadata["precision_class"] == config["precision_class"], "feature precision class mismatch")
    require(metadata.get("attention_dtype") == "float16", "attention must be float16")
    require(metadata.get("predictor_scope") == "full_sequence", "predictor scope mismatch")
    ids, boundary = record["input_ids"], record["response_start"]
    T, P, R = len(ids), len(ids)-1, len(ids)-boundary
    F, M = config["features"], config["model"]["expected_shape"]
    L, H, A = M["layers"], M["hidden_size"], M["attention_heads"]
    V = int(metadata["vocabulary_size"])
    require(0 < boundary < T <= F["max_total_tokens"], "response/token ceiling mismatch")
    require(record["generated_ids"] == ids[boundary:] and record["response_end_exclusive"] == T
            and record["response_feature_row_start"] == boundary-1, "response boundary mismatch")
    require(all(type(i) is int and 0 <= i < V for i in ids), "input vocabulary range mismatch")
    for key, expected in (("response_start", boundary), ("predictor_rows", P),
                          ("response_feature_row_start", boundary-1)):
        require(int(metadata[key]) == expected, f"feature metadata {key} mismatch")
    for name, expected in (("hidden_state_tuple_indices", resolve_tuple_layers(L+1, F["hidden_state_layers"])),
                            ("attention_tuple_indices", resolve_tuple_layers(L, F["attention_layers"]))):
        require(json.loads(metadata[name]) == list(expected), f"feature layer indices mismatch: {name}")
    shapes = {"input_ids": (T,), "next_token_ids": (P,), "response_token_ids": (R,),
              "response_token_positions": (R,), "predictor_positions": (P,),
              "topk_probabilities": (P, F["probability_top_k"]), "topk_token_ids": (P, F["probability_top_k"]),
              "hidden_states": (P, len(F["hidden_state_layers"]), H),
              "attention_window": (P, F["attention_history"], A, len(F["attention_layers"]))}
    if "TS" in F["feature_families"]:
        shapes.update(layer_top_logits=(P, len(F["layer_top_logits_layers"]), F["intermediate_top_k"]),
                      layer_top_token_ids=(P, len(F["layer_top_logits_layers"]), F["intermediate_top_k"]),
                      token_similarities=(P, len(F["token_similarity_layers"])-1, F["intermediate_top_k"]),
                      output_ranks=(P,L), lookback_lens=(P,L,A), attention_prompt_mass=(P,L,A),
                      attention_generated_mass=(P,L,A))
        for key, expected in (("layer_top_logits_tuple_indices", resolve_tuple_layers(L+1,F["layer_top_logits_layers"])),
                               ("token_similarity_tuple_indices", resolve_tuple_layers(L+1,F["token_similarity_layers"])),
                               ("output_rank_tuple_indices", range(L)), ("lookback_layers", range(L))):
            require(json.loads(metadata[key]) == list(expected), f"layer alignment mismatch: {key}")
    require(set(tensors) == set(shapes), "feature inventory mismatch (labels are forbidden)")
    integers = {"input_ids", "next_token_ids", "response_token_ids", "response_token_positions", "predictor_positions",
                "topk_token_ids", "layer_top_token_ids", "output_ranks"}
    for name, tensor in tensors.items():
        require(tuple(tensor.shape) == shapes[name], f"feature shape mismatch: {name}")
        allowed = ({torch.int32} if name in integers else {torch.float16} if name in {"hidden_states", "attention_window"}
                   else {torch.float16, torch.float32} if name == "topk_probabilities" else {torch.float32})
        require(tensor.dtype in allowed, f"feature dtype mismatch: {name}")
        require(bool(torch.isfinite(tensor).all()), f"non-finite feature: {name}")
    for key, expected in (("input_ids", ids), ("next_token_ids", ids[1:]), ("response_token_ids", ids[boundary:]),
                          ("response_token_positions", list(range(boundary,T))), ("predictor_positions", list(range(P)))):
        require(tensors[key].tolist() == expected, f"token alignment mismatch: {key}")
    for name in ("topk_token_ids", "layer_top_token_ids"):
        if name in tensors:
            values = tensors[name]
            require(bool(((values >= 0) & (values < V)).all()), f"vocabulary range: {name}")
            require(not bool((values.sort(dim=-1).values.diff(dim=-1) == 0).any()), f"duplicate token candidates: {name}")
    prob = tensors["topk_probabilities"].float()
    require(bool(((prob >= 0) & (prob <= 1)).all()) and bool((prob.sum(-1) <= 1.003).all()), "invalid probabilities")
    require(bool((prob.diff(dim=-1) <= 0).all()), "TP candidates are not descending")
    aw = tensors["attention_window"].float()
    require(bool(((aw >= 0) & (aw <= 1.002)).all()) and bool((aw.sum(1) <= 1.003).all()), "invalid attention window")
    for history in range(1, min(F["attention_history"],P+1)):
        require(not bool(aw[:history,history].any()), "attention padding is nonzero")
    if "TS" in F["feature_families"]:
        require(bool((tensors["layer_top_logits"].diff(dim=-1) <= 0).all()), "layer logits are not descending")
        require(bool((tensors["token_similarities"].abs() <= 1.003).all()), "invalid cosine similarity")
        ranks = tensors["output_ranks"]
        require(bool(((ranks >= 1) & (ranks <= V)).all()) and bool((ranks[0] == 1).all()), "invalid public output rank")
        pm, gm, lr = (tensors[n] for n in ("attention_prompt_mass", "attention_generated_mass", "lookback_lens"))
        require(bool((pm >= 0).all()) and bool((gm >= 0).all()) and
                torch.allclose(pm+gm, torch.ones_like(pm), atol=.003, rtol=0), "attention mass mismatch")
        require(not bool(gm[:boundary].any()), "generated mass leaked before response")
        require(bool((lr[:boundary+1] == 1).all()), "public lookback prefix mismatch")
        if P > boundary+1:
            denom = torch.arange(1,P-boundary, dtype=torch.float32).reshape(-1,1,1)
            mean = gm[boundary+1:]/denom
            expected = mean/(mean+pm[boundary+1:]/boundary)
            require(torch.allclose(lr[boundary+1:], expected, atol=.003, rtol=.01), "lookback formula mismatch")
    if "tensor_dtypes" in metadata:
        require(json.loads(metadata["tensor_dtypes"]) == {k: str(t.dtype).replace("torch.", "") for k,t in tensors.items()},
                "tensor dtype manifest mismatch")
