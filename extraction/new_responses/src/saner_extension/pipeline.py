from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .artifacts import atomic_json, checked_json, read_checkpoint, sample_paths, verify_sample, writer_lock, write_checkpoint, write_sample
from .config import OUTPUT_ROOT, digest
from .features import extract_features
from .generation import generate, reference_record
from .labels import annotate_response, select_reference
from .runtime import check_runtime, code_sha256, compact_profile, gpu_gate, load_tokenizer, tokenizer_provenance
from .sources import load_tasks, split_for_group


def run_root(output_dir, config):
    root = Path(output_dir).expanduser().resolve()
    if root.name != OUTPUT_ROOT or config["runtime"]["output_root_name"] != OUTPUT_ROOT:
        raise ValueError("use a new outputs_new directory; outputs of other settings cannot be resumed")
    return root/config["model_key"]/config["dataset"]


def prepare_tasks(tasks, tokenizer, config):
    result = []
    for task in tasks:
        messages = [{"role": "user", "content": task["question"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        references = [len(tokenizer.encode(text, add_special_tokens=False)) for text in task["references"]]
        maximum = len(prompt)+max(config["generation"]["max_new_tokens"], max(references))
        if not prompt or maximum > config["features"]["max_total_tokens"]:
            raise ValueError(f"task {task['task_id']} exceeds the token ceiling ({maximum}); no truncation is allowed")
        result.append({"task": task, "prompt_ids": prompt, "prompt_tokens": tokenizer.convert_ids_to_tokens(prompt),
                       "templated_question": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                       "reference_token_lengths": references, "max_sequence_tokens": maximum})
    return sorted(result, key=lambda row: (-row["max_sequence_tokens"], row["task"]["task_uid"]))


def validate_manifest(rows, config):
    if len(rows) != config["source"]["expected_tasks"]:
        raise ValueError("manifest task count differs from config")
    seen, groups, task_ids = set(), {}, set()
    for row in rows:
        task = row["task"]
        uid, group = task["task_uid"], task["group_id"]
        original = {k: v for k,v in task.items() if k != "source_task_sha256"}
        if digest(original) != task["source_task_sha256"] or task["track"] != "extension" or task["dataset"] != config["dataset"]:
            raise ValueError("source task identity/checksum mismatch")
        if uid in seen or task["task_id"] in task_ids or not group:
            raise ValueError("duplicate task id or missing group key")
        seen.add(uid)
        task_ids.add(task["task_id"])
        if task["split"] != split_for_group(group, config["split"]["seed"]) or groups.setdefault(group, task["split"]) != task["split"]:
            raise ValueError("train/validation/test group leakage or split drift")
        if (len(row["prompt_ids"]) != len(row["prompt_tokens"]) or not row["prompt_ids"] or
            len(row["reference_token_lengths"]) != len(task["references"]) or
            row["max_sequence_tokens"] != len(row["prompt_ids"])+max(config["generation"]["max_new_tokens"], max(row["reference_token_lengths"])) or
            row["max_sequence_tokens"] > config["features"]["max_total_tokens"]):
            raise ValueError("prepared token budget mismatch")
    if rows != sorted(rows, key=lambda r: (-r["max_sequence_tokens"],r["task"]["task_uid"])):
        raise ValueError("manifest is not in frozen longest-first order")


def prepare(config, output_dir, cache_dir):
    root = run_root(output_dir, config)
    with writer_lock(root):
        if (root/"metadata/protocol.json").exists():
            _, rows = load_prepared(root, config)
            return {"status": "prepared", "tasks": len(rows), "noop": True}
        tasks, source = load_tasks(config, cache_dir)
        tokenizer = load_tokenizer(config, cache_dir)
        rows = prepare_tasks(tasks, tokenizer, config)
        validate_manifest(rows, config)
        source.update(track="extension", run_id=config["run_id"])
        protocol = {"schema": "extension_protocol_v1", "track": "extension", "config": config,
                    "code_sha256": code_sha256(), "manifest_sha256": digest(rows), "source": source,
                    "provenance": json.loads((Path(__file__).parent/"provenance.json").read_text(encoding="utf-8")),
                    "tokenizer": tokenizer_provenance(tokenizer, config),
                    "generation_count": len(rows), "feature_sample_count": 2*len(rows),
                    "split_tasks": dict(Counter(r["task"]["split"] for r in rows)),
                    "split_groups": {split: len({r["task"]["group_id"] for r in rows if r["task"]["split"] == split})
                                     for split in ("train", "val", "test")},
                    "smoke": {"sample_start": 0, "max_samples": 1, "task_uid": rows[0]["task"]["task_uid"],
                              "max_sequence_tokens": rows[0]["max_sequence_tokens"]}}
        # The protocol file is written last and marks a finished prepare step.
        write_checkpoint(root/"manifests/tasks.json", rows)
        write_checkpoint(root/"metadata/protocol.json", protocol)
        return {"status": "prepared", "tasks": len(rows), "feature_samples": 2*len(rows),
                "max_sequence_tokens": rows[0]["max_sequence_tokens"], "split_tasks": protocol["split_tasks"]}


def load_prepared(root, config):
    protocol = checked_json(root/"metadata/protocol.json")
    rows = checked_json(root/"manifests/tasks.json")
    if (protocol.get("schema") != "extension_protocol_v1" or protocol.get("track") != "extension" or
        protocol["config"] != config or protocol["code_sha256"] != code_sha256() or protocol["manifest_sha256"] != digest(rows)):
        raise ValueError("prepared protocol/code/manifest drift; use a new versioned location")
    validate_manifest(rows, config)
    if protocol["generation_count"] != len(rows) or protocol["feature_sample_count"] != 2*len(rows):
        raise ValueError("protocol support count mismatch")
    return protocol, rows


def _check_record(record, row, config, protocol_sha):
    task = row["task"]
    if any(record.get(key) != task[key] for key in ("task_uid", "task_id", "group_id", "split", "source_task_sha256", "question")):
        raise ValueError("generation/task join mismatch")
    if (record["protocol_sha256"] != protocol_sha or record["run_id"] != config["run_id"] or
        record["track"] != "extension" or record["model_id"] != config["model"]["id"] or
        record["model_revision"] != config["model"]["revision"] or record["precision_class"] != config["precision_class"] or
        record["input_ids"][:record["response_start"]] != row["prompt_ids"] or
        record["input_tokens"][:record["response_start"]] != row["prompt_tokens"] or
        record["generation_parameters"] != config["generation"]):
        raise ValueError("generation protocol/prompt mismatch")
    if record["view"] == "model_output_only":
        count = len(record["generated_ids"])
        if not 0 < count <= config["generation"]["max_new_tokens"] or record["generated_ids"] != record["input_ids"][record["response_start"]:]:
            raise ValueError("generation id/length mismatch")
        reason = "eos" if record["generated_ids"][-1] in record["eos_token_ids"] else "length"
        if record["finish_reason"] != reason or reason == "length" and count != config["generation"]["max_new_tokens"]:
            raise ValueError("generation termination mismatch")


def inspect_run(root, config, protocol, rows):
    """Deep verification also runs before resume; corrupted completed samples never get replaced."""
    protocol_sha = digest(protocol)
    completed, markers, profiles, reasons = set(), {}, [], Counter()
    expected_ids = {r["task"]["task_uid"]+suffix for r in rows for suffix in ("-model_output", "-reference_control")}
    actual_ids = {p.name.removesuffix(".done.json") for p in root.glob("*.done.json")}
    if not actual_ids <= expected_ids:
        raise ValueError("unexpected completed sample ids")
    generation_ids = {p.stem for p in (root/"state/generations").glob("*.json")}
    if not generation_ids <= {r["task"]["task_uid"] for r in rows}:
        raise ValueError("unexpected generation checkpoints")
    for row in rows:
        uid = row["task"]["task_uid"]
        checkpoint = root/"state/generations"/(uid+".json")
        generation = read_checkpoint(checkpoint, protocol_sha, uid) if checkpoint.exists() else None
        if generation:
            _check_record(generation, row, config, protocol_sha)
            reasons[generation["finish_reason"]] += 1
        for suffix in ("-model_output", "-reference_control"):
            sample_id = uid+suffix
            result = verify_sample(root, sample_id, config, protocol_sha)
            if result is None:
                continue
            record, labels, meta = result
            _check_record(record, row, config, protocol_sha)
            if generation is None:
                raise ValueError("completed feature sample has no generation checkpoint")
            import hashlib
            selected = select_reference(generation["raw_reply"], row["task"]["references"])
            if labels["reference_sha256"] != hashlib.sha256(selected.encode()).hexdigest():
                raise ValueError("selected reference provenance mismatch")
            if suffix == "-model_output":
                if record["generation_record_sha256"] != digest(generation):
                    raise ValueError("output does not bind the generation checkpoint")
            elif record.get("parent_generation_sha256") != digest(generation):
                raise ValueError("reference control parent mismatch")
            gpu_gate(meta, config)
            completed.add(sample_id)
            markers[sample_id] = digest(checked_json(sample_paths(root,sample_id)["done"]))
            profiles.append(compact_profile(meta))
    summary = {"schema": "extension_state_v1", "track": "extension", "run_id": config["run_id"],
               "protocol_sha256": protocol_sha, "expected_generations": len(rows),
               "saved_generations": len(generation_ids), "completed_samples": len(completed),
               "expected_samples": 2*len(rows), "finish_reasons": dict(reasons),
               "sample_markers_sha256": digest(markers)}
    if profiles:
        summary["gpu_peak_allocated_bytes"] = max(int(p["gpu_peak_allocated_bytes"]) for p in profiles)
        summary["gpu_peak_reserved_bytes"] = max(int(p["gpu_peak_reserved_bytes"]) for p in profiles)
        summary["gpu_total_memory_bytes"] = min(int(p["gpu_total_memory_bytes"]) for p in profiles)
        summary["attention_dtype"] = "float16"
    if completed:
        environment = checked_json(root/"metadata/environment.json")
        summary["environment_sha256"] = digest(environment)
        summary["model_metadata_sha256"] = digest(checked_json(root/"metadata/model.json"))
    smoke_path = root/"state/SMOKE.json"
    if smoke_path.exists():
        smoke = checked_json(smoke_path)
        expected_smoke = {"schema": "extension_smoke_v1", "track": "extension", "protocol_sha256": protocol_sha,
                          "task_uid": rows[0]["task"]["task_uid"]}
        if smoke != expected_smoke or not all(smoke["task_uid"]+suffix in completed for suffix in ("-model_output", "-reference_control")):
            raise ValueError("invalid smoke marker")
        summary["smoke_sha256"] = digest(smoke)
    return summary, completed


def status(config, output_dir):
    root = run_root(output_dir,config)
    protocol, rows = load_prepared(root,config)
    summary, _ = inspect_run(root,config,protocol,rows)
    completion = root/"state/COMPLETED.json"
    if completion.exists():
        if summary != checked_json(completion) or summary["completed_samples"] != summary["expected_samples"]:
            raise ValueError("run completion inventory/checksum mismatch")
        summary = {**summary, "status": "complete"}
    else:
        summary = {**summary, "status": "incomplete"}
    return summary


def run(config, output_dir, cache_dir, device="cuda:0", sample_start=0, max_samples=None):
    from ._replay.models import load_model

    root = run_root(output_dir,config)
    with writer_lock(root):
        protocol, rows = load_prepared(root,config)
        protocol_sha = digest(protocol)
        summary, completed = inspect_run(root,config,protocol,rows)
        complete_path = root/"state/COMPLETED.json"
        if complete_path.exists():
            if checked_json(complete_path) != summary:
                raise ValueError("run completion drift")
            return {**summary, "status": "complete", "noop": True}
        if sample_start < 0 or sample_start >= len(rows) or max_samples is not None and max_samples <= 0:
            raise ValueError("invalid sample range")
        smoke_path = root/"state/SMOKE.json"
        is_smoke = sample_start == 0 and max_samples == 1
        smoke = {"schema": "extension_smoke_v1", "track": "extension", "protocol_sha256": protocol_sha,
                 "task_uid": rows[0]["task"]["task_uid"]}
        if smoke_path.exists():
            if checked_json(smoke_path) != smoke or not all(smoke["task_uid"]+suffix in completed for suffix in ("-model_output", "-reference_control")):
                raise ValueError("invalid smoke marker")
        elif not is_smoke:
            raise ValueError("first run must use --sample-start 0 --max-samples 1")
        selected = rows[sample_start:] if max_samples is None else rows[sample_start:sample_start+max_samples]
        pending = [r for r in selected if any(r["task"]["task_uid"]+s not in completed for s in ("-model_output", "-reference_control"))]
        if pending:
            environment = check_runtime(config,device)
            write_checkpoint(root/"metadata/environment.json",environment)
            bundle = load_model(config,Path(cache_dir),device)
            if tokenizer_provenance(bundle.tokenizer,config) != protocol["tokenizer"]:
                raise ValueError("loaded tokenizer differs from prepared tokenizer")
            write_checkpoint(root/"metadata/model.json",dict(bundle.provenance))
            try:
                for row in pending:
                    task, prompt = row["task"], row["prompt_ids"]
                    path = root/"state/generations"/(task["task_uid"]+".json")
                    if path.exists():
                        record = read_checkpoint(path,protocol_sha,task["task_uid"])
                    else:
                        record = generate(task,prompt,bundle,config,protocol_sha)
                        # Save the generation before extracting features.
                        write_checkpoint(path,record)
                    _check_record(record,row,config,protocol_sha)
                    if record["generation_gpu_profile"]:
                        gpu_gate(record["generation_gpu_profile"],config)
                    reference = select_reference(record["raw_reply"],task["references"])
                    for view in (record,reference_record(record,reference,bundle.tokenizer)):
                        if view["sample_id"] in completed:
                            continue
                        labels = annotate_response(bundle.tokenizer,view["generated_ids"],reference)
                        features = extract_features(config,view,bundle,protocol_sha)
                        gpu_gate(features.metadata,config)
                        write_sample(root,view,labels,features,config)
                        completed.add(view["sample_id"])
                        atomic_json(root/"state/progress.json", {"schema": "extension_progress_v1", "track": "extension",
                            "protocol_sha256": protocol_sha, "completed_samples": len(completed),
                            "expected_samples": 2*len(rows), "last_sample_id": view["sample_id"],
                            "gpu_profile": compact_profile(features.metadata), "status": "running"})
                        print(f"{len(completed)}/{2*len(rows)} {view['sample_id']}", flush=True)
            except BaseException as error:
                atomic_json(root/"state/interruption.json", {"schema": "extension_interruption_v1", "track": "extension",
                    "protocol_sha256": protocol_sha, "exception_type": type(error).__name__,
                    "completed_samples": len(completed), "resume": "repeat_same_run_command"})
                raise
        summary, completed = inspect_run(root,config,protocol,rows)
        if is_smoke:
            write_checkpoint(smoke_path,smoke)
            summary["smoke_sha256"] = digest(smoke)
        if len(completed) == 2*len(rows):
            write_checkpoint(complete_path,summary)
        atomic_json(root/"state/progress.json",summary)
        return {**summary, "status": "complete" if complete_path.exists() else "incomplete", "noop": not pending}
