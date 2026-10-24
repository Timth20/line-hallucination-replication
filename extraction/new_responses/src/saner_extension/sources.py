from __future__ import annotations

import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import digest

PAPER_VIEWS = ("line_diff_dsc_instruct_dataset", "line_diff_dsc6_instruct_dataset",
               "line_diff_dsc33_instruct_dataset", "line_diff_cl7_instruct_dataset", "line_diff_l38b_instruct_dataset")
DEFECTS4J_PROJECTS = ("Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
                     "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(url, sha256, destination):
    destination = Path(destination)
    if destination.exists():
        if file_sha256(destination) != sha256:
            raise ValueError(f"cached source checksum mismatch: {destination.name}")
        return destination
    if not url.startswith("https://"):
        raise ValueError("only HTTPS pinned public sources are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name+".", suffix=".part", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as target, urllib.request.urlopen(url, timeout=120) as source:
            while block := source.read(1024 * 1024):
                target.write(block)
            target.flush()
            os.fsync(target.fileno())
        if file_sha256(temporary) != sha256:
            raise ValueError("downloaded source checksum mismatch")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def split_for_group(group_id, seed):
    if group_id.startswith("defects4j:"):
        groups = sorted(("defects4j:"+p for p in DEFECTS4J_PROJECTS),
                        key=lambda group: hashlib.sha256(f"{seed}|{group}".encode()).hexdigest())
        if group_id not in groups:
            raise ValueError("unknown Defects4J project outside the frozen source support")
        rank = groups.index(group_id)
        # Largest-remainder allocation over the 17 projects.
        return "train" if rank < 11 else "val" if rank < 14 else "test"
    bucket = int(hashlib.sha256(f"{seed}|{group_id}".encode()).hexdigest()[:16], 16) % 10000
    return "train" if bucket < 6400 else "val" if bucket < 8000 else "test"


def _finish(task):
    task["split"] = split_for_group(task["group_id"], 20260907)
    task["task_uid"] = digest([task["dataset"], task["task_id"], task["question"]])[:24]
    task["track"] = "extension"
    task["source_task_sha256"] = digest(task)
    return task


def paper_tasks_from_rows(rows, dataset):
    tasks = {}
    for row_index, row in enumerate(rows):
        if row["dataset"] != dataset:
            continue
        task_id = str(row["task_id"])
        if not task_id or not isinstance(row["question"], str) or not row["question"] or not row["answer"]:
            raise ValueError("incomplete published task")
        file_path = row["meta"].get("file_path")
        if file_path and (PurePosixPath(file_path).is_absolute() or PureWindowsPath(file_path).is_absolute() or ".." in PurePosixPath(file_path).parts):
            raise ValueError("unsafe published project-relative file path")
        group_family = "humaneval_family" if dataset in ("humaneval", "humaneval_java") else dataset
        group_id = group_family+":"+(task_id.split("_")[0] if dataset == "defects4j" else task_id)
        task = tasks.setdefault(task_id, {"dataset": dataset, "task_id": task_id, "group_id": group_id,
            "question": row["question"], "meta": row["meta"], "references": [], "source_occurrences": [],
            "scenario": "code-generation" if dataset in ("mbpp", "humaneval") else "code-repair",
            "source_support": "union_of_five_published_main_views_not_full_upstream_benchmark"})
        if task["question"] != row["question"] or task["meta"] != row["meta"]:
            raise ValueError(f"published task has conflicting question/metadata: {dataset}/{task_id}")
        task["references"].append(row["answer"])
        task["source_occurrences"].append({"source_view": row.get("_source_view", row.get("model", "unknown")),
            "source_split": row.get("_source_split"), "row_index": row.get("_row_index", row_index),
            "source_idx": row.get("idx"), "reference_sha256": hashlib.sha256(row["answer"].encode()).hexdigest()})
    result = []
    for task_id, task in sorted(tasks.items()):
        task["references"] = sorted(set(task["references"]), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        task["source_occurrences"] = sorted(task["source_occurrences"], key=digest)
        result.append(_finish(task))
    return result


def bcb_tasks_from_rows(rows):
    tasks, seen = [], set()
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in seen:
            raise ValueError("duplicate BigCodeBench task_id")
        seen.add(task_id)
        question = row["instruct_prompt"]
        if not question or not row["test"] or not row["canonical_solution"]:
            raise ValueError("incomplete BigCodeBench task")
        reference = "```python\n" + row["code_prompt"] + row["canonical_solution"].rstrip("\n") + "\n```\n"
        tasks.append(_finish({"dataset": "bigcodebench", "task_id": task_id,
            "group_id": "bigcodebench:" + digest(" ".join(question.split()))[:24],
            "question": question, "references": [reference], "scenario": "code-generation",
            "source_support": "bigcodebench_full_instruct_v0.1.4",
            "meta": {k: row.get(k) for k in ("test", "entry_point", "libs", "code_prompt", "complete_prompt", "doc_struct")},
            "source_occurrences": [{"source_view": "v0.1.4", "task_id": task_id}]}))
    return sorted(tasks, key=lambda t: t["task_id"])


def load_tasks(config, cache_dir):
    source = config["source"]
    archive = download(source["archive_url"], source["archive_sha256"], Path(cache_dir)/"sources"/(source["archive_sha256"] + (".zip" if source["kind"] == "paper_task_union" else ".parquet")))
    if source["kind"] == "paper_task_union":
        import pyarrow.ipc as ipc
        rows = []
        with zipfile.ZipFile(archive) as handle:
            for view in PAPER_VIEWS:
                names = sorted(n for n in handle.namelist() if n.startswith("Datasets/data/"+view+"/") and n.endswith(".arrow"))
                if not names:
                    raise ValueError(f"missing pinned paper view: {view}")
                for name in names:
                    table = ipc.open_stream(io.BytesIO(handle.read(name))).read_all()
                    table = table.select(["dataset", "task_id", "question", "answer", "meta", "idx", "model"])
                    for index, row in enumerate(table.to_pylist()):
                        rows.append({**row, "_source_view": view, "_source_split": name.split("/")[-2], "_row_index": index})
        tasks = paper_tasks_from_rows(rows, config["dataset"])
    else:
        import pyarrow.parquet as parquet
        tasks = bcb_tasks_from_rows(parquet.read_table(archive).to_pylist())
    if len(tasks) != int(source["expected_tasks"]):
        raise ValueError(f"expected {source['expected_tasks']} tasks, found {len(tasks)}")
    return tasks, {"schema": "extension_source_v1", "repository": source["repository"],
        "revision": source["revision"], "archive_url": source["archive_url"],
        "archive_sha256": file_sha256(archive), "logical_cache_key": archive.name,
        "task_count": len(tasks), "task_manifest_sha256": digest(tasks)}
