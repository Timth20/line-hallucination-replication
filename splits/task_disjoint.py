"""Task-disjoint split (Section III of the paper): every record of a task is in one of train, validation and test.

    python splits/task_disjoint.py data/splits/task_disjoint_<model>.csv     # checks the split column of a file

A task is assigned by a seeded hash of its key (Defects4J projects by hash order), the same for every model.
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys

SEED = 20260907
DEFECTS4J_PROJECTS = ("Chart", "Cli", "Closure", "Codec", "Collections", "Compress", "Csv", "Gson", "JacksonCore",
                      "JacksonDatabind", "JacksonXml", "Jsoup", "JxPath", "Lang", "Math", "Mockito", "Time")
NAMES = {"train": "train", "val": "validation", "test": "test"}


def group_id(dataset, task):
    if dataset not in ("mbpp", "humaneval", "humaneval_java", "defects4j") or not isinstance(task, str) or not task:
        raise ValueError("unknown dataset or missing task key")
    if dataset == "defects4j":
        if not re.fullmatch(r"[A-Za-z]+_\d+", task):
            raise ValueError("invalid repair task key")
        return "defects4j:" + task.split("_")[0]
    if not task.isdecimal():
        raise ValueError("invalid problem key")
    family = "humaneval_family" if dataset.startswith("humaneval") else dataset
    return family + ":" + task


def split_for_group(group, seed=SEED):
    if group.startswith("defects4j:"):
        groups = sorted(("defects4j:" + p for p in DEFECTS4J_PROJECTS),
                        key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest())
        if group not in groups:
            raise ValueError("unknown Defects4J project")
        rank = groups.index(group)
        return "train" if rank < 11 else "val" if rank < 14 else "test"
    bucket = int(hashlib.sha256(f"{seed}|{group}".encode()).hexdigest()[:16], 16) % 10000
    return "train" if bucket < 6400 else "val" if bucket < 8000 else "test"


def split_of(dataset, task):
    return NAMES[split_for_group(group_id(dataset, task))]


def main(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    wrong = [r["sample_id"] for r in rows if split_of(r["dataset"], r["task_id"]) != r["split"]]
    print(f"{len(rows)} records, {len(wrong)} not in the split of their task")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
