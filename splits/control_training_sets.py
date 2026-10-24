"""RQ2 control: two equal-size training sets on the original split, with and without records of test tasks.

    python splits/control_training_sets.py       # writes data/splits/control_training_sets.json

Records of test tasks (tasks with a test record) replace a stratified random draw of the other training records;
the draw closest in line count to the set without them is kept.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["deepseek-coder-1.3b", "deepseek-coder-6.7b", "codellama-7b", "deepseek-coder-33b"]
SEED, DRAWS = 20261018, 1000


def task_key(dataset, task):
    if dataset == "defects4j":
        if not re.fullmatch(r"[A-Za-z]+_\d+", task):
            raise ValueError(f"invalid repair task key {task}")
        return "defects4j:" + task
    if dataset not in ("mbpp", "humaneval", "humaneval_java") or not task.isdecimal():
        raise ValueError(f"unknown dataset or task key {dataset} {task}")
    return ("humaneval_family" if dataset.startswith("humaneval") else dataset) + ":" + task


def control_sets(rows):
    roles = {role: [r["sample_id"] for r in rows if r["split"] == role] for role in ("train", "validation", "test")}
    meta = {r["sample_id"]: r for r in rows}
    lines = lambda xs: sum(int(meta[s]["lines"]) for s in xs)
    tk = {s: task_key(meta[s]["dataset"], meta[s]["task_id"]) for s in meta}
    stratum = {s: (meta[s]["dataset"], meta[s]["kind"]) for s in meta}
    test_tasks = {tk[s] for s in roles["test"]}
    leaky = [s for s in roles["train"] if tk[s] in test_tasks]
    clean = [s for s in roles["train"] if tk[s] not in test_tasks]
    val = [s for s in roles["validation"] if tk[s] not in test_tasks]
    rng = random.Random(SEED)
    need = Counter(stratum[s] for s in leaky)
    by_stratum = defaultdict(list)
    for s in sorted(clean):
        by_stratum[stratum[s]].append(s)
    for st in need:
        if len(by_stratum[st]) < need[st]:
            raise ValueError(f"stratum {st} has fewer clean records than records of test tasks")
    lines_b, lines_l = lines(clean), lines(leaky)
    best = None
    for draw in range(DRAWS):
        cand = set()
        for st in sorted(need):
            cand |= set(rng.sample(by_stratum[st], need[st]))
        gap = abs(lines_l + lines_b - lines(cand) - lines_b)
        if best is None or gap < best[0]:
            best = (gap, draw, cand)
    dropped = best[2]
    without = sorted(clean)
    with_ = sorted(leaky + [s for s in clean if s not in dropped])
    assert Counter(stratum[s] for s in with_) == Counter(stratum[s] for s in without)
    assert not any(tk[s] in test_tasks for s in without + val) and set(leaky) <= set(with_)
    return {"train_with_test_tasks": with_, "train_without_test_tasks": without, "validation": sorted(val),
            "test": sorted(roles["test"])}


def main():
    out = {}
    for m in MODELS:
        with open(ROOT / f"data/splits/original_split_{m}.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        out[m] = control_sets(rows)
        print(m, {k: len(v) for k, v in out[m].items()})
    target = ROOT / "data/splits/control_training_sets.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("wrote", target.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
