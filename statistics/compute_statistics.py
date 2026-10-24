"""Compute every statistic reported in the paper from the line-level test predictions in data/.

    python statistics/compute_statistics.py            # writes statistics/results.json

Follows Section III of the paper: trapezoidal PR-AUC, 1,000 resamples of the test records, paired percentile CIs,
and the 95% prediction interval of one training run for RQ1.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ITERATIONS, SEED = 1000, 0
T_975_2 = 4.302652729911275          # t(0.975, K - 1) for K = 3 training runs
MAIN = "tp_aw_hs"
NINE = ["hs", "aw", "tp", "hs_aw", "hs_tp", "tp_aw", "tp_aw_hs", "ts_or_tp_hs", "lr"]
ORIGINAL_MODELS = ["deepseek-coder-1.3b", "deepseek-coder-6.7b", "codellama-7b", "deepseek-coder-33b"]
ALL_MODELS = ORIGINAL_MODELS + ["qwen2.5-coder-7b"]
# Table 2 of the original paper: PR-AUC (mean, sd)
PUBLISHED = {
    "deepseek-coder-1.3b": dict(hs=(0.697, 0.065), aw=(0.694, 0.091), tp=(0.292, 0.029), hs_aw=(0.651, 0.098),
                                hs_tp=(0.684, 0.067), tp_aw=(0.630, 0.093), tp_aw_hs=(0.719, 0.085),
                                ts_or_tp_hs=(0.673, 0.066), lr=(0.665, 0.065)),
    "deepseek-coder-6.7b": dict(hs=(0.572, 0.067), aw=(0.493, 0.083), tp=(0.213, 0.029), hs_aw=(0.594, 0.071),
                                hs_tp=(0.555, 0.071), tp_aw=(0.535, 0.081), tp_aw_hs=(0.606, 0.068),
                                ts_or_tp_hs=(0.617, 0.058), lr=(0.551, 0.066)),
    "codellama-7b": dict(hs=(0.756, 0.051), aw=(0.775, 0.054), tp=(0.270, 0.029), hs_aw=(0.798, 0.041),
                         hs_tp=(0.801, 0.044), tp_aw=(0.743, 0.053), tp_aw_hs=(0.811, 0.044),
                         ts_or_tp_hs=(0.789, 0.046), lr=(0.761, 0.046)),
    "deepseek-coder-33b": dict(hs=(0.590, 0.087), aw=(0.303, 0.038), tp=(0.244, 0.024), hs_aw=(0.555, 0.086),
                               hs_tp=(0.560, 0.085), tp_aw=(0.426, 0.071), tp_aw_hs=(0.575, 0.085),
                               ts_or_tp_hs=(0.513, 0.071), lr=(0.546, 0.094)),
}
# its baselines, in the order of ORIGINAL_MODELS
PUBLISHED_BASELINES = {
    "Random": [(0.313, 0.047), (0.201, 0.029), (0.303, 0.039), (0.205, 0.031)],
    "Prompting zero-shot": [(0.450, 0.018), (0.273, 0.028), (0.263, 0.028), (0.166, 0.016)],
    "Prompting one-shot": [(0.462, 0.026), (0.263, 0.029), (0.269, 0.026), (0.254, 0.033)],
    "Prompting CoT": [(0.417, 0.031), (0.260, 0.051), (0.231, 0.082), (0.047, 0.013)],
    "CodeBERT": [(0.539, 0.038), (0.359, 0.034), (0.471, 0.040), (0.339, 0.037)],
    "MSP": [(0.277, 0.031), (0.250, 0.028), (0.220, 0.026), (0.192, 0.022)],
    "Perplexity": [(0.240, 0.030), (0.215, 0.026), (0.202, 0.025), (0.163, 0.020)],
    "Attention Score": [(0.493, 0.054), (0.336, 0.047), (0.454, 0.055), (0.303, 0.045)],
    "MTE": [(0.265, 0.031), (0.266, 0.038), (0.270, 0.030), (0.168, 0.020)],
    "Linear TS+OR+TP+HS": [(0.677, 0.070), (0.550, 0.064), (0.742, 0.051), (0.471, 0.086)],
    "Linear LR": [(0.520, 0.080), (0.385, 0.058), (0.621, 0.050), (0.552, 0.092)],
    "MLP HS": [(0.669, 0.071), (0.570, 0.069), (0.735, 0.056), (0.579, 0.089)],
}
# (a, b): a - b is reported on every test set
C1 = [(MAIN, d) for d in NINE if d != MAIN] + [("hs", "tp")]
RQ4 = [("richtp", "tp"), ("hs_tp", "tp"), ("hs_richtp", "richtp"), ("hs_richtp", "hs_tp"),
       ("richtp_tp_config", "tp"), ("hs_richtp_hs_tp_config", "richtp_tp_config"), ("richtp", "richtp_tp_config"),
       ("hs_richtp", "hs_richtp_hs_tp_config"), ("hs_richtp_hs_tp_config", "hs_tp")]
RQ5 = [("student", "richtp"), ("hs_richtp", "student"), ("distilled", "richtp"), ("distilled", "student"),
       ("hs_richtp", "distilled")]
CONTROL = [("control_with_test_tasks", "control_without_test_tasks")]


def pr_auc(y, p):
    """PR-AUC with the trapezoidal rule."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    total = y.sum()
    if not total:
        return math.nan
    order = np.argsort(p, kind="mergesort")[::-1]
    s, yy = p[order], y[order]
    n = len(s)
    start = np.r_[True, s[1:] != s[:-1]]
    starts = np.flatnonzero(start)
    sizes = np.diff(np.r_[starts, n])
    pos = np.add.reduceat(yy, starts)
    before = np.r_[0.0, np.cumsum(pos)[:-1]]
    block = np.cumsum(start) - 1
    k = np.arange(n) - starts[block] + 1
    tp = before[block] + k * pos[block] / sizes[block]
    precision = tp / np.arange(1, n + 1)
    recall = tp / total
    P = np.r_[1.0, precision]
    R = np.r_[0.0, recall]
    return float(np.sum(np.diff(R) * (P[1:] + P[:-1]) / 2))


def summary(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=0)), lo=float(lo), hi=float(hi), n=int(len(x)),
                significant=bool(lo > 0 or hi < 0))


def unpaired(a_mean, a_sd, b_mean, b_sd):
    """CI of b - a for values not scored on the same resamples."""
    d = b_mean - a_mean
    half = 1.96 * math.sqrt(a_sd ** 2 + b_sd ** 2)
    return dict(delta=d, lo=d - half, hi=d + half, significant=bool(abs(d) > half))


class TestSet:
    def __init__(self, entry):
        path = ROOT / entry["file"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], f"checksum mismatch: {entry['file']}"
        reader = csv.reader(io.StringIO(gzip.decompress(raw).decode("utf-8")))
        header = next(reader)
        rows = list(reader)
        assert len(rows) == entry["lines"], entry["file"]
        self.record = [r[0] for r in rows]                       # a response or a reference solution
        self.y = np.array([int(r[4]) for r in rows], dtype=np.int64)
        self.runs = {name: np.array([float(r[i]) for r in rows]) for i, name in enumerate(header) if i >= 5}
        self.detectors = defaultdict(list)
        for name in self.runs:
            self.detectors[name.split("@")[0]].append(name)

    def bootstrap(self):
        order = list(dict.fromkeys(self.record))
        by = defaultdict(list)
        for i, u in enumerate(self.record):
            by[u].append(i)
        index = [np.array(by[u], dtype=np.int64) for u in order]
        rng = np.random.RandomState(SEED)
        per_run = {n: [] for n in self.runs}
        rand = []
        for _ in range(ITERATIONS):
            draw = rng.choice(len(order), len(order), replace=True)
            rows = np.concatenate([index[j] for j in draw])
            if len(rows) == 0 or not self.y[rows].sum():
                continue
            y = self.y[rows]
            pi, n = float(y.mean()), len(y)
            rand.append(pi + (1 - pi) / (2 * n))
            for name, p in self.runs.items():
                per_run[name].append(pr_auc(y, p[rows]))
        self.per_run = {n: np.array(v) for n, v in per_run.items()}
        self.rand = np.array(rand)
        self.value = {d: np.mean([self.per_run[n] for n in names], axis=0) for d, names in self.detectors.items()}

    def describe(self, d):
        s = summary(self.value[d])
        runs = [dict(seed=int(n.split("@")[1]), point=pr_auc(self.y, self.runs[n]), mean=float(self.per_run[n].mean()))
                for n in self.detectors[d]]
        return dict(mean=s["mean"], sd=s["sd"], lo=s["lo"], hi=s["hi"],
                    point=float(np.mean([r["point"] for r in runs])),
                    minus_random=summary(self.value[d] - self.rand), runs=runs)

    def differences(self, pairs):
        out = {f"{a} - {b}": summary(self.value[a] - self.value[b])
               for a, b in pairs if a in self.value and b in self.value}
        if {"tp", "hs_tp", "richtp", "hs_richtp"} <= set(self.value):
            out["change in HS gain"] = summary((self.value["hs_richtp"] - self.value["richtp"])
                                               - (self.value["hs_tp"] - self.value["tp"]))
        if {"tp", "hs_tp", "richtp_tp_config", "hs_richtp_hs_tp_config"} <= set(self.value):
            out["change in HS gain (control)"] = summary(
                (self.value["hs_richtp_hs_tp_config"] - self.value["richtp_tp_config"])
                - (self.value["hs_tp"] - self.value["tp"]))
        return out


def main():
    t0 = time.time()
    manifest = json.loads((DATA / "predictions/MANIFEST.json").read_text(encoding="utf-8"))
    res = dict(metric="PR-AUC (trapezoidal rule)",
               resampling=f"test records, {ITERATIONS} times, numpy.random.RandomState({SEED})",
               test_sets={}, rq1={}, rq2={})
    sets = {}
    for entry in manifest["test_sets"]:
        name = Path(entry["file"]).name[:-len(".csv.gz")]
        ts = TestSet(entry)
        ts.bootstrap()
        sets[name] = ts
        res["test_sets"][name] = dict(
            description=entry["description"], lines=int(len(ts.y)), positives=int(ts.y.sum()),
            share_hallucinated=float(ts.y.mean()), records=len(set(ts.record)),
            resamples=int(len(ts.rand)), detectors={d: ts.describe(d) for d in ts.detectors},
            differences=ts.differences(C1 + RQ4 + RQ5 + CONTROL))
        print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)

    for i, m in enumerate(ORIGINAL_MODELS):
        orig = res["test_sets"][f"original_split_{m}"]["detectors"]
        disj = res["test_sets"][f"task_disjoint_{m}"]["detectors"]
        pub = PUBLISHED[m]
        rq1 = {}
        for d in NINE:
            runs = [r["mean"] for r in orig[d]["runs"]]
            mean, sd = statistics.mean(runs), statistics.stdev(runs)
            half = T_975_2 * sd * math.sqrt(1 + 1 / len(runs))
            rq1[d] = dict(published=pub[d][0], reproduced=mean, run_means=runs,
                          prediction_interval=[mean - half, mean + half],
                          reproduces=bool(abs(mean - pub[d][0]) <= half))
        main = orig[MAIN]
        baselines = {}
        for name, v in PUBLISHED_BASELINES.items():
            baselines[name] = dict(value=v[i][0], **unpaired(v[i][0], v[i][1], main["mean"], main["sd"]))
        res["rq1"][m] = dict(detectors=rq1, main_minus_baselines=baselines)
        res["rq2"][m] = {d: unpaired(orig[d]["mean"], orig[d]["sd"], disj[d]["mean"], disj[d]["sd"]) for d in NINE}

    fit = json.loads((DATA / "distillation_selection.json").read_text(encoding="utf-8"))
    res["rq5_distillation"] = {
        pair: dict(base_validation_mean_r2=v["base_validation_mean_r2"],
                   validation_mean_r2=float(np.mean([r["validation_mean_r2"] for r in v["runs"]])),
                   gain=float(np.mean([r["validation_mean_r2"] for r in v["runs"]])) - v["base_validation_mean_r2"],
                   runs=v["runs"])
        for pair, v in fit.items()}
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "statistics/results.json"
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    n_rep = sum(c["reproduces"] for v in res["rq1"].values() for c in v["detectors"].values())
    print(f"RQ1: {n_rep} of 36 published values reproduce; wrote {out.relative_to(ROOT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
