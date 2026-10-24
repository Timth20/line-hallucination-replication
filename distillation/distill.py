"""RQ5: distill a Student toward the Teacher's hidden states on the code of the Teacher's Mixed training records.

    python distillation/distill.py prepare --teacher <model> --features <Teacher extraction output> --out <dir>
    python distillation/distill.py train   --teacher <model> --run <1|2|3> --lr <lr> --features ... --out <dir>
    python distillation/distill.py select  --teacher <model> --run <1|2|3> --out <dir>

`prepare` fits a ridge map from the base Student to the Teacher states; `train` keeps it fixed and learns LoRA weights;
`select` writes the adapter.safetensors that student_hidden_states.py reads.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import lora as L  # noqa: E402

CONFIG = json.loads((ROOT / "configs/distillation.json").read_text(encoding="utf-8"))
TRAIN, RIDGE = CONFIG["train"], CONFIG["ridge"]
SD_FLOOR = 1e-6


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1), encoding="utf-8", newline="\n")


class RunningMoments:
    def __init__(self, dim, device):
        self.n = 0
        self.s = torch.zeros(dim, dtype=torch.float64, device=device)
        self.ss = torch.zeros(dim, dtype=torch.float64, device=device)

    def add(self, x):
        x = x.to(self.s.device, torch.float64)
        self.n += x.shape[0]
        self.s += x.sum(dim=0)
        self.ss += (x * x).sum(dim=0)

    def result(self):
        mean = self.s / self.n
        var = (self.ss / self.n - mean * mean).clamp_min(0.0)
        return mean.float(), var.sqrt().clamp_min(SD_FLOOR).float()


def ridge_solutions(gram, cross, alphas):
    """(X'X + alpha I)^-1 X'Y for every alpha, from one eigendecomposition."""
    lam, vec = torch.linalg.eigh(gram.double())
    proj = vec.T @ cross.double()
    for alpha in alphas:
        yield alpha, (vec @ (proj / (lam + alpha)[:, None])).float()


def mean_r2(sse, sst):
    return float((1.0 - sse.double() / sst.double().clamp_min(1e-12)).mean())


def subsample_rows(n_rows, quota, rng):
    if quota >= n_rows:
        return np.arange(n_rows)
    return np.sort(rng.choice(n_rows, size=quota, replace=False))


def lr_lambda_for(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


def step_batches(train_ids, seed, per_step, steps):
    """Batches cut from a seeded permutation of the records per epoch."""
    batches, epoch, buf = [], 0, []
    while len(batches) < steps:
        order = np.random.default_rng(seed + epoch).permutation(len(train_ids))
        for i in order:
            buf.append(train_ids[i])
            if len(buf) == per_step:
                batches.append(buf)
                buf = []
                if len(batches) == steps:
                    break
        epoch += 1
    return batches


class Teacher:
    """The Teacher's records of one split role, sorted by sample id, and its hidden states on their code rows."""

    def __init__(self, features, model, role):
        with open(ROOT / f"data/splits/task_disjoint_{model}.csv", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == role]
        self.features = Path(features)
        self.records = []
        for r in sorted(rows, key=lambda r: r["sample_id"]):
            base = self.features / r["dataset"] / r["sample_id"]
            output = json.loads(Path(f"{base}.output.json").read_text(encoding="utf-8"))
            output = output.get("data", output)
            ids = torch.tensor(output["input_ids"], dtype=torch.int64)
            n_rows, resp = len(ids) - 1, int(output["response_start"]) - 1
            if not 0 < resp < n_rows:
                raise ValueError("unexpected response boundary: " + r["sample_id"])
            self.records.append((r["sample_id"], n_rows, resp, ids, f"{base}.features.safetensors"))

    def rows(self, i, layers):
        sid, n_rows, resp, ids, path = self.records[i]
        with safe_open(path, framework="pt") as f:
            states = f.get_tensor("hidden_states")
            available = [int(x) for x in json.loads(f.metadata()["hidden_state_layers"])]
        if states.shape[0] != n_rows:
            raise ValueError("feature rows differ from the token ids: " + sid)
        value = states[:, [available.index(layer) for layer in layers], :].reshape(n_rows, -1).float()
        return value[resp:]


def student_rows(model, ids, layers, start=0):
    """[T - 1 - start, len(layers) * d_s], layer-major, from rows start..T-2."""
    hs = L.forward_hidden(model, ids.numpy().astype("int64"))
    return torch.cat([hs[layer][0][start:-1] for layer in layers], dim=1)


def pair(model):
    return CONFIG["pairs"][model]


def train_seed(run):
    return int(TRAIN["seeds"][str(run)])


def stage_prepare(args):
    started = time.time()
    p = pair(args.teacher)
    layers = p["layers"]
    out = Path(args.out) / args.teacher
    out.mkdir(parents=True, exist_ok=True)
    train, val = Teacher(args.features, args.teacher, "train"), Teacher(args.features, args.teacher, "validation")
    rt, rv = RIDGE["train_rows"], RIDGE["val_rows"]
    code_rows_train = sum(r - q for _, r, q, _, _ in train.records)
    code_rows_val = sum(r - q for _, r, q, _, _ in val.records)
    rng = np.random.default_rng(RIDGE["seed"])
    q_train = {sid: math.ceil(rt * (r - q) / code_rows_train) for sid, r, q, _, _ in train.records}
    q_val = {sid: math.ceil(rv * (r - q) / code_rows_val) for sid, r, q, _, _ in val.records}
    model, _ = L.load_student(p["student"], p["student_revision"], cache_dir=args.hf_cache)
    model.eval()
    # moments of the code rows, and the ridge subsamples as raw values
    tm = sm = None
    X, Y, Xv, Yv = [], [], [], []
    with torch.inference_mode():
        for n, (sid, rows, resp, ids, _) in enumerate(train.records):
            t = train.rows(n, layers)
            s = student_rows(model, ids, layers, start=resp).float()
            if s.shape[0] != t.shape[0] or not torch.isfinite(s).all() or not torch.isfinite(t).all():
                raise ValueError("bad rows for " + sid)
            if tm is None:
                tm, sm = RunningMoments(t.shape[1], "cuda"), RunningMoments(s.shape[1], "cuda")
            tm.add(t.cuda())
            sm.add(s)
            k = torch.as_tensor(subsample_rows(t.shape[0], q_train[sid], rng))
            X.append(s[k.cuda()].cpu())
            Y.append(t[k])
        for n, (sid, rows, resp, ids, _) in enumerate(val.records):
            t = val.rows(n, layers)
            s = student_rows(model, ids, layers, start=resp).float()
            k = torch.as_tensor(subsample_rows(t.shape[0], q_val[sid], rng))
            Xv.append(s[k.cuda()].cpu())
            Yv.append(t[k])
    mu_t, sd_t = tm.result()
    mu_s, sd_s = sm.result()
    X = ((torch.cat(X).cuda() - mu_s) / sd_s)
    Y = ((torch.cat(Y).cuda() - mu_t) / sd_t)
    Xv = ((torch.cat(Xv).cuda() - mu_s) / sd_s)
    Yv = ((torch.cat(Yv).cuda() - mu_t) / sd_t)
    gram = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device="cuda")
    cross = torch.zeros(X.shape[1], Y.shape[1], dtype=torch.float64, device="cuda")
    for i in range(0, X.shape[0], 8192):
        xb = X[i:i + 8192].double()
        gram += xb.T @ xb
        cross += xb.T @ Y[i:i + 8192].double()
    del X, Y
    yv_mean = Yv.double().mean(dim=0)
    sst_sub = ((Yv.double() - yv_mean) ** 2).sum(dim=0)
    table, best = [], None
    for alpha, W in ridge_solutions(gram, cross, RIDGE["alphas"]):
        sse = torch.zeros_like(sst_sub)
        for i in range(0, Xv.shape[0], 8192):
            sse += ((Xv[i:i + 8192] @ W).double() - Yv[i:i + 8192].double()).pow(2).sum(dim=0)
        r2 = mean_r2(sse, sst_sub)
        table.append(dict(alpha=alpha, val_mean_r2=r2))
        log(f"alpha {alpha:g}: validation mean R2 {r2:.4f}")
        if best is None or r2 > best[1] or (r2 == best[1] and alpha > best[0]):
            best = (alpha, r2, W.cpu())
    # SST of all standardized validation code rows, for the R2 during training
    vm = None
    for n in range(len(val.records)):
        z = (val.rows(n, layers).cuda() - mu_t) / sd_t
        if vm is None:
            vm = RunningMoments(z.shape[1], "cuda")
        vm.add(z)
    val_mean = (vm.s / vm.n)
    val_sst = (vm.ss - vm.n * val_mean * val_mean).clamp_min(1e-12)
    torch.save(dict(layers=layers, mu_s=mu_s.cpu(), sd_s=sd_s.cpu(), mu_t=mu_t.cpu(), sd_t=sd_t.cpu(), W=best[2],
                    alpha=best[0], val_mean=val_mean.cpu(), val_sst=val_sst.cpu(), val_rows=int(vm.n)),
               out / "prepare.pt")
    write_json(out / "PREPARE.json", dict(teacher=args.teacher, layers=layers, alpha_table=table, alpha=best[0],
                                          ridge_val_mean_r2=best[1], seconds=round(time.time() - started, 1)))
    log("prepare done", f"alpha {best[0]:g}", f"R2 {best[1]:.4f}")


def evaluate(model, val, layers, prep):
    """Mean R^2 over the standardized Teacher dimensions on all validation code rows."""
    model.eval()
    W = prep["W"].cuda()
    mu_s, sd_s, mu_t, sd_t = (prep[k].cuda() for k in ("mu_s", "sd_s", "mu_t", "sd_t"))
    sse = torch.zeros_like(prep["val_sst"], device="cuda")
    with torch.inference_mode():
        for n, (sid, rows, resp, ids, _) in enumerate(val.records):
            z = (student_rows(model, ids, layers, start=resp).float() - mu_s) / sd_s
            y = (val.rows(n, layers).cuda() - mu_t) / sd_t
            sse += ((z @ W) - y).double().pow(2).sum(dim=0)
    model.train()
    return mean_r2(sse, prep["val_sst"].cuda())


def stage_train(args):
    started = time.time()
    p = pair(args.teacher)
    prep = torch.load(Path(args.out) / args.teacher / "prepare.pt", weights_only=False)
    layers = prep["layers"]
    lr = float(args.lr)
    if lr not in TRAIN["learning_rates"]:
        raise ValueError("learning rate outside the grid")
    run = Path(args.out) / args.teacher / f"run{args.run}" / f"lr{lr:g}"
    run.mkdir(parents=True, exist_ok=False)
    train, val = Teacher(args.features, args.teacher, "train"), Teacher(args.features, args.teacher, "validation")
    max_steps, every = TRAIN["max_steps"], TRAIN["evaluate_every_steps"]
    torch.manual_seed(train_seed(args.run))
    model, params = L.load_student(p["student"], p["student_revision"], lora=True, cache_dir=args.hf_cache)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    W = prep["W"].cuda()
    mu_s, sd_s, mu_t, sd_t = (prep[k].cuda() for k in ("mu_s", "sd_s", "mu_t", "sd_t"))
    optimizer = torch.optim.AdamW(params, lr=lr, betas=tuple(TRAIN["betas"]), eps=TRAIN["eps"],
                                  weight_decay=TRAIN["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_for(TRAIN["warmup_steps"], max_steps))
    batches = step_batches(list(range(len(train.records))), train_seed(args.run), TRAIN["sequences_per_step"], max_steps)
    history, best, step = [], dict(metric=-np.inf, step=None), 0

    def run_eval():
        nonlocal best
        metric = evaluate(model, val, layers, prep)
        history.append(dict(step=step, metric=metric))
        if metric > best["metric"]:
            best = dict(metric=metric, step=step)
            torch.save(dict(lora=L.lora_state(model), step=step, metric=metric, lr=lr), run / "best.pt")
        log(f"step {step} validation mean R2 {metric:.5f}")

    run_eval()
    while step < max_steps:
        batch = batches[step]
        denom = sum(train.records[i][1] - train.records[i][2] for i in batch)
        optimizer.zero_grad(set_to_none=True)
        for i in batch:
            sid, rows, resp, ids, _ = train.records[i]
            z = (student_rows(model, ids, layers, start=resp).float() - mu_s) / sd_s
            y = (train.rows(i, layers).cuda() - mu_t) / sd_t
            loss = ((z @ W) - y).pow(2).mean(dim=1).sum() / denom
            loss.backward()
            del z, y, loss
        torch.nn.utils.clip_grad_norm_(params, TRAIN["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1
        if step % every == 0 or step == max_steps:
            run_eval()
            k = TRAIN["patience"]
            if best["step"] is not None and len(history) > k and all(h["metric"] <= best["metric"] for h in history[-k:]) \
                    and best["step"] < history[-k]["step"]:
                log(f"early stop at step {step}; best step {best['step']}")
                break
    write_json(run / "RESULT.json", dict(teacher=args.teacher, run=args.run, seed=train_seed(args.run), lr=lr,
                                         layers=layers, final_step=step, best=best, base_metric=history[0]["metric"],
                                         history=history, seconds=round(time.time() - started, 1)))


def stage_select(args):
    root = Path(args.out) / args.teacher / f"run{args.run}"
    runs = [json.loads((root / f"lr{lr:g}" / "RESULT.json").read_text(encoding="utf-8"))
            for lr in TRAIN["learning_rates"]]
    chosen = sorted(runs, key=lambda r: (-r["best"]["metric"], r["best"]["step"], r["lr"]))[0]
    from safetensors.torch import save_file
    best = torch.load(root / f"lr{chosen['lr']:g}" / "best.pt", map_location="cpu", weights_only=False)
    save_file({k: v.contiguous() for k, v in best["lora"].items()}, str(root / "adapter.safetensors"))
    write_json(root / "SELECTED.json", dict(
        teacher=args.teacher, run=args.run, distillation_seed=train_seed(args.run),
        paired_detector_seed=CONFIG["paired_detector_seed"][str(args.run)], selected_lr=chosen["lr"],
        selected_step=chosen["best"]["step"], validation_mean_r2=chosen["best"]["metric"],
        base_validation_mean_r2=chosen["base_metric"],
        rule="highest validation mean R2; if equal, the earlier step, then the smaller learning rate"))
    log("selected", f"lr {chosen['lr']:g}", f"step {chosen['best']['step']}", f"R2 {chosen['best']['metric']:.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "train", "select"])
    parser.add_argument("--teacher", required=True, choices=sorted(CONFIG["pairs"]))
    parser.add_argument("--run", type=int, choices=[1, 2, 3])
    parser.add_argument("--lr")
    parser.add_argument("--features", help="extraction output of the Teacher (<root>/<model>)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hf-cache")
    args = parser.parse_args()
    if args.stage in ("train", "select") and not args.run:
        raise SystemExit("--run is required")
    if args.stage == "train" and not args.lr:
        raise SystemExit("--lr is required")
    if args.stage in ("prepare", "train") and not args.features:
        raise SystemExit("--features is required")
    {"prepare": stage_prepare, "train": stage_train, "select": stage_select}[args.stage](args)


if __name__ == "__main__":
    main()
