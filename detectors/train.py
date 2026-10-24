"""Train a detector of configs/detectors.json and score it on a test split.

    python detectors/train.py fit   --test-set <test set> --detector <detector> --seed <seed>
                                    --features <extraction output> --out <run directory>
                                    [--student <dir>] [--distilled-student <dir>]
    python detectors/train.py score --run <run directory> --features <extraction output> --split <split file>
                                    --out <predictions.json> [--student <dir>] [--distilled-student <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as D  # noqa: E402
import features as Fe  # noqa: E402
import model as M  # noqa: E402

ROOT = HERE.parent
PINNED = {"transformers": "4.46.3", "accelerate": "1.1.1"}
CONTROL = {"control_with_test_tasks": "train_with_test_tasks",
           "control_without_test_tasks": "train_without_test_tasks"}


class Rows(torch.utils.data.Dataset):
    def __init__(self, samples, blocks, students):
        if not samples:
            raise ValueError("empty split")
        self.samples, self.blocks, self.students = samples, blocks, students

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return dict(features=Fe.build(s, self.blocks, self.students), claim_masks=s.claim_masks,
                    labels=s.label_values, row=s.row)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1), encoding="utf-8", newline="\n")


def check_runtime():
    import accelerate
    import transformers
    found = {"transformers": transformers.__version__, "accelerate": accelerate.__version__}
    if found != PINNED:
        raise SystemExit(f"the training code needs {PINNED}, found {found}")


def students_of(args):
    out = {}
    if args.student:
        out["student"] = Fe.StudentStates(args.student)
    if args.distilled_student:
        out["distilled_student"] = Fe.StudentStates(args.distilled_student)
    return out


def predictions(model, samples, blocks, students, recipe, use_cuda, fp32_fallback=None):
    """Line probabilities with FP16 autocast on CUDA; with fp32_fallback, a non-finite batch is rerun in FP32."""
    device = "cuda" if use_cuda else "cpu"
    batch_size = int(recipe["batch_size"])
    model.to(device).eval()
    rows = Rows(samples, blocks, students)
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            records = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
            batch = {k: v.to(device) for k, v in M.collate(records).items()}
            if use_cuda:
                try:
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model(**batch)
                except FloatingPointError:
                    if fp32_fallback is None:
                        raise
                    output = model(**batch)
                    fp32_fallback.extend(r["row"]["sample_id"] for r in records)
            else:
                output = model(**batch)
            prob = output["logits"].float().cpu().sigmoid()
            for i, record in enumerate(records):
                for j, label in enumerate(record["labels"].tolist()):
                    result.append(dict(sample_id=record["row"]["sample_id"], dataset=record["row"]["dataset"],
                                       task_id=record["row"]["task_id"], line=j, label=int(label),
                                       probability=float(prob[i, j])))
    return result


def trapezoidal_pr_auc(values):
    from sklearn.metrics import auc, precision_recall_curve
    precision, recall, _ = precision_recall_curve([r["label"] for r in values], [r["probability"] for r in values])
    return float(auc(recall, precision))


def fit(args):
    from safetensors.torch import save_file
    from transformers import set_seed
    check_runtime()
    config = json.loads((ROOT / "configs/detectors.json").read_text(encoding="utf-8"))[args.test_set][args.detector]
    if args.seed not in config["seeds"]:
        raise SystemExit(f"seed {args.seed} is not a seed of this detector: {config['seeds']}")
    recipe, head, blocks = dict(config["recipe"], seed=args.seed), dict(config["head"]), config["features"]
    split = ROOT / "data" / "splits" / f"{args.test_set}.csv"
    model_name = args.test_set.split("_", 2)[-1]          # original_split_<model> or task_disjoint_<model>
    load = lambda role: D.load_split(args.features, split, role)
    train_samples, val_samples = load("train"), load("validation")
    if args.detector in CONTROL:
        sets = json.loads((ROOT / "data/splits/control_training_sets.json").read_text(encoding="utf-8"))[model_name]
        keep, keep_val = set(sets[CONTROL[args.detector]]), set(sets["validation"])
        train_samples = [s for s in train_samples if s.row["sample_id"] in keep]
        val_samples = [s for s in val_samples if s.row["sample_id"] in keep_val]
        if len(train_samples) != len(keep) or len(val_samples) != len(keep_val):
            raise SystemExit("the control training sets do not match the split file")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    M.OFFLOAD["enabled"] = bool(args.offload)
    use_cuda = torch.cuda.is_available() and not args.cpu
    students = students_of(args)
    train_rows, val_rows = Rows(train_samples, blocks, students), Rows(val_samples, blocks, students)
    set_seed(int(recipe["seed"]))
    input_dim = train_rows[0]["features"].shape[-1]
    write_json(out / "run.json", dict(test_set=args.test_set, detector=args.detector, features=blocks, head=head,
                                      recipe=recipe, input_dim=input_dim))
    started = time.time()
    model = M.ClaimModel(input_dim, head, recipe["pos_weight"])
    trainer = M.make_trainer(model, train_rows, val_rows, recipe, out / "trainer", use_cuda=use_cuda)
    result = trainer.train()
    updates = trainer.lr_scheduler.last_epoch
    if updates < 1 or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("no finite trained detector or optimizer update")
    history = trainer.state.log_history
    del trainer
    values = predictions(model, val_samples, blocks, students, recipe, use_cuda)
    model.cpu()
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(out / "model.safetensors"))
    write_json(out / "training.json", dict(global_step=result.global_step, optimizer_updates=updates,
                                           eval_loss=[h["eval_loss"] for h in history if "eval_loss" in h],
                                           train_loss=[h["loss"] for h in history if "loss" in h],
                                           seconds=round(time.time() - started, 1),
                                           validation_pr_auc=trapezoidal_pr_auc(values)))
    print(f"{args.test_set} {args.detector} seed {args.seed}: validation PR-AUC {trapezoidal_pr_auc(values):.3f}")


def score(args):
    from safetensors.torch import load_file
    run = json.loads((Path(args.run) / "run.json").read_text(encoding="utf-8"))
    model = M.ClaimModel(run["input_dim"], run["head"], run["recipe"]["pos_weight"])
    model.load_state_dict(load_file(str(Path(args.run) / "model.safetensors"), device="cpu"))
    samples = D.load_split(args.features, args.split, "test")
    use_cuda = torch.cuda.is_available() and not args.cpu
    fallback = [] if args.fp32_fallback else None
    if fallback is not None and int(run["recipe"]["batch_size"]) != 1:
        raise SystemExit("--fp32-fallback needs batch size 1")
    values = predictions(model, samples, run["features"], students_of(args), run["recipe"], use_cuda, fallback)
    if any(not math.isfinite(v["probability"]) for v in values):
        raise FloatingPointError("nonfinite predictions")
    write_json(args.out, dict(run=run, fp32_fallback=fallback, predictions=values))
    print(f"scored {len(samples)} records, {len(values)} lines")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--test-set", required=True)
    f.add_argument("--detector", required=True)
    f.add_argument("--seed", required=True, type=int)
    f.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fp32-fallback", action="store_true",
                   help="run a batch whose FP16 forward is not finite again in FP32 (batch size 1)")
    for p in (f, s):
        p.add_argument("--features", required=True, help="extraction output of the model (<root>/<model>)")
        p.add_argument("--student")
        p.add_argument("--distilled-student")
        p.add_argument("--cpu", action="store_true")
    f.add_argument("--offload", action="store_true", help="keep the tensors saved for backward in host memory")
    args = parser.parse_args()
    {"fit": fit, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
