"""RQ5: the hidden states of a Student on the Teacher's token sequences, for the Student and Distilled detectors.

    python distillation/student_hidden_states.py --teacher <model> --features <Teacher extraction output>
        --split <split file> --out <dir> [--adapter <run>/adapter.safetensors]

Row r is aligned with the Teacher's feature row r; detectors/train.py reads the files with --student or
--distilled-student.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lora as L  # noqa: E402

CONFIG = json.loads((HERE.parent / "configs/distillation.json").read_text(encoding="utf-8"))


def rows_of(model, ids, layers):
    """[T-1, len(layers), d] in bfloat16, from positions 0..T-2."""
    with torch.no_grad():
        hs = L.forward_hidden(model, ids.numpy().astype("int64"))
    return torch.stack([hs[layer][0][:-1] for layer in layers], dim=1).to(torch.bfloat16).cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teacher", required=True, choices=sorted(CONFIG["pairs"]))
    parser.add_argument("--features", required=True)
    parser.add_argument("--split", required=True, help="data/splits/task_disjoint_<model>.csv or bigcodebench_<model>.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--hf-cache")
    args = parser.parse_args()
    p = CONFIG["pairs"][args.teacher]
    layers = CONFIG["extract_layers"]
    model, _ = L.load_student(p["student"], p["student_revision"], adapter=args.adapter, cache_dir=args.hf_cache)
    model.eval()
    with open(args.split, encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    for n, r in enumerate(sorted(records, key=lambda r: r["sample_id"])):
        output = json.loads((Path(args.features) / r["dataset"] / f"{r['sample_id']}.output.json").read_text(encoding="utf-8"))
        output = output.get("data", output)
        ids = torch.tensor(output["input_ids"], dtype=torch.int64)
        value = rows_of(model, ids, layers)
        if value.shape[0] != len(ids) - 1 or not torch.isfinite(value.float()).all():
            raise ValueError("bad Student rows for " + r["sample_id"])
        target = Path(args.out) / r["dataset"] / f"{r['sample_id']}.features.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        save_file({"hidden_states": value.contiguous()}, str(target), metadata={"hidden_state_layers": json.dumps(layers)})
        if n % 200 == 0:
            print(f"{n}/{len(records)} records", flush=True)


if __name__ == "__main__":
    main()
