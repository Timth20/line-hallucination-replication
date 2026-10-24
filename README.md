# line-hallucination-replication

Replication package for the paper "Are Hidden States Worth It? Revisiting Line-Level Hallucination Detection in Code
LLMs".

## Recomputing the results

`data/predictions/` has the line-level test predictions of all detectors in the paper. The statistics and tables can
be recomputed on a CPU:

```bash
pip install numpy==1.26.4
python statistics/compute_statistics.py   # writes statistics/results.json (a few minutes)
python statistics/make_tables.py          # writes statistics/tables.md (Tables III-VI)
```

## Running the experiments

GPUs and the model weights from Hugging Face are needed. Python packages are in `requirements.txt`; the two folders in
`extraction/` have their own environments.

Feature extraction is in `extraction/`: `original_outputs/` for the four original models on the outputs released by
the original study, and `new_responses/` for generating and labeling the new responses (Qwen2.5-Coder-7B on the four
original benchmarks, the five models on BigCodeBench). Our responses and labels are also in `data/responses/`.

Detectors are trained and scored with `detectors/train.py`. `configs/detectors.json` has the configuration and seeds
of each detector, and `data/splits/` the records of each split. The original models on the nine original feature sets
use the released configurations of the original study; the other configurations come from the search in Section III
of the paper. Example:

```bash
python detectors/train.py fit --test-set task_disjoint_deepseek-coder-6.7b --detector hs_tp --seed 42 \
    --features <extraction output>/<model> --out runs/dsc6_hs_tp_s42
python detectors/train.py score --run runs/dsc6_hs_tp_s42 --features <extraction output>/<model> \
    --split data/splits/task_disjoint_deepseek-coder-6.7b.csv --out runs/dsc6_hs_tp_s42/test.json
```

For BigCodeBench, score the seed-42 detector of the model's Mixed test set with
`--split data/splits/bigcodebench_<model>.csv`.

`splits/task_disjoint.py` is the task-disjoint split. `splits/control_training_sets.py` builds the training sets of
the RQ2 control experiment.

RQ5 (distillation): `distillation/distill.py` and `distillation/student_hidden_states.py`, with the settings in
`configs/distillation.json`. Usage is in the docstring of each script. Distillation runs 1-3 are paired with detector
seeds 42-44.

## Data

`data/predictions/<test set>.csv.gz`: one row per test line with `sample_id`, `line`, `dataset`, `task_id`, `label`
(1 = hallucinated), and one column per detector run (`<detector>@<seed>`). Resampling is over `sample_id`. Test sets
are `original_split_*` (RQ1), `task_disjoint_*` (Mixed, task-disjoint split) and `bigcodebench_*`.

Detector names: `hs`, `aw`, `tp`, `hs_aw`, `hs_tp`, `tp_aw`, `tp_aw_hs` (main detector), `ts_or_tp_hs`, `lr`,
`richtp`, `hs_richtp`, `student`, `distilled`, and the detectors of the two control experiments.

`data/responses/`: our responses with token ids, lines and line labels (gzipped JSONL).
`data/distillation_selection.json`: selected step and validation R² of each distillation run.

## License

MIT
