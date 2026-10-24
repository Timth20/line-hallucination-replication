# Generation, labels and features of the new responses

Generates the new responses, labels their lines and extracts the same features as `../original_outputs/`.
Qwen2.5-Coder-7B-Instruct answers the MBPP, HumanEval, HumanEval-Java and Defects4J tasks in the data released by the
original study; the five models answer BigCodeBench v0.1.4 (Full-Instruct). Greedy decoding, at most 1,280 new
tokens. Lines are labeled with `difflib.ndiff` against the closest reference solution, following the original
labeling code. Features are extracted for the response (`model_output`) and for the chosen reference solution
(`reference_control`), each after the prompt.

Environment (run everything from this folder):

```bash
conda env create -f environment.yml
conda activate saner-extension
```

One NVIDIA GPU with compute capability 8.x or 9.x. Required memory: 44 GiB for most configs, 22 GiB for DSC-1.3B,
74 GiB for Defects4J and 130 GiB for DSC-33B (4-bit). CodeLlama needs `HF_TOKEN`. `run` checks the package versions
against `src/saner_extension/runtime_lock.json`.

Example (configs are in `configs/experiments/`):

```bash
CFG=configs/experiments/qwen2_5_coder_7b_mbpp.yaml
python main.py prepare --config $CFG
python main.py run     --config $CFG --sample-start 0 --max-samples 1
python main.py run     --config $CFG
python main.py verify  --config $CFG
```

Tasks are sorted by length, and the first `run` must be the longest one (`--sample-start 0 --max-samples 1`).
Rerunning resumes and reuses saved generations. Default output: `../outputs_new`.

Each task gives `<task_uid>-model_output.*` and `<task_uid>-reference_control.*` (output, labels, features, done
marker). The responses and labels are also in `data/responses/`.
