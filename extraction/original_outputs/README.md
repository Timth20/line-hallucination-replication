# Features of the original models

Runs DeepSeek-Coder-1.3B/6.7B/33B-Instruct and CodeLlama-7B-Instruct on the outputs released by the original study
and saves the token-level features of every record. The release is downloaded at a pinned commit (SHA-256 checked),
and each record is fed to the model with its released token ids in one forward pass.

Environment (run everything from this folder):

```bash
conda env create -f environment.yml
conda activate saner-paper-extraction
```

GPU memory: about 24 GB for DSC-1.3B, 48 GB for the 7B models and 140 GB for DSC-33B (4-bit, as in the released
configuration). Most of it goes to the full attention maps used for the lookback ratio. CodeLlama is gated on Hugging
Face, so log in first. `run` checks the installed packages against `src/saner_reproduction/runtime_lock.json`.

Example for DSC-1.3B (configs are in `configs/experiments/`):

```bash
CFG=configs/experiments/paper_dsc_1_3b.yaml
OUT=../outputs_original
python main.py prepare --config $CFG --output-dir $OUT
python main.py run     --config $CFG --output-dir $OUT --device cuda:0 --sample-start 895 --max-samples 1
python main.py run     --config $CFG --output-dir $OUT --device cuda:0
python main.py verify  --config $CFG --output-dir $OUT
```

The first `run` only processes the longest record (index 895 for DSC-1.3B, 680 for DSC-6.7B, 913 for CodeLlama-7B,
698 for DSC-33B) to check GPU memory; the other runs do not start before it passes. Rerunning resumes and skips
finished records. The output folder must be named `outputs_original`.

Output: `<model>/<dataset>/<sample_id>.output.json`, `.labels.json`, `.features.safetensors` and `.done.json`.
Features: top-8 probabilities, hidden states, attention window, top logits and token similarities of the last layers,
output ranks and lookback ratios. Feature row p predicts token p+1 (the response starts at row `response_start - 1`).
`detectors/train.py --features` takes `<output>/<model>`.
