# Project runbook

Run commands from your repository root (the author's local checkout is
`D:\PROJECTS\SOC-RAG-Guard`). The commands below use Windows PowerShell and
`.venv/Scripts/python.exe`; install dependencies using the root README first.
The current `project.py` launcher expects the Windows environment layout.
On Linux/macOS, invoke modules directly with your environment's Python, for
example `python -m unittest discover -s tests -v` or
`python src/experiments/protocol.py --help`.

## Fresh-clone requirements

A clone includes source, documentation, selected result tables and figures.
It excludes the source dataset, full experiment outputs, model weights and
serialized preprocessor. Offline tests and result reading are available after
dependency installation. Training needs the local dataset; final prediction
needs a compatible trained bundle. RAG generation additionally needs Ollama,
configured models and the required experiment artifacts. Dataset acquisition
instructions remain blocked on verified provenance; see [status](PROJECT_STATUS.md).
The `--help` commands show the current required arguments.

## Status and tests

```powershell
.\.venv\Scripts\python.exe project.py status
.\.venv\Scripts\python.exe project.py test
```

## Final detector inference

```powershell
.\.venv\Scripts\python.exe project.py predict --input your_flow_features.csv --output new_predictions.csv
```

Input needs the 17 feature columns in `src/experiments/protocol.py`. Target labels are not required. Output must be a new file. The default bundle is `results/research_evaluation_v1/final_model`. CPU/GPU floating-point results can differ slightly. Predictions are investigative leads requiring analyst validation.

## Experiment commands

The launcher maps a script name to `src/experiments/<name>.py`:

```powershell
.\.venv\Scripts\python.exe project.py run protocol --help
.\.venv\Scripts\python.exe project.py run research_analysis --help
.\.venv\Scripts\python.exe project.py run extended_rag --help
```

To prepare a **new** comparison rather than overwrite a completed run:

```powershell
.\.venv\Scripts\python.exe project.py run protocol prepare --data dataset/train_test_network.csv --run results/comparison_new
.\.venv\Scripts\python.exe project.py run protocol train --run results/comparison_new --epochs 1 --smoke
.\.venv\Scripts\python.exe project.py run protocol train --run results/comparison_new --epochs 10
.\.venv\Scripts\python.exe project.py run protocol summarize --run results/comparison_new
.\.venv\Scripts\python.exe project.py run uncertainty --run results/comparison_new
```

Detailed scientific settings and limitations are in `docs/EXPERIMENT_PROTOCOL.md`. Original completed-run instructions remain in their result folders; translate `experiments_v1/<script>.py` to `src/experiments/<script>.py`, or use `project.py run <script>`.

## Results to read

Use the [results index](../results/README.md) for tracked GitHub artifacts.
The inventory below also describes local-only outputs and archives; a fresh
clone does not contain every listed path.

- `docs/reports/EXPERIMENT_REPORT.md`: publication-facing experiment report.
- `docs/reports/research_submission.zip`: curated code, documentation, figures, result tables and review materials.
- `results/research_evaluation_v1/RESEARCH_REPORT.md`: original consolidated run report retained for provenance.
- `results/research_evaluation_v1/figures/`: feature importance, confusion, ROC and RAG plots.
- `results/research_evaluation_v1/final_model/`: full-data refit artifact and preprocessor.
- `results/comparison_v1/`: original grouped-fold predictions and checkpoints.
- `results/research_evaluation_v1/extended_rag/rag_ablation/`: 50-pair RAG experiment, raw outputs, automatic audits and human review sheets.
- `results/research_evaluation_v1/research_artifacts.zip`: historical research artifact archive retained for provenance.

## Independent human review

Use `HUMAN_REVIEW_GUIDE.md`, the frozen corpus and blinded packet in the extended RAG output directory. Give each reviewer their own `human_reviewer_N.csv`; withhold automated labels and `review_key.csv`. After the actual reviews:

```powershell
.\.venv\Scripts\python.exe project.py run reviewer_agreement --first reviewer_1_completed.csv --second reviewer_2_completed.csv --out results/reviewer_agreement.json
```

Adjudicate disagreements before scoring. The agreement tool refuses blank or mismatched review sheets. Automated labels must not be presented as human review.

## Earlier work

Earlier scripts are under `archive/legacy_scripts`, and review copies under `archive/review_bundle`. Their project-relative roots were adjusted for the new depth; historical hardcoded paths remain historical. A verified ZIP of the exact pre-move code is under `archive/reorganization_backups`. Read these scripts before running: several train models or write older result filenames.

The existing `rag/` paths remain stable for retrieval and triage commands. The local Ollama service is separate from `.venv` and is required for generation/embedding calls.
