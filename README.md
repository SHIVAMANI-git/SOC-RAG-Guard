# SOC-RAG Guard

Network-flow classification and evidence-grounded security triage.

SOC-RAG Guard studies two questions: how five neural classifiers perform on
shared grouped network-flow partitions, and whether reference evidence improves
security-triage reports generated from fixed detector predictions. It is a
research prototype; external and chronological validation remain outstanding.

## Start here

| Researcher question | Read |
|---|---|
| What was studied and what was found? | [Experimental report](docs/reports/EXPERIMENT_REPORT.md) |
| Where did the project start and how did it develop? | [Development history](docs/DEVELOPMENT_HISTORY.md) |
| What is completed or pending? | [Project status](docs/PROJECT_STATUS.md) |
| How does the system work? | [Architecture and source map](docs/ARCHITECTURE.md) |
| How were leakage and evaluation handled? | [Experimental protocol](docs/EXPERIMENT_PROTOCOL.md) |
| How do I run it? | [Runbook](docs/RUNBOOK.md) |
| Which results are available in GitHub? | [Results index](results/README.md) |
| How should future work be maintained? | [Contribution and maintenance guide](CONTRIBUTING.md) |

## Project status

**Current stage: experimental evaluation completed; independent validation pending.**
The progression is baseline flow detection → refined grouping and preprocessing
→ five-model comparison → explanations and final refit → paired RAG evaluation
→ validation and reproducibility. Read the [development history](docs/DEVELOPMENT_HISTORY.md)
for evidence at each stage and [maintenance guidance](CONTRIBUTING.md) for how
experiments and artifacts are preserved.

As reviewed on **30 October 2026**, the grouped five-model comparison,
SHAP/LIME analysis, full-data Transformer refit, and 50-pair automated RAG
study are completed locally. Independent human adjudication, dataset provenance
confirmation, external validation and chronological evaluation remain pending.
See [the status ledger](docs/PROJECT_STATUS.md) for evidence and completion criteria.

The Transformer has saved out-of-fold attack F1 **0.8925** and accuracy
**0.8432**; corrected comparisons do not establish architecture superiority.
In the RAG study, automated supported-claim scores improved among complete
pairs, but five retrieval-condition generations failed and the failure-adjusted
95% interval includes zero. These are report-quality results, not a detector
accuracy gain. See the [report](docs/reports/EXPERIMENT_REPORT.md) for denominators,
intervals and limitations.

## Reports

| Resource | Location |
|---|---|
| Experimental report | [EXPERIMENT_REPORT.md](docs/reports/EXPERIMENT_REPORT.md) |
| Model comparison | [model_comparison.csv](results/research_evaluation_v1/model_comparison.csv) |
| Figures | [results/research_evaluation_v1/figures](results/research_evaluation_v1/figures/) |
| Execution guide | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Evaluation protocol | [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) |

## Installation

Python 3.10 is recommended. Create an isolated environment and install the
versioned dependencies (including the formatting check):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

The optional RAG generation workflow also requires a locally running Ollama
service and the model configured by the experiment command. The classifier,
tests and saved-result review do not require that service.

## Repository layout

```text
SOC-RAG-Guard/
â”œâ”€â”€ src/experiments/     Training, explanation, statistics and evaluation
â”œâ”€â”€ tests/               Offline verification
â”œâ”€â”€ rag/                 Retrieval, evidence corpus and triage
â”œâ”€â”€ dataset/             Local source data and partitions (not versioned)
â”œâ”€â”€ results/             Tracked summaries/figures; local full experiment artifacts
â”œâ”€â”€ docs/                Research guide, status, protocol and execution documentation
â”œâ”€â”€ .github/workflows/   Automated test and formatting checks
â”œâ”€â”€ CONTRIBUTING.md      Maintenance and change-validation guide
â”œâ”€â”€ archive/             Local historical snapshots (not versioned)
â”œâ”€â”€ .venv/               Local project environment (not versioned)
â”œâ”€â”€ project.py           Command entry point
â””â”€â”€ README.md
```

## Dataset Description

The local source file, `dataset/train_test_network.csv`, contains **211,043
network-flow records** with 44 columns. The binary target is `label`: **50,000
normal records (0)** and **161,043 attack records (1)**. The following counts
were verified directly against the CSV's `type` and `label` columns; `type` is
not a detector input.

| Type | Records |
|---|---:|
| normal | 50,000 |
| backdoor | 20,000 |
| ddos | 20,000 |
| dos | 20,000 |
| injection | 20,000 |
| mitm | 1,043 |
| password | 20,000 |
| ransomware | 20,000 |
| scanning | 20,000 |
| xss | 20,000 |
| **Total** | **211,043** |

### Purpose and selection rationale

This CSV supplies labeled flow observations for binary attack detection and
held-out telemetry for the paired RAG report study. Its normal/attack labels
support supervised classification, while its connection fields support grouped
evaluation. The 17 inputs describe traffic volume, protocol, service and
connection behavior; addresses, ports and target annotations are excluded from
detector inputs in the implemented protocol.

These are observable properties of the current design, not a verified history
of dataset selection. The original reason for selecting this dataset over
alternatives, any upstream sampling, and the rationale for this exact feature
subset are not documented in the inspected materials. No feature-selection
optimality or representativeness claim is made. Confirm these decisions with
the original data provider before reporting them as research rationale.

### Detector features and preprocessing

The detector uses **17 source features: 14 numeric and 3 categorical**, as
specified in [protocol.py](src/experiments/protocol.py):

- Numeric: `duration`, `src_bytes`, `dst_bytes`, `missed_bytes`, `src_pkts`,
  `src_ip_bytes`, `dst_pkts`, `dst_ip_bytes`, `dns_qclass`, `dns_qtype`,
  `dns_rcode`, `http_request_body_len`, `http_response_body_len`,
  `http_status_code`.
- Categorical: `proto`, `service`, `conn_state`.

IP addresses, ports, `label`, `type` and the remaining source columns are
excluded from detector inputs. Invalid numeric values and infinities become
missing values; categorical missing values become `<MISSING>`. Numeric median
imputation and standardization, and categorical one-hot encoding, are fitted
only on the training partition of each fold. Unseen categories are ignored by
the encoder. The encoded input width can exceed 17.

### Five-fold grouping and validation limitations

The evaluation uses five shuffled, stratified grouped outer folds (seed 42).
Connection grouping uses `src_ip`, `src_port`, `dst_ip`, `dst_port` and `proto`,
with endpoint order canonicalized so that both directions share a group.
Records sharing either a bidirectional connection or an exact normalized
17-feature signature are joined into connected components. Each component
stays within one partition, and the code checks for pairwise component,
signature and connection overlap across training, validation and test sets.
Within each outer development set, the first split of a four-fold stratified
grouped splitter supplies training and validation partitions. All compared
architectures use the same saved assignments; group sizes cause unequal folds.

These controls address exact-signature and connection leakage, but do not
establish independence between hosts, campaigns or collection environments.
The dataset was used during earlier development and is not an external
validation set. Splits are not chronological, and feature order is not a
traffic time sequence: the results do not establish performance on future
traffic, temporal drift or unseen environments. Independent external and
chronological holdout evaluations remain necessary. The final full-data refit
has no additional independent test estimate. See the
[evaluation protocol](docs/EXPERIMENT_PROTOCOL.md).

### Availability and provenance

The source CSV and derived partitions are local-only: `dataset/` is excluded
from Git by [.gitignore](.gitignore) and is not included in the GitHub repository
or submission archive. The inspected project materials do not verify the
original dataset name, publisher, release, license or download location.
Consequently, no dataset attribution or download link is asserted here.
Reproduction requires a separately obtained, provenance-verified copy placed
at `dataset/train_test_network.csv`, followed by the [runbook](docs/RUNBOOK.md).
The preparation code records the source SHA256 in `protocol.json`, and training
checks that fingerprint; this verifies file consistency, not original provenance.

## Execution

Run from the repository root. Status and offline tests can run after installation.
Prediction additionally requires the local trained model/preprocessor bundle and
a CSV containing the 17 required features; these are not supplied by a fresh clone:

```powershell
.\.venv\Scripts\python.exe project.py status
.\.venv\Scripts\python.exe project.py test
.\.venv\Scripts\python.exe project.py predict --input flows.csv --output predictions.csv
```

The launcher uses the project environment for each operation. Prediction input requires the 17 flow features defined in `src/experiments/protocol.py`; labels are optional. Each prediction output must use a new filename.

Individual experiment commands are available through:

```powershell
.\.venv\Scripts\python.exe project.py run protocol --help
.\.venv\Scripts\python.exe project.py run extended_rag --help
```

## Evaluation

The classifier comparison includes 1D-CNN, LSTM, CNN-LSTM, BiLSTM and Transformer Encoder models evaluated on five shared grouped partitions. The input comprises tabular flow attributes. Feature order does not represent chronological traffic observations.

Exact feature signatures and bidirectional connection groups are isolated across training, validation and test partitions. Preprocessing is fitted on training data. The Transformer deployment candidate was selected using validation loss and refitted on the full dataset; the refit has no additional independent test estimate.

The paired RAG study uses identical held-out telemetry and detector predictions in both conditions. Report quality, citations, generation failures and latency are evaluated separately. Automated quality estimates remain subject to reviewer error and require independent human adjudication. The failure-adjusted interval does not establish a reliable end-to-end enhancement.

## Reproducibility

Saved datasets, predictions, model artifacts and completed experiment records retain their original contents. Historical runbooks may reference earlier source locations; current commands are documented in `docs/RUNBOOK.md`. The migration record and source snapshots are listed in `docs/MIGRATION.md`.

Use a separate result directory for each changed configuration. The dataset is not included in the submission archive. Historical Random Forest outputs are supplementary baselines.

Large datasets, intermediate checkpoints, raw model responses, generated
prediction tables and backup archives are intentionally excluded from Git.
Their local copies are not removed. The repository retains the reviewed report,
summary statistics, figures and final-model metadata. Model weights and the
serialized preprocessor are local-only and excluded by the artifact patterns
in `.gitignore`. A fresh clone cannot run final-model inference until a compatible
bundle has been supplied or regenerated. To reproduce a
full training run, place the source CSV at `dataset/train_test_network.csv` and
follow `docs/RUNBOOK.md`.

## Verification

```powershell
.\.venv\Scripts\python.exe project.py test
.\.venv\Scripts\python.exe -m black --check project.py src tests rag
```

GitHub Actions runs the same tests and formatting check on pushes and pull
requests.
