# Development history and current research stage

This is an evidence-based progression through the project, reconstructed from
retained code, split directories and reports. It is not a dated laboratory diary:
exact start dates and undocumented design decisions are not asserted here.

## 1. Starting point: supervised network-flow detection

The initial work used a local labeled network-flow CSV to distinguish normal
traffic from attacks. The retained source contains 211,043 records. Earlier
scripts and Random Forest outputs show a baseline stage before the current
five-neural-model comparison. Dataset publisher, acquisition history and the
original reason for choosing this source remain unverified.

Evidence: [dataset description](../README.md#dataset-description). Earlier
scripts are local-only under `archive/legacy_scripts/`; historical baseline
outputs are local-only under `results/`.

## 2. Refining features and evaluation partitions

Retained local partition directories progress from `split_v1` through
`split_v2_no_ports`, grouped variants and five-tuple validation. These names
record iterations; they do not prove that any earlier protocol was leakage-free.
The current implementation uses 17 detector features and connected components
of exact normalized signatures and bidirectional connections. It fits
preprocessing on training data and keeps components separate across partitions.

Evidence: [current protocol](EXPERIMENT_PROTOCOL.md) and
[implementation](../src/experiments/protocol.py). Historical partitions remain
local under `dataset/`; do not substitute them for the current saved assignments.

## 3. Comparing neural detectors

CNN, LSTM, CNN-LSTM, BiLSTM and Transformer were evaluated on five shared grouped
folds. The project added group-bootstrap uncertainty estimates and exploratory
statistical comparisons. These results do not establish architecture superiority.

Evidence: [model comparison](../results/research_evaluation_v1/model_comparison.csv)
and [experimental report](reports/EXPERIMENT_REPORT.md).

## 4. Explanations and a final inference candidate

SHAP and LIME analyses were added. A Transformer was selected using validation
loss and refitted on all records. The saved selection record states that the
criterion was formalized after the initial comparison and was not preregistered.
The refitted model has no new independent test score.

Evidence: [selection record](../results/research_evaluation_v1/model_selection.json)
and [explanation figures](../results/research_evaluation_v1/figures/).

## 5. Evidence-grounded triage and paired evaluation

Retrieval and triage demonstrations were followed by a 50-flow matched study.
Both conditions used the same telemetry and fixed CNN predictions. The study
measured generated-report quality, failures and latency. Five retrieval-condition
outputs failed; independent human review remains incomplete. The corpus has
three summaries and all three are retrieved, so this tests evidence inclusion
rather than ranking quality.

Evidence: [paired evaluation findings](reports/EXPERIMENT_REPORT.md#6-paired-retrieval-evaluation).
Raw generations and blank independent-review sheets are retained locally.

## 6. Repository maintenance and research communication

The project separated current source, tests, documentation, selected public
results and ignored local artifacts. Tests and formatting checks are configured
in CI. Run fingerprints and historical source snapshots preserve provenance.
On 30 October 2026, documentation was expanded and role-specific names were
replaced with research-purpose names. Existing measurements were preserved.

Evidence: [migration record](MIGRATION.md), [maintenance guide](../CONTRIBUTING.md)
and [results availability](../results/README.md).

## Current stage: experimental evaluation completed; validation pending

The local detector comparison, explanations, refit and automated paired RAG
study are complete. The project is now at the validation and reproducibility
stage. Priorities are to verify dataset provenance and selection history,
complete independent human adjudication, evaluate external and chronological
holdouts, and document artifact access for a fresh clone. Production readiness
and independently validated end-to-end RAG improvement are not established.

Use the [status ledger](PROJECT_STATUS.md) for completion criteria and future updates.
