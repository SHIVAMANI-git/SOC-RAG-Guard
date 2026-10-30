# Project status

Reviewed: 30 October 2026. This ledger separates completed local experiments
from artifacts available in Git. Completion does not imply deployment readiness.

## Completed work

| Area | Verified scope | Evidence |
|---|---|---|
| Data inspection | 211,043 records; 50,000 normal; 161,043 attack; 17 detector features | [Dataset description](../README.md#dataset-description), local source CSV |
| Preprocessing and partitioning | Training-only preprocessing; five grouped outer folds with inner validation; signature and bidirectional connection isolation | [Protocol implementation](../src/experiments/protocol.py), [methodology](EXPERIMENT_PROTOCOL.md) |
| Detector comparison | CNN, LSTM, CNN-LSTM, BiLSTM and Transformer evaluated | [Comparison table](../results/research_evaluation_v1/model_comparison.csv) |
| Statistical analysis | Group-bootstrap intervals and exploratory corrected comparisons | [Report](reports/EXPERIMENT_REPORT.md), [paired tests](../results/research_evaluation_v1/corrected_t_tests.csv) |
| Explanations | SHAP and LIME: 100 held-out records per architecture per method | [Report](reports/EXPERIMENT_REPORT.md), [figures](../results/research_evaluation_v1/figures/) |
| Final detector | Validation-selected Transformer refitted on full data; saved reload/inference verification | [Selection](../results/research_evaluation_v1/model_selection.json), [verification](../results/research_evaluation_v1/final_model/verification.json); weights/preprocessor local-only |
| RAG comparison | 50 matched flows, fixed CNN predictions; automated report evaluation; generation failures retained | [Report, section 6](reports/EXPERIMENT_REPORT.md#6-paired-retrieval-evaluation) |
| Engineering checks | Offline tests and formatting workflow provided | [Tests](../tests/), [CI workflow](../.github/workflows/quality.yml) |

## Pending work and completion criteria

| Priority | Work | Completion evidence needed |
|---|---|---|
| High | Confirm dataset origin and selection history | Original publisher, release, source URL/citation, license, acquisition and sampling details; author-confirmed dataset/feature rationale |
| High | Independent human adjudication | Completed independent review sheets, agreement results and adjudicated claim/recommendation scores; automated labels do not qualify |
| High | External validation | Evaluation on an independently sourced environment with training decisions frozen and results documented |
| High | Temporal validation | Verified timestamps/collection chronology and leakage-aware future-time holdout; current feature ordering is not temporal |
| Medium | Resolve RAG output truncation | New separately recorded experiment with valid-output rates, quality and latency; preserve the five failures in the original study |
| Medium | Evaluate retrieval ranking | A larger justified evidence corpus and relevance evaluation; the current study retrieves all three local summaries |
| Medium | Improve clean-clone reproducibility | Verified dataset acquisition instructions and a documented compatible model-bundle distribution or regeneration procedure with fingerprints |

## Interpretation limits

The dataset was used in earlier development. Grouping does not prove host,
campaign or environment independence. The full-data refit has no independent
test estimate. Statistical comparisons do not establish superiority or
equivalence. Automated RAG judgments require human validation, and the
failure-adjusted interval includes zero. Retrieval does not modify the detector
predictions in the paired study. Production readiness is not established.

## Updating this ledger

Update this page when a new result is verified. Link the evidence, distinguish
local artifacts from tracked ones, and keep incomplete work pending until its
completion criterion is met. Preserve historical experiment measurements.

For the starting point, development stages and maintenance history, read the [development history](DEVELOPMENT_HISTORY.md).
