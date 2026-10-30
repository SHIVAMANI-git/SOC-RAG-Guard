# Results and evidence index

## Included in GitHub

| Artifact | What it supports |
|---|---|
| [Experimental report](../docs/reports/EXPERIMENT_REPORT.md) | Main detector and with/without-RAG findings, denominators and limitations |
| [Model comparison](research_evaluation_v1/model_comparison.csv) | Out-of-fold classifier metrics and group-bootstrap F1 intervals |
| [Model selection](research_evaluation_v1/model_selection.json) | Validation-based final-model choice |
| [ANOVA](research_evaluation_v1/anova.csv) and [corrected paired tests](research_evaluation_v1/corrected_t_tests.csv) | Exploratory statistical comparisons |
| [Feature importance](research_evaluation_v1/feature_importance.csv) and [LIME fidelity](research_evaluation_v1/lime_fidelity.csv) | Explanation summaries and surrogate fidelity |
| [Figures](research_evaluation_v1/figures/) | Classifier, explanation and RAG plots |
| [Final-model verification](research_evaluation_v1/final_model/verification.json) | Saved local verification record; not model weights |
| [Detailed research report](research_evaluation_v1/RESEARCH_REPORT.md) | Historical consolidated report; paths may predate migration |

For **with RAG versus without RAG**, start with section 6 of the experimental
report and the `rag_groundedness.png`, `rag_effect_intervals.png` and
`rag_latency.png` figures. Classification predictions were fixed across both
conditions; these comparisons measure report quality, failures and latency.

## Local-only artifacts (not available in a fresh clone)

- `results/comparison_v1/`: original fold assignments, predictions and checkpoints.
- `results/research_evaluation_v1/extended_rag/rag_ablation/`: raw paired outputs, automated audits and independent-review materials.
- Model weights and serialized preprocessing objects under `final_model/`.
- Submission ZIPs, intermediate outputs and historical Random Forest results.

Historical Random Forest results use a different protocol and are supplementary;
they are not paired architecture comparisons with the neural experiments.
Local artifact paths are documented for orientation, not offered as GitHub downloads.
See the [runbook](../docs/RUNBOOK.md) and [status ledger](../docs/PROJECT_STATUS.md)
for reproduction requirements and remaining gaps.
