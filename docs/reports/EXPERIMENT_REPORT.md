# SOC-RAG Guard: Experimental Evaluation

Study date: 30 October 2026

## 1. Study design

Five neural architectures were evaluated on five shared grouped partitions of 211,043 network-flow records. Exact feature signatures and bidirectional connections were isolated across training, validation and test sets. Preprocessing was fitted on training data. Each input is an ordered vector of tabular attributes; it is not a chronological traffic sequence.

## 2. Classifier performance

| Architecture | Accuracy | Attack F1 | ROC-AUC | F1 bootstrap 95% interval |
|---|---:|---:|---:|---|
| cnn | 0.8383 | 0.8897 | 0.8366 | [0.8297, 0.9569] |
| lstm | 0.8340 | 0.8857 | 0.8238 | [0.8235, 0.9534] |
| cnn_lstm | 0.8398 | 0.8905 | 0.8710 | [0.8272, 0.9633] |
| bilstm | 0.8414 | 0.8904 | 0.8685 | [0.8301, 0.9580] |
| transformer | 0.8432 | 0.8925 | 0.8288 | [0.8320, 0.9601] |

The intervals resample connected components and condition on fixed out-of-fold predictions. They do not include model-retraining uncertainty. Large components and unequal folds limit generalization.

## 3. Statistical comparisons

The exploratory repeated-measures F1 comparison yielded F(4,16)=1.1642, p=0.363221, and Greenhouse-Geisser-adjusted p=0.354919. Corrected pairwise t-tests found no Holm-adjusted significant F1 difference. These results neither establish architecture superiority nor demonstrate equivalence.

Overlapping training sets limit independent-replicate assumptions. The ANOVA is a descriptive supplement to the dependence-corrected paired comparisons. Full test statistics, p-values and intervals are retained in the result tables.

## 4. Final model

The transformer was selected using mean inner-validation binary cross-entropy (0.205369) and refitted for 8 epochs on the full dataset. Reload and batch-inference checks passed. The full-data refit has no additional independent test estimate.

## 5. Feature explanations

SHAP and LIME each cover 100 held-out examples per architecture. Training-only background samples were used. Encoded categorical contributions were grouped by original feature.

The highest SHAP contributions for the selected architecture in this sample were proto, conn_state, src_pkts, service, dns_rcode. Local surrogate fidelity varies, and low-R2 LIME estimates are illustrative. Independent perturbations can produce unrealistic feature combinations. These explanations do not establish causality.

## 6. Paired retrieval evaluation

Fifty new matched flows were selected from distinct groups, with ten groups per fold. Original pilot groups were excluded. Both conditions received the same telemetry, fixed CNN out-of-fold probabilities and generation settings. Reference evidence was the intervention.

| Measure | With retrieval | Without retrieval |
|---|---:|---:|
| Valid reports / attempts | 45 / 50 | 50 / 50 |
| Mean end-to-end seconds | 4.309 | 2.900 |
| Supported claims, automated assessment of complete pairs | 88.96% | 73.96% |
| Unsupported claims, automated assessment of complete pairs | 7.70% | 25.30% |
| Partially supported claims, automated assessment of complete pairs | 3.33% | 0.74% |

Among 45 complete pairs, the automated supported-claim difference was 15.00 percentage points; its paired bootstrap 95% interval was [8.41, 21.59]. The exploratory Holm-adjusted sign-flip p-value was 0.000500.

Including generation failures as zero usability reward, the difference was 7.47 percentage points, with a 95% interval of [-1.37, 15.90]. This interval includes zero.

Automated estimates are subject to judge error and self-evaluation bias. Spot checks identified overgenerous claim-support judgments and cases where valid investigation requests were misclassified. Independent human adjudication is pending. The observed complete-pair result therefore does not establish independently validated end-to-end enhancement.

The common 512-token limit caused five retrieved-evidence reports to terminate with incomplete JSON. Failures were retained. The corpus contains three local summaries and all three were retrieved, so the study evaluates evidence inclusion rather than retrieval ranking.

## 7. Limitations

- Group imbalance, prior dataset use and absence of external validation limit deployment claims.
- Feature-vector processing does not demonstrate temporal traffic modelling.
- Explanation fidelity and perturbation realism vary between records.
- Quality assessment requires independent review of claims and recommendations.
- Classification predictions are unchanged by retrieval in this experiment.

## 8. Reproducibility and artifacts

Execution instructions: [RUNBOOK.md](../RUNBOOK.md). Methodology: [EXPERIMENT_PROTOCOL.md](../EXPERIMENT_PROTOCOL.md).

Original metrics, checkpoints, plots, responses and review records are under `results/comparison_v1/` and `results/research_evaluation_v1/`. Those records retain their original contents. Current source is under `src/experiments/`; superseded pilot material is archived.

## References

- [Grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Statistical comparison](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
- [Repeated-measures ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html)
