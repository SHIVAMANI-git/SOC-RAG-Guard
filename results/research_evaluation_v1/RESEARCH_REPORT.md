# SOC-RAG Guard â€” research evaluation report

Date: 30 October 2026. All Python execution used the repository `.venv`. This report presents observed results; it does not assume that RAG must improve quality.

## Completion checklist

| Requested work | Delivered |
|---|---|
| Five deep models | CNN, LSTM, CNN-LSTM, BiLSTM, Transformer; 25 shared grouped outer-fold fits |
| SHAP and LIME | Each method: 100 held-out examples per architecture, 500 total; raw values, original-feature summaries and plots |
| Finalize models | All comparison checkpoints retained; validation-selected Transformer refitted on all data and tested for reload/inference |
| ANOVA | Fold-block repeated-measures ANOVA, Greenhouse-Geisser correction, explicitly exploratory |
| t-test, p-values, intervals | CV-dependence-corrected paired t, Holm correction, fold intervals, group-bootstrap intervals; paired RAG intervals and exploratory sign-flip tests |
| ROC and confusion matrices | All 25 fold plots plus pooled ROC and combined confusion matrices |
| With/without RAG | 50 fresh matched group-sampled flows, 100 generation attempts; actual failures, latency, citations, automated quality and recommendation audits |
| Independently prove RAG quality enhancement | Not established by this automated study; human review materials supplied and limitations reported |

## Neural comparison

| Model | Accuracy | Attack F1 | ROC-AUC | Conditional cluster F1 95% interval |
|---|---:|---:|---:|---|
| cnn | 0.8383 | 0.8897 | 0.8366 | [0.8297, 0.9569] |
| lstm | 0.8340 | 0.8857 | 0.8238 | [0.8235, 0.9534] |
| cnn_lstm | 0.8398 | 0.8905 | 0.8710 | [0.8272, 0.9633] |
| bilstm | 0.8414 | 0.8904 | 0.8685 | [0.8301, 0.9580] |
| transformer | 0.8432 | 0.8925 | 0.8288 | [0.8320, 0.9601] |

The input is an ordered vector of 17 tabular flow features, not a chronological traffic sequence. IPs, ports, target label and attack type are excluded from detector inputs. Exact feature signatures and bidirectional connection groups are kept together transitively. All preprocessing fits inside the training partition. The dataset has 211,043 rows; severe group imbalance causes unequal folds and limits inference. Existing Random Forest values remain historical baselines only.

Exploratory F1 repeated-measures ANOVA: F(4,16)=1.1642; uncorrected p=0.363221; Greenhouse-Geisser p=0.354919. Overlapping training folds violate independent-replicate assumptions; the correction for sphericity does not resolve CV dependence. Corrected paired F1 comparisons in the original run found no Holm-adjusted significant winner. These results do not prove equivalence.

Final selection: transformer, mean inner-validation BCE=0.205369, refit for 8 epochs on all rows. Selection used validation scores, not outer test ranking. The selection rule was formalized after the initial experiment and is not preregistered. The final refit has no new independent test performance estimate.

## Feature explanations

Top five transformer SHAP features in the fixed held-out sample: proto, conn_state, src_pkts, service, dns_rcode.

Categorical one-hot contributions are summed back to their original feature before taking absolute importance. LIME aggregation includes its selected top-10 encoded terms; absent terms are treated as zero and cancellation can affect grouped ranks. These are sample summaries, not population-level or causal effects. Perturbations can violate one-hot/correlation structure. LIME fidelity must be checked before interpreting an example: mean local R2 is bilstm=0.159, cnn=0.354, cnn_lstm=0.185, lstm=0.150, transformer=0.137. Low-fidelity examples are illustrative only.

## Extended RAG comparison

Ten new distinct groups per fold, sampled with seed 20261030 before generating reports, excluding all original pilot groups. Each arm receives identical held-out telemetry and the same fixed CNN OOF probability. The model, system prompt, schema, temperature, seed and 512-token cap are shared; arm order is randomized. Only reference evidence differs. Results describe a uniform group sample with the first row representing each selected group, not a row-weighted deployment population.

| Measure | With RAG | Without RAG |
|---|---:|---:|
| Valid generations / attempts | 45 / 50 | 50 / 50 |
| Mean latency, seconds, all attempts | 4.309 | 2.900 |
| Fully supported claim fraction, complete pairs, automated | 88.96% | 73.96% |
| Partially supported fraction, complete pairs, automated | 3.33% | 0.74% |
| Fully unsupported fraction, complete pairs, automated | 7.70% | 25.30% |
| Not fully supported fraction, complete pairs, automated | 11.04% | 26.04% |
| Failure-adjusted supported-claim reward | 80.07% | 72.60% |

Complete-pair automated groundedness difference (with minus without): 0.1500; paired bootstrap 95% interval [0.08407407407407409, 0.2159351851851851]; exploratory sign-flip p=0.000100, Holm p=0.000500. Only 45 pairs have two valid reports. Complete-pair quality is subject to survivor bias; the separate failure-adjusted reward retains all 50 pairs.

All-pair failure-adjusted reward difference: 0.0747; paired bootstrap 95% interval [-0.013666666666666652, 0.159]. A failed report receives zero usability reward, not an invented unsupported-claim fraction.

Quality figures are automated proxies from the same Qwen model family as the generator. They are not independent human adjudication. The revised judge separates telemetry support from reference entailment; raw decisions and validation rules are saved. The original failed pilot judge remains preserved. Citation entailment is assessed only for reference text that was actually retrieved; no-RAG field-name references are not supplied corpus IDs, so ID compliance is not a neutral standalone measure of factual quality.

Some reports hit the shared token cap and returned incomplete JSON. These attempts are counted as failures and retained with raw responses. Both arms were tested under the same cap. No failed generation was selectively repaired.

The frozen corpus has only three local summaries, and top-k=3 returns the entire corpus. This experiment measures evidence inclusion, not retrieval ranking. It does not independently establish the original documents' correctness. Source revisions, environment shifts and external traffic remain untested. RAG does not alter detector decisions, so detector classification accuracy is unchanged by design.

A recommendation-specific automated audit is also provided, because unsupported specificity may occur outside the claims array. Human validation should review both sections.

## Independent review and reproducibility

Two blank reviewer sheets and a blinded report packet are in extended_rag/rag_ablation/. Actual human labels are intentionally unfilled. reviewer_agreement.py checks coverage and computes agreement/kappa after real reviews, producing a disagreement sheet for adjudication. No human agreement statistic has been invented.

Current reproduction commands and final-model inference are in [the project runbook](../../docs/RUNBOOK.md). Tests are recorded in tests.txt. Original grouped CV files remain under ../comparison_v1/. Artifact hashes and the research artifact archive accompany this report.

Statistical reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html ; dependence-aware comparison reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html .

Automated audit caution: spot-checking found a false-positive pattern in which a request to verify threat intelligence was judged unjustified because intelligence was not already supplied. Such recommendations can be reasonable validation steps. Claim support can also be overgenerous for compound interpretations. Both support and recommendation verdict counts are diagnostic estimates, not validated quality measurements. Review the original texts before making a scientific claim from those counts.

The RAG experiment intentionally retains the prespecified CNN OOF detector for comparability. It does not evaluate the new full-data Transformer refit plus RAG as an independently tested end-to-end deployed system.

Navigation names were updated on 30 October 2026; scientific measurements were not recomputed. Historical fingerprints and archive contents retain original run names. See [migration notes](../../docs/MIGRATION.md).
