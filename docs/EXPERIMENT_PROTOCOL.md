# Experimental protocol

## Objective

Compare five neural classifiers for binary network-flow classification and evaluate the effect of retrieved evidence on security-triage reports.

## Dataset and features

The source dataset contains 211,043 flow records. The input comprises 14 numerical attributes and three categorical attributes: protocol, service and connection state. Addresses, ports, target labels and attack types are excluded from classifier inputs.

Each record is a tabular observation. The ordering used by convolutional, recurrent and attention-based architectures is a feature ordering; it does not represent a chronological traffic sequence.

## Partitioning

Records belong to the same connected component when they share either an exact normalized feature signature or a bidirectional five-tuple. Components remain intact across training, validation and test partitions. Labels are used for stratification, not for group construction.

Five outer grouped folds use seed 42. Each outer development partition supplies one predefined inner grouped validation partition. All architectures share the saved row assignments. Numerical imputation, standardization and categorical encoding are fitted on inner training data. Positive-class weighting uses training labels only.

The grouping constraints produce unequal folds. Isolation of feature signatures and connections does not establish independence between hosts, campaigns or environments. The dataset was used in earlier development and is not an external validation set.

## Models and optimization

| Architecture | Representation |
|---|---|
| 1D-CNN | Convolutions over the encoded feature vector |
| LSTM | Unidirectional recurrent processing of encoded features |
| CNN-LSTM | Convolutional projection followed by recurrent processing |
| BiLSTM | Bidirectional recurrent processing of encoded features |
| Transformer Encoder | Feature embeddings, positional parameters and self-attention |

The common training configuration uses AdamW, learning rate 0.001, batch size 512 and ten epochs. Minimum inner-validation binary cross-entropy selects the saved checkpoint. The classification threshold is 0.5. Outer test partitions are excluded from checkpoint selection.

The final architecture is selected using mean inner-validation loss. Its refit epoch count is the median selected epoch across folds. Refitting uses all available records and therefore provides no additional independent test estimate.

## Performance analysis

Reported measures are accuracy, attack precision, recall, F1, balanced accuracy, ROC-AUC and confusion counts. Saved probabilities retain original row IDs. Pooled out-of-fold measures and equal-weight fold means are reported separately where applicable.

F1 is the primary comparison measure. Pairwise tests use an approximate cross-validation dependence correction and Holm adjustment. Whole-component bootstrap intervals condition on fixed out-of-fold predictions and exclude retraining uncertainty. Fold-block repeated-measures ANOVA includes a Greenhouse-Geisser correction and is reported as exploratory. Overlapping training sets limit all fold-based inferential claims.

Historical Random Forest results serve as supplementary baselines. Different historical partitions preclude paired significance comparisons with the neural study.

## Feature explanations

Kernel SHAP and LIME each cover 20 held-out records per fold and architecture: 100 records per architecture, 500 explanations per method. Background samples contain training records only. Encoded categorical contributions are aggregated to their original feature before absolute importance is calculated.

The summaries describe the selected examples. Independent perturbations may produce unrealistic feature combinations, and LIME fidelity varies between examples. Local surrogate R2 accompanies the saved explanations; low-fidelity estimates require cautious interpretation. Importance is not a causal effect.

## Paired RAG evaluation

The extended study selects ten distinct groups from each outer fold using seed 20261030. Groups from the original ten-pair pilot are excluded. Selection precedes generation and is independent of detector correctness. The first record represents each selected group.

Both conditions receive identical telemetry and the same prespecified CNN out-of-fold probability. Generation model, system prompt, response schema, temperature, seed and 512-token limit are shared. Arm order is randomized. Retrieved evidence is the intervention. Warmup is excluded from timing; retrieval, generation and end-to-end durations are retained.

The frozen corpus contains three local reference summaries. Top-k equals three, so the study evaluates evidence inclusion rather than retrieval-ranking quality. Source-ID compliance, citation entailment and factual support are distinct measures.

Generation failures remain in success-rate and latency calculations. Complete-pair quality covers pairs with two valid reports. A separate usability-adjusted measure assigns zero supported-claim reward to failed generations and retains all attempted pairs. Claim fractions are undefined for failed reports.

Automated support and recommendation assessments are diagnostic estimates. The judge uses the same model family as the generator and has known interpretation errors. Independent human adjudication remains pending. The supplied review sheets, frozen evidence and blinded packet support that process.

## Review categories

| Label | Definition |
|---|---|
| Supported | Every factual component is supported by the supplied evidence or telemetry |
| Partial | Supported and unsupported factual components occur in the same claim |
| Unsupported | The assertion lacks evidential support or contradicts the supplied information |
| Unverifiable | The available information does not permit a decision |

Citation entailment is assessed separately. General guidance cannot establish a specific flow probability or packet count. A request to investigate a possibility does not establish that the event occurred and should not be treated as such during recommendation review.

## Reproducibility

Saved records include dataset and source fingerprints, row assignments, seeds, software versions, preprocessing artifacts, checkpoints, probabilities, raw generation responses and review decisions. Current commands are in [RUNBOOK.md](RUNBOOK.md). Earlier pilot scripts and fixed annotations are retained under `archive/pilot_evaluation/`; original result records retain their historical names and provenance.

## References

- [Grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Statistical comparison of cross-validation results](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html)
- [Repeated-measures ANOVA](https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html)
