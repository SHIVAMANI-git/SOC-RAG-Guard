# Experiment source index

| Module | Role |
|---|---|
| `protocol.py` | Feature definitions, normalization, grouping, fold preparation, model training and summaries |
| `uncertainty.py` | Uncertainty analysis |
| `explain.py` | Feature explanations |
| `research_analysis.py` | Consolidated analyses and final-model preparation |
| `generate_report.py` | Consolidated reporting |
| `rag_ablation.py` | Controlled with/without-evidence experiment |
| `extended_rag.py` | Extended generation and automated judging |
| `reviewer_agreement.py` | Agreement checks for completed independent reviews |
| `predict_final.py` | Batch inference using the local final bundle |
| `verify_and_report.py` | Experiment verification and reporting |

Use [the runbook](../../docs/RUNBOOK.md) for commands and
[the protocol](../../docs/EXPERIMENT_PROTOCOL.md) for scientific settings.
Inspect a module's `--help` before running it. Experiments need local artifacts;
use new output directories to preserve completed results.
