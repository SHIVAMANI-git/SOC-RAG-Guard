# Retrieval and triage

This directory retains the existing retrieval/triage implementation, its knowledge-base files and earlier evaluation outputs. Paths are unchanged.

- `retriever.py`: local embedding retrieval.
- `triage.py`: evidence-grounded triage generation.
- `knowledge_base/`: local evidence summaries and original embedding cache.
- `evaluation/`: earlier reports and claim-review files.

The controlled paired RAG experiment is in `../src/experiments/rag_ablation.py` and `extended_rag.py`. Current completed outputs are in `../results/research_evaluation_v1/extended_rag/rag_ablation/`.

Generation requires the local Ollama service and configured models. Read `../docs/RUNBOOK.md` for current commands and limitations. The corpus contains locally authored summaries; source IDs alone do not establish claim entailment.

The evidence JSON is tracked; the embedding cache and generated evaluation
outputs are local-only. For public result links, see the [results index](../results/README.md).
For workflow context, see the [architecture](../docs/ARCHITECTURE.md).
