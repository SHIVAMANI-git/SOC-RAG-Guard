# Folder reorganization

The repository was reorganized on 30 October 2026. The exact move list, backup location and preservation checks are in `reorganization_verification.json` beside this file.

| Previous path | Current location |
|---|---|
| `experiments_v1/*.py` | `src/experiments/*.py` |
| `experiments_v1/test_*.py` | `tests/test_*.py` |
| `experiments_v1/README.md` | Protocol in `docs/EXPERIMENT_PROTOCOL.md`; short pointer beside source |
| `src/*.py` | `archive/legacy_scripts/*.py` |
| `folder for review/` | `archive/review_bundle/` |
| `folder for review.zip` | `archive/review_bundle.zip` |
| Empty `code/`, `report/` | `archive/unused_folders/code/`, `archive/unused_folders/report/` |
| Earlier experiment bytecode cache | `archive/runtime_cache/experiments_before_reorganization/` |

At that initial reorganization, `dataset/`, `results/`, `rag/` and `.venv/` retained their existing paths. The later naming cleanup below supersedes the evaluation-result folder name. Every pre-existing dataset and result file was checked by SHA256 before and after the moves. No saved experiment measurement was recomputed as part of this cleanup.

Executable source roots were adjusted for the new folder depth. Tests now explicitly discover the canonical experiment modules. `project.py` provides a stable command entry point and runs work in the existing virtual environment.

Historical result-folder runbooks, code hashes and submitted ZIPs are kept unchanged. They describe the layout used for the original run; they are not current navigation instructions. Updating those in place would invalidate their artifact hashes. Use `RUNBOOK.md` and the root README for the current layout. New source may differ from saved source fingerprints because of path/documentation changes; the exact pre-reorganization source is backed up.

The backup ZIP contains original code and review files, not duplicate copies of the large dataset or saved results. Its checksum manifest records original source files. The data/results preservation manifest is stored alongside it. Review the manifests before any future manual rollback; do not overwrite newer work blindly.

## Research naming cleanup (30 October 2026)

| Previous name | Current name |
|---|---|
| `src/experiments/coordinator_completion.py` | `src/experiments/research_analysis.py` |
| `src/experiments/completion_report.py` | `src/experiments/generate_report.py` |
| `tests/test_completion.py` | `tests/test_research_analysis.py` |
| `results/coordinator_completion_v1/` | `results/research_evaluation_v1/` |
| `COORDINATOR_REPORT.md` | `RESEARCH_REPORT.md` within that run |
| `coordinator_submission.zip` | `research_artifacts.zip` within that run (local-only) |

Imports, launcher defaults, generated report names, public documentation and
Git ignore rules were updated together. The current report and runbook received
navigation edits only; original copies and a source/documentation snapshot are
stored locally in `archive/naming_cleanup_20261030/`. Scientific artifacts were
fingerprinted before and after the rename. The local preservation manifest is
`archive/naming_cleanup_20261030/preservation_check.json`.

Historical JSON paths, source hashes and ZIP contents remain unchanged to
preserve their meaning. Translate old paths with this table when inspecting a
past run. Old terminology is retained only where needed for historical provenance;
current entry points use the new names. Do not rerun an experiment into the
completed evaluation folder to update names.
