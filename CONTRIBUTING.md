# Contribution and maintenance guide

## Folder responsibilities

Put current experiment code in `src/experiments/`, retrieval/triage components
in `rag/`, regression checks in `tests/`, and research documentation in `docs/`.
Keep the root README focused on orientation, data, status and entry points.
Update the corresponding folder index when adding a workflow.

Datasets belong in ignored `dataset/`; environments in `.venv/`; historical
snapshots in ignored `archive/`. Store new experiment outputs in a new
`results/<descriptive_run_name>/` directory. Do not overwrite completed runs or
move evidence without checking embedded paths and documenting the migration.

## Evidence and reproducibility

Record dataset fingerprints, source revision, configuration, seeds, splits,
software environment and model-selection rules for each new experiment.
Keep training, validation and test roles separate. Report failed generations
alongside successful ones and label automated review explicitly.

The results allowlist in `.gitignore` deliberately publishes only selected
summaries and figures. Before adding a new summary, explicitly review its
allowlist entry and confirm it is tracked. Keep large raw outputs, data and
serialized models local unless a deliberate distribution plan is established.
Never force-add the dataset to make reproduction instructions work.

## Validate a change

From the repository root after installing `requirements-dev.txt`:

```powershell
.\.venv\Scripts\python.exe project.py test
.\.venv\Scripts\python.exe -m black --check project.py src tests rag
git diff --check
git status --short
git diff --stat
```

For documentation changes, verify paths both on disk and in Git: a local file
can exist but be absent from a fresh clone. Mark local-only artifacts as such.
For code changes, run relevant offline tests and validate changed workflows in
a new output directory. Record checks and material limitations in the change
summary. Do not treat an existing result table as evidence of a newly run test.

## Keep research claims current

Update `docs/PROJECT_STATUS.md` when completion evidence becomes available.
Update the report and README together when scientific conclusions change.
Do not invent dataset attribution, selection rationale, licensing or external
validation. Keep old measurements intact and distinguish new results by run.
Review the diff before committing; stage only intended files.
