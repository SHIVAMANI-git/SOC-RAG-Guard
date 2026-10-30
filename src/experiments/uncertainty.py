"""Cluster-bootstrap intervals conditional on fixed out-of-fold predictions."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import MODELS, save


def main(run, draws=2000):
    manifest = pd.read_csv(run / "manifest.csv")
    groups = manifest.group.to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(42)
    contributions = []
    for name in MODELS:
        p = pd.read_csv(run / f"{name}_oof.csv")
        if not np.array_equal(p.row_id, manifest.row_id):
            raise ValueError("Row mismatch")
        y = p.label.to_numpy()
        z = p.prediction.to_numpy()
        contributions.append(
            np.column_stack(
                [
                    np.bincount(groups, weights=v, minlength=len(unique))
                    for v in [
                        (y == 0) & (z == 0),
                        (y == 0) & (z == 1),
                        (y == 1) & (z == 0),
                        (y == 1) & (z == 1),
                    ]
                ]
            )
        )
    all_counts = np.stack(contributions)
    values = np.empty((draws, len(MODELS)))
    for i in range(draws):
        weights = np.bincount(
            rng.integers(0, len(unique), len(unique)), minlength=len(unique)
        )
        cm = np.einsum("mgc,g->mc", all_counts, weights)
        values[i] = 2 * cm[:, 3] / (2 * cm[:, 3] + cm[:, 1] + cm[:, 2])
    result = {
        "method": "Shared resampling of whole connected-component groups, 2000 draws, seed 42",
        "limitation": "Conditional on fixed OOF models. Does not capture model retraining uncertainty or independent environments. Giant clusters and overlapping training folds limit inference.",
        "f1_cluster95": {},
        "paired_f1_difference_cluster95": {},
    }
    for i, name in enumerate(MODELS):
        result["f1_cluster95"][name] = np.quantile(
            values[:, i], [0.025, 0.975]
        ).tolist()
        for j in range(i + 1, len(MODELS)):
            result["paired_f1_difference_cluster95"][name + " minus " + MODELS[j]] = (
                np.quantile(values[:, i] - values[:, j], [0.025, 0.975]).tolist()
            )
    save(run / "cluster_bootstrap.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    main(args.run)
