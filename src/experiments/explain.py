"""Kernel SHAP and LIME explanations with training-only background samples."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from protocol import Network, normalize, predict, save, sha


def main(run, count):
    import shap
    from lime.lime_tabular import LimeTabularExplainer

    cfg = json.loads((run / "protocol.json").read_text())
    if sha(cfg["data"]) != cfg["data_sha256"]:
        raise ValueError("Dataset changed")
    x = normalize(pd.read_csv(cfg["data"], low_memory=False))
    splits = np.load(run / "splits.npz")
    rng = np.random.default_rng(42)
    torch.set_num_threads(4)
    for directory in sorted((run / "models").glob("*_fold*")):
        if not (directory / "metrics.json").exists():
            continue
        fold = int(directory.name.rsplit("fold", 1)[1])
        tr = splits[f"train_{fold}"]
        te = splits[f"test_{fold}"]
        bg = rng.choice(tr, min(64, len(tr)), replace=False)
        sample = np.random.default_rng(42 + fold).choice(
            te, min(count, len(te)), replace=False
        )
        pre = joblib.load(directory / "preprocessor.joblib")
        a = pre.transform(x.iloc[bg]).astype("float32")
        b = pre.transform(x.iloc[sample]).astype("float32")
        ck = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model = Network(ck["kind"], ck["input_dim"])
        model.load_state_dict(ck["state_dict"])
        model.eval()

        def prob(z):
            return predict(model, np.asarray(z, dtype="float32"), "cpu")

        np.random.seed(42)
        explanation = shap.KernelExplainer(prob, a).shap_values(b, nsamples=128)
        np.savez_compressed(
            directory / "shap.npz",
            values=explanation,
            row_ids=sample,
            background_ids=bg,
            feature_names=pre.get_feature_names_out(),
        )
        lime = LimeTabularExplainer(
            a,
            feature_names=pre.get_feature_names_out().tolist(),
            class_names=["normal", "attack"],
            random_state=42,
            discretize_continuous=False,
        )
        records = []
        for row_id, row in zip(sample, b):
            exp = lime.explain_instance(
                row,
                lambda z: np.column_stack([1 - prob(z), prob(z)]),
                num_features=10,
                num_samples=512,
            )
            records.append(
                {
                    "row_id": int(row_id),
                    "terms": exp.as_list(),
                    "local_surrogate_r2": float(exp.score),
                }
            )
        save(directory / "lime.json", records)
        save(
            directory / "explanation_notes.json",
            {
                "count": len(sample),
                "background": "64 training rows only",
                "scope": "Encoded feature contributions, not causal effects. Independent perturbations can create invalid one-hot combinations and unrealistic correlated flows. Small samples are illustrative, not global importance.",
                "shap_nsamples": 128,
                "lime_samples": 512,
            },
        )
        print("Explained", directory.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--count", type=int, default=2)
    args = p.parse_args()
    main(args.run, args.count)
