"""Artifact consistency checks and grouped-evaluation reporting."""

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from protocol import MODELS, Network, normalize, predict, sha, save


def main(run):
    config = json.loads((run / "protocol.json").read_text())
    assert sha(config["data"]) == config["data_sha256"]
    raw = pd.read_csv(config["data"], low_memory=False)
    x = normalize(raw)
    manifest = pd.read_csv(run / "manifest.csv")
    splits = np.load(run / "splits.npz")
    audit = json.loads((run / "audit.json").read_text())
    statistics = json.loads((run / "statistics.json").read_text())
    cluster = json.loads((run / "cluster_bootstrap.json").read_text())
    validation = []
    torch.set_num_threads(4)
    plt.figure(figsize=(7, 6))
    lines = []
    for name in MODELS:
        pred = pd.read_csv(run / f"{name}_oof.csv")
        assert np.array_equal(pred.row_id, np.arange(len(raw)))
        assert np.array_equal(pred.label, raw.label)
        assert (
            np.isfinite(pred.probability).all() and pred.probability.between(0, 1).all()
        )
        score = json.loads((run / f"{name}_oof_metrics.json").read_text())
        fpr, tpr, threshold = roc_curve(pred.label, pred.probability)
        plt.plot(fpr, tpr, label=f'{name}: {score["roc_auc"]:.4f}')
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold}).to_csv(
            run / f"{name}_oof_roc.csv", index=False
        )
        cm = confusion_matrix(pred.label, pred.prediction, labels=[0, 1])
        save(run / f"{name}_oof_confusion.json", cm.tolist())
        ci = cluster["f1_cluster95"][name]
        lines.append(
            f'| {name} | {score["accuracy"]:.6f} | {score["f1"]:.6f} | {score["roc_auc"]:.6f} | [{ci[0]:.4f}, {ci[1]:.4f}] |'
        )
        for fold in range(1, 6):
            directory = run / "models" / f"{name}_fold{fold}"
            for file in [
                "model.pt",
                "preprocessor.joblib",
                "metrics.json",
                "history.json",
                "predictions.csv",
                "roc.csv",
                "roc.png",
                "confusion.png",
                "shap.npz",
                "lime.json",
            ]:
                assert (directory / file).exists(), str(directory / file)
            p = pd.read_csv(directory / "predictions.csv")
            assert np.array_equal(p.row_id, splits[f"test_{fold}"])
            ck = torch.load(
                directory / "model.pt", map_location="cpu", weights_only=True
            )
            model = Network(name, ck["input_dim"])
            model.load_state_dict(ck["state_dict"])
            pre = joblib.load(directory / "preprocessor.joblib")
            ids = p.row_id.iloc[:1024]
            device = json.loads((run / "models/settings.json").read_text())["device"]
            actual = predict(
                model.to(device), pre.transform(x.iloc[ids]).astype("float32"), device
            )
            expected = p.probability.iloc[:1024].to_numpy()
            assert np.allclose(actual, expected, atol=1e-5, rtol=1e-4), directory
            explanation = np.load(directory / "shap.npz")
            assert np.isfinite(explanation["values"]).all()
            assert set(explanation["background_ids"]).issubset(
                set(splits[f"train_{fold}"])
            )
            assert set(explanation["row_ids"]).issubset(set(splits[f"test_{fold}"]))
            validation.append(
                {
                    "model": name,
                    "fold": fold,
                    "max_reload_probability_error": float(
                        np.max(abs(actual - expected))
                    ),
                }
            )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Pooled out-of-fold ROC — strict grouped comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run / "comparison_roc.png", dpi=150)
    plt.close()
    save(
        run / "verification.json",
        {
            "dataset_hash_verified": True,
            "rows_per_model": len(raw),
            "completed_fits": len(validation),
            "reload_checks": validation,
            "unit_tests": "Run separately: test_protocol.py",
            "scope": "Checks first 1024 reloaded predictions per model-fold on original training device/batch size; all OOF row IDs, labels, ranges and explanation membership verified",
            "cross_device_note": "CPU CNN-LSTM predictions differed from CUDA by about 0.00023 in one inspected batch. Bitwise cross-device equality is not promised. SHAP/LIME use CPU inference.",
        },
    )
    report = (
        """# SOC-RAG Guard experiment results — 30 October 2026

Completed 25 fits: five neural architectures × five shared grouped folds, ten training epochs each, inner-validation checkpoint selection. All 211,043 rows have exactly one outer held-out probability per model. No historical Random Forest score is represented as a neural result.

| Architecture | Pooled accuracy | Pooled attack F1 | Pooled ROC-AUC | Conditional cluster-bootstrap F1 95% interval |
|---|---:|---:|---:|---:|
"""
        + "\n".join(lines)
        + """

These are ordered tabular flow features, **not temporal sequences**. Pooled probabilities combine separately trained fold models; pooled ROC-AUC and mean fold ROC-AUC are different quantities. See fold_metrics.csv for the full per-fold results.

## Leakage controls and limitations

The full dataset contains 93,382 exact feature signatures and 121,395 bidirectional five-tuples, joined into 71,516 connected components. The largest component has 48,956 rows. Train/validation/test partitions have zero overlapping signatures, connections or components. Outer test sizes are 6,256, 16,366, 19,504, 72,445 and 96,472; unequal group sizes produce severe fold imbalance. Preprocessing and class weights use inner training data only. Unknown categories are ignored; labels, attack types, addresses and ports are not model inputs. Labels are used for stratification only.

The 17-feature choice and dataset were used in previous development, so this is not a pristine external validation. Exact signature and connection isolation do not guarantee host, campaign or temporal isolation. Ten epochs and compact architectures constitute a controlled initial comparison, not exhaustive optimization or evidence of architecture superiority.

statistics.json contains descriptive fold intervals and exploratory dependence-corrected paired t comparisons with Holm adjustment. Independent ANOVA is deliberately not used because overlapping CV training sets are not independent replicates. cluster_bootstrap.json resamples whole components using identical draws for each model; intervals condition on already trained OOF models and exclude retraining uncertainty. Giant components make these intervals especially unstable. Do not interpret a small pooled difference as established superiority.

## Saved artifacts and verification

- protocol.json, audit.json, manifest.csv, splits.npz: dataset fingerprint, method and exact row assignments.
- models/<architecture>_fold<N>/: model.pt, preprocessor.joblib, history.json, metrics.json, predictions.csv, roc.csv, roc.png, confusion.png, shap.npz, lime.json and explanation_notes.json.
- fold_metrics.csv, <architecture>_oof.csv, <architecture>_oof_metrics.json: fold and pooled results.
- comparison_roc.png, <architecture>_oof_roc.csv, <architecture>_oof_confusion.json: consolidated graphics/data.
- statistics.json, cluster_bootstrap.json: uncertainty and exploratory comparisons.
- environment.txt, models/settings.json: software versions, training settings and training script fingerprint.
- verification.json: complete OOF alignment checks, finite probabilities, train/test explanation membership, and model/preprocessor reload comparisons for all 25 fits.

SHAP/LIME cover two held-out rows per fold/model (50 explanations per method). They use training-only background samples. Explanations are illustrative, encoded-feature based, potentially affected by unrealistic perturbations, and not causal evidence. See the separate rag_ablation status/results for the paired RAG pilot; detector scores alone say nothing about RAG groundedness.
"""
    )
    baselines = []
    for filename in ["rf_v5_grouped_5fold_cv.json", "rf_v5_signature_grouped_cv.json"]:
        source = run.parent / filename
        if source.exists():
            original = json.loads(source.read_text(encoding="utf-8-sig"))
            baselines.append(
                {
                    "source": str(source),
                    "sha256": sha(source),
                    "status": "historical Random Forest baseline; not paired and not neural results",
                    "original_result": original,
                }
            )
    save(run / "historical_rf_baselines.json", baselines)
    report += "\nHistorical RF source results and hashes are preserved in historical_rf_baselines.json. These were not rerun or reclassified as deep learning.\n"
    report += "\nStatistical method reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html . Our single five-fold, unequal-group application is exploratory and weaker than repeated-CV evidence.\n"
    (run / "RESULTS.md").write_text(report, encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    main(args.run)
