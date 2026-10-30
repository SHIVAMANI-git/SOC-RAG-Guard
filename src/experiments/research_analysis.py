"""Statistical comparison, model selection, and full-data model fitting."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from protocol import MODELS, NUM, CAT, FEATURES, Network, normalize, predict, save, sha


def rm_anova(values):
    """One-factor repeated-measures ANOVA with Greenhouse-Geisser correction."""
    a = np.asarray(values, dtype=float)
    n, k = a.shape
    if n < 3 or k < 2 or not np.isfinite(a).all():
        raise ValueError("Invalid balanced repeated-measures matrix")
    mean = a.mean()
    ss_model = n * np.sum((a.mean(0) - mean) ** 2)
    residual = a - a.mean(1, keepdims=True) - a.mean(0, keepdims=True) + mean
    ss_error = np.sum(residual**2)
    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    f = (ss_model / df1) / (ss_error / df2)
    c = np.eye(k) - np.ones((k, k)) / k
    s = c @ np.cov(a, rowvar=False) @ c
    epsilon = float(
        np.clip(np.trace(s) ** 2 / ((k - 1) * np.trace(s @ s)), 1 / (k - 1), 1)
    )
    return {
        "F": float(f),
        "df_model": df1,
        "df_error": df2,
        "p_uncorrected": float(stats.f.sf(f, df1, df2)),
        "greenhouse_geisser_epsilon": epsilon,
        "df_model_GG": df1 * epsilon,
        "df_error_GG": df2 * epsilon,
        "p_GG": float(stats.f.sf(f, df1 * epsilon, df2 * epsilon)),
    }


def source_feature(name):
    if name.startswith("num__"):
        return name[5:]
    for field in CAT:
        if name.startswith("cat__" + field + "_"):
            return field
    raise ValueError("Unknown encoded feature " + name)


def analysis(run, out):
    out.mkdir(parents=True, exist_ok=True)
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    table = pd.read_csv(run / "fold_metrics.csv")
    anovas = {}
    for metric in ["f1", "roc_auc", "balanced_accuracy"]:
        a = table.pivot(index="fold", columns="model", values=metric)[MODELS]
        anovas[metric] = rm_anova(a.to_numpy())
        friedman = stats.friedmanchisquare(*[a[m].to_numpy() for m in MODELS])
        anovas[metric]["friedman_sensitivity"] = {
            "statistic": float(friedman.statistic),
            "p": float(friedman.pvalue),
        }
    save(
        out / "anova.json",
        {
            "analysis": "Repeated-measures ANOVA: architecture factor, fold block; Greenhouse-Geisser correction",
            "status": "Exploratory descriptive supplement, NOT confirmatory evidence. Fold training sets overlap, so even blocked ANOVA cannot establish independent replication. Only five folds; sphericity correction does not fix CV dependence.",
            "results": anovas,
            "reference": "https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html",
        },
    )
    pd.DataFrame(
        [
            {
                "metric": m,
                **{k: v for k, v in result.items() if not isinstance(v, dict)},
            }
            for m, result in anovas.items()
        ]
    ).to_csv(out / "anova.csv", index=False)
    old = json.loads((run / "statistics.json").read_text())
    pd.DataFrame(old["paired_corrected_t_holm"]).to_csv(
        out / "corrected_t_tests.csv", index=False
    )
    all_importance = []
    fidelity = []
    for kind in MODELS:
        shap_records = []
        lime_records = []
        for fold in range(1, 6):
            explanation_root = (
                out / "explanations" if (out / "explanations/models").exists() else run
            )
            directory = explanation_root / "models" / f"{kind}_fold{fold}"
            exp = np.load(directory / "shap.npz")
            names = joblib.load(
                directory / "preprocessor.joblib"
            ).get_feature_names_out()
            values = exp["values"]
            if values.ndim == 3:
                values = values.squeeze(-1)
            for row_id, row in zip(exp["row_ids"], values):
                grouped = {f: 0.0 for f in FEATURES}
                for name, value in zip(names, row):
                    grouped[source_feature(str(name))] += float(value)
                for feature, value in grouped.items():
                    shap_records.append(
                        {
                            "method": "SHAP",
                            "model": kind,
                            "fold": fold,
                            "row_id": int(row_id),
                            "feature": feature,
                            "absolute_contribution": abs(value),
                        }
                    )
            for exp_lime in json.loads((directory / "lime.json").read_text()):
                grouped = {f: 0.0 for f in FEATURES}
                for name, value in exp_lime["terms"]:
                    grouped[source_feature(name)] += float(value)
                for feature, value in grouped.items():
                    lime_records.append(
                        {
                            "method": "LIME",
                            "model": kind,
                            "fold": fold,
                            "row_id": exp_lime["row_id"],
                            "feature": feature,
                            "absolute_contribution": abs(value),
                        }
                    )
                fidelity.append(
                    {
                        "model": kind,
                        "fold": fold,
                        "row_id": exp_lime["row_id"],
                        "local_R2": exp_lime["local_surrogate_r2"],
                    }
                )
        for method, records in [("SHAP", shap_records), ("LIME", lime_records)]:
            frame = pd.DataFrame(records)
            all_importance.extend(records)
            importance = (
                frame.groupby("feature").absolute_contribution.mean().sort_values()
            )
            plt.figure(figsize=(8, 6))
            importance.plot.barh(color="#267a9c" if method == "SHAP" else "#b86736")
            plt.xlabel("Mean absolute grouped contribution")
            plt.title(f"{kind}: {method}, {frame.row_id.nunique()} held-out examples")
            plt.tight_layout()
            plt.savefig(fig / f"{kind}_{method.lower()}_importance.png", dpi=160)
            plt.close()
    values = pd.DataFrame(all_importance)
    values.to_csv(out / "feature_contributions.csv", index=False)
    ranks = (
        values.groupby(["method", "model", "feature"])
        .absolute_contribution.mean()
        .reset_index()
    )
    ranks.to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(fidelity).to_csv(out / "lime_fidelity.csv", index=False)
    for method in ["SHAP", "LIME"]:
        a = ranks[ranks.method == method].pivot(
            index="feature", columns="model", values="absolute_contribution"
        )[MODELS]
        a = a.div(a.sum(axis=0), axis=1)
        plt.figure(figsize=(8, 8))
        plt.imshow(a, aspect="auto", cmap="YlGnBu")
        plt.xticks(range(5), MODELS, rotation=25)
        plt.yticks(range(len(a)), a.index)
        plt.colorbar(label="Within-model normalized absolute contribution")
        plt.title(f"{method}: illustrative feature importance")
        plt.tight_layout()
        plt.savefig(fig / f"{method.lower()}_comparison.png", dpi=160)
        plt.close()
    fig_cm, axes = plt.subplots(2, 3, figsize=(14, 8))
    for kind, ax in zip(MODELS, axes.flat):
        pred = pd.read_csv(run / f"{kind}_oof.csv")
        cm = confusion_matrix(pred.label, pred.prediction, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(
            ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title(kind)
    axes.flat[-1].axis("off")
    fig_cm.suptitle("Pooled grouped out-of-fold confusion matrices")
    fig_cm.tight_layout()
    fig_cm.savefig(fig / "confusion_matrices.png", dpi=170)
    plt.close(fig_cm)
    decisions = []
    for kind in MODELS:
        losses = []
        epochs = []
        for fold in range(1, 6):
            history = json.loads(
                (run / "models" / f"{kind}_fold{fold}" / "history.json").read_text()
            )
            best = min(history, key=lambda row: row["validation_bce"])
            losses.append(best["validation_bce"])
            epochs.append(best["epoch"])
        decisions.append(
            {
                "model": kind,
                "mean_inner_validation_BCE": float(np.mean(losses)),
                "selected_epochs": epochs,
                "final_epochs": int(np.median(epochs)),
            }
        )
    decisions.sort(key=lambda row: row["mean_inner_validation_BCE"])
    save(
        out / "model_selection.json",
        {
            "criterion": "Lowest mean minimum inner-validation BCE; outer test scores excluded from selection. Criterion formalized after initial comparison, not preregistered.",
            "selected": decisions[0],
            "all_candidates": decisions,
            "evaluation": "CV scores describe the five prespecified architectures. Selected-model performance is not an unbiased new test estimate; requires external validation. Full-data refit has no new held-out score.",
        },
    )
    print("ANOVA and plots complete; selected", decisions[0], flush=True)


def finalize(run, out):
    target = out / "final_model"
    target.mkdir(exist_ok=True)
    if (target / "verification.json").exists():
        return
    decision = json.loads((out / "model_selection.json").read_text())["selected"]
    config = json.loads((run / "protocol.json").read_text())
    if sha(config["data"]) != config["data_sha256"]:
        raise ValueError("Dataset changed")
    df = pd.read_csv(config["data"], low_memory=False)
    x = normalize(df)
    y = df.label.to_numpy(dtype="float32")
    pre = ColumnTransformer(
        [
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                ),
                NUM,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ]
    )
    a = pre.fit_transform(x).astype("float32")
    seed = 20261030
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kind = decision["model"]
    model = Network(kind, a.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    counts = np.bincount(y.astype(int))
    lossfn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(counts[0] / counts[1], device=device)
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(a), torch.from_numpy(y)),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history = []
    for epoch in range(1, decision["final_epochs"] + 1):
        model.train()
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = lossfn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        history.append({"epoch": epoch, "weighted_train_BCE": total / len(y)})
        print("Final refit", kind, epoch, flush=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "kind": kind,
            "input_dim": a.shape[1],
            "features": pre.get_feature_names_out().tolist(),
            "epochs": decision["final_epochs"],
        },
        target / "model.pt",
    )
    joblib.dump(pre, target / "preprocessor.joblib")
    save(target / "history.json", history)
    p = predict(model, a[:1024], device)
    ck = torch.load(target / "model.pt", map_location=device, weights_only=True)
    other = Network(kind, a.shape[1]).to(device)
    other.load_state_dict(ck["state_dict"])
    q = predict(
        other,
        joblib.load(target / "preprocessor.joblib")
        .transform(x.iloc[:1024])
        .astype("float32"),
        device,
    )
    if not np.allclose(p, q, atol=1e-6):
        raise AssertionError("Reload mismatch")
    save(
        target / "verification.json",
        {
            "model": kind,
            "training_rows": len(df),
            "epochs": decision["final_epochs"],
            "seed": seed,
            "device": device,
            "reload_max_probability_error": float(np.max(abs(p - q))),
            "new_test_metric": None,
            "note": "Full-data deployment candidate; no claim of independently measured final-model performance",
            "data_sha256": sha(config["data"]),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare_explanations", "analysis", "finalize"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.action == "prepare_explanations":
        dest = a.out / "explanations"
        dest.mkdir(parents=True, exist_ok=False)
        for name in ["protocol.json", "splits.npz"]:
            shutil.copy2(a.run / name, dest / name)
        for model_dir in (a.run / "models").glob("*_fold*"):
            target = dest / "models" / model_dir.name
            target.mkdir(parents=True)
            for name in ["model.pt", "preprocessor.joblib", "metrics.json"]:
                shutil.copy2(model_dir / name, target / name)
    elif a.action == "analysis":
        analysis(a.run, a.out)
    else:
        finalize(a.run, a.out)
