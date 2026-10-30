"""Grouped cross-validation for tabular network-flow classification."""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
]
CAT = ["proto", "service", "conn_state"]
FEATURES = NUM + CAT
MODELS = ["cnn", "lstm", "cnn_lstm", "bilstm", "transformer"]


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(df):
    x = df[FEATURES].copy()
    for c in NUM:
        x[c] = pd.to_numeric(x[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in CAT:
        x[c] = x[c].fillna("<MISSING>").astype(str)
    return x


def codes(frame):
    return pd.factorize(pd.MultiIndex.from_frame(frame), sort=False)[0]


def groups_for(df, x):
    """Connected components: same feature signature OR bidirectional five-tuple."""
    signature = codes(x)
    a = df.src_ip.astype(str) + ":" + df.src_port.astype(str)
    b = df.dst_ip.astype(str) + ":" + df.dst_port.astype(str)
    connection = codes(
        pd.DataFrame({"a": np.minimum(a, b), "b": np.maximum(a, b), "proto": df.proto})
    )
    parent = np.arange(len(df))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values in [signature, connection]:
        first = {}
        for i, k in enumerate(values):
            if k in first:
                parent[root(i)] = root(first[k])
            else:
                first[k] = i
    groups = pd.factorize(
        np.asarray([root(i) for i in range(len(df))], dtype=np.int64), sort=False
    )[0]
    return groups, signature, connection


def prepare(data, out, seed=42):
    out.mkdir(parents=True, exist_ok=False)
    df = pd.read_csv(data, low_memory=False)
    x = normalize(df)
    y = pd.to_numeric(df.label, errors="raise").to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected binary labels 0,1")
    g, s, c = groups_for(df, x)
    assignment = np.zeros(len(df), int)
    splits = {}
    audit = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for fold, (dev, te) in enumerate(outer.split(x, y, g), 1):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + fold)
        it, iv = next(inner.split(x.iloc[dev], y[dev], g[dev]))
        tr, va = dev[it], dev[iv]
        entry = {"fold": fold}
        for name, ids in [("train", tr), ("validation", va), ("test", te)]:
            if len(np.unique(y[ids])) != 2:
                raise ValueError(f"{fold} {name} has only one class")
            splits[f"{name}_{fold}"] = ids
            entry[name] = {
                "rows": len(ids),
                "groups": len(np.unique(g[ids])),
                "class_counts": np.bincount(y[ids].astype(int), minlength=2).tolist(),
            }
        for ids1, ids2 in [(tr, va), (tr, te), (va, te)]:
            for key in [g, s, c]:
                if np.intersect1d(key[ids1], key[ids2]).size:
                    raise AssertionError("Group leakage")
        entry["all_pairwise_signature_connection_component_overlaps"] = 0
        audit.append(entry)
        assignment[te] = fold
    np.savez_compressed(out / "splits.npz", **splits)
    pd.DataFrame(
        {
            "row_id": np.arange(len(df)),
            "group": g,
            "fold": assignment,
            "label": y.astype(int),
        }
    ).to_csv(out / "manifest.csv", index=False)
    save(
        out / "audit.json",
        {
            "rows": len(df),
            "groups": len(np.unique(g)),
            "largest_group_rows": int(np.bincount(g).max()),
            "signature_groups": len(np.unique(s)),
            "connection_groups": len(np.unique(c)),
            "folds": audit,
        },
    )
    save(
        out / "protocol.json",
        {
            "data": str(data.resolve()),
            "data_sha256": sha(data),
            "seed": seed,
            "features": FEATURES,
            "grouping": "connected components of exact normalized input signatures and bidirectional five-tuples",
            "threshold": 0.5,
            "input_note": "Ordered tabular features, NOT temporal sequences",
            "selection": "minimum inner-validation BCE; outer test never used for selection",
            "historical_RF": "Supplementary historical baseline only; different folds are not paired comparisons",
        },
    )
    print(json.dumps(audit, indent=2), flush=True)


class Network(nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.body = nn.Sequential(
                nn.Conv1d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            width = 16 * d
        elif kind in ["lstm", "bilstm", "cnn_lstm"]:
            self.conv = nn.Conv1d(1, 8, 3, padding=1) if kind == "cnn_lstm" else None
            self.body = nn.LSTM(
                8 if self.conv else 1,
                16,
                batch_first=True,
                bidirectional=kind == "bilstm",
            )
            width = 32 if kind == "bilstm" else 16
        else:
            self.embed = nn.Linear(1, 16)
            self.position = nn.Parameter(torch.randn(1, d, 16) * 0.02)
            self.body = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(16, 2, 32, dropout=0.1, batch_first=True), 1
            )
            width = 16
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        if self.kind == "cnn":
            z = self.body(x.unsqueeze(1))
        elif self.kind == "transformer":
            z = self.body(self.embed(x.unsqueeze(-1)) + self.position).mean(1)
        else:
            z = (
                torch.relu(self.conv(x.unsqueeze(1))).transpose(1, 2)
                if self.conv
                else x.unsqueeze(-1)
            )
            _, (h, _) = self.body(z)
            z = torch.cat([h[-2], h[-1]], 1) if self.kind == "bilstm" else h[-1]
        return self.head(z).squeeze(1)


def predict(model, x, device, batch=1024):
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                torch.sigmoid(model(torch.as_tensor(x[i : i + batch], device=device)))
                .cpu()
                .numpy()
                for i in range(0, len(x), batch)
            ]
        )


def metrics(y, p):
    z = p >= 0.5
    return {
        "accuracy": float(accuracy_score(y, z)),
        "balanced_accuracy": float(balanced_accuracy_score(y, z)),
        "precision": float(precision_score(y, z, zero_division=0)),
        "recall": float(recall_score(y, z, zero_division=0)),
        "f1": float(f1_score(y, z, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def train(run, epochs, batch, models, smoke=False):
    config = json.loads((run / "protocol.json").read_text())
    data = Path(config["data"])
    if sha(data) != config["data_sha256"]:
        raise ValueError("Dataset changed")
    df = pd.read_csv(data, low_memory=False)
    x = normalize(df)
    y = df.label.to_numpy(dtype=np.float32)
    splits = np.load(run / "splits.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    target = run / ("smoke" if smoke else "models")
    target.mkdir(exist_ok=True)
    settings = {
        "epochs": epochs,
        "batch": batch,
        "smoke": smoke,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "lr": 0.001,
        "seed": config["seed"],
        "script_sha256": sha(__file__),
    }
    if (target / "settings.json").exists() and json.loads(
        (target / "settings.json").read_text()
    ) != settings:
        raise ValueError("Settings changed; use new run directory")
    save(target / "settings.json", settings)
    for fold in range(1, 6):
        tr, va, te = [splits[f"{k}_{fold}"] for k in ["train", "validation", "test"]]
        if smoke:
            tr, va, te = tr[:1024], va[:256], te[:256]
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
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CAT,
                ),
            ]
        )
        a = pre.fit_transform(x.iloc[tr]).astype("float32")
        b = pre.transform(x.iloc[va]).astype("float32")
        c = pre.transform(x.iloc[te]).astype("float32")
        for kind in models:
            dest = target / f"{kind}_fold{fold}"
            if (dest / "metrics.json").exists():
                continue
            dest.mkdir(exist_ok=True)
            seed = config["seed"] + fold
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = Network(kind, a.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001)
            counts = np.bincount(y[tr].astype(int), minlength=2)
            lossfn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(counts[0] / counts[1], device=device)
            )
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(a), torch.from_numpy(y[tr])
                ),
                batch_size=batch,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            start = time.perf_counter()
            history = []
            best = float("inf")
            for epoch in range(1, epochs + 1):
                model.train()
                total = 0
                for xb, yb in loader:
                    opt.zero_grad()
                    loss = lossfn(model(xb.to(device)), yb.to(device))
                    loss.backward()
                    opt.step()
                    total += float(loss.detach()) * len(xb)
                p = predict(model, b, device)
                vp = np.clip(p, 1e-7, 1 - 1e-7)
                v = float(-(y[va] * np.log(vp) + (1 - y[va]) * np.log(1 - vp)).mean())
                history.append(
                    {
                        "epoch": epoch,
                        "train_weighted_bce": total / len(a),
                        "validation_bce": v,
                    }
                )
                if v < best:
                    best = v
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "kind": kind,
                            "input_dim": a.shape[1],
                            "epoch": epoch,
                            "features": pre.get_feature_names_out().tolist(),
                        },
                        dest / "model.pt",
                    )
                print(
                    f"{kind} fold {fold} epoch {epoch}/{epochs}: validation BCE={v:.5f}",
                    flush=True,
                )
            seconds = time.perf_counter() - start
            model.load_state_dict(
                torch.load(dest / "model.pt", weights_only=True, map_location=device)[
                    "state_dict"
                ]
            )
            start = time.perf_counter()
            p = predict(model, c, device)
            inference = time.perf_counter() - start
            joblib.dump(pre, dest / "preprocessor.joblib")
            save(dest / "history.json", history)
            pd.DataFrame(
                {
                    "row_id": te,
                    "label": y[te].astype(int),
                    "probability": p,
                    "prediction": (p >= 0.5).astype(int),
                    "fold": fold,
                }
            ).to_csv(dest / "predictions.csv", index=False)
            if len(np.unique(y[te])) == 2:
                fpr, tpr, threshold = roc_curve(y[te], p)
                pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": threshold}).to_csv(
                    dest / "roc.csv", index=False
                )
                plt.figure()
                plt.plot(fpr, tpr)
                plt.plot([0, 1], [0, 1], "--")
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title(f"{kind}, fold {fold}")
                plt.savefig(dest / "roc.png")
                plt.close()
                result = metrics(y[te], p)
            else:
                result = {"single_class_smoke": True}
            cm = confusion_matrix(y[te], p >= 0.5, labels=[0, 1])
            ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot()
            plt.savefig(dest / "confusion.png")
            plt.close()
            save(
                dest / "metrics.json",
                dict(
                    result,
                    fold=fold,
                    model=kind,
                    train_seconds=seconds,
                    inference_seconds=inference,
                    confusion_matrix=cm.tolist(),
                    test_rows=len(te),
                ),
            )


def summarize(run):
    rows = [json.loads(p.read_text()) for p in (run / "models").glob("*/metrics.json")]
    table = pd.DataFrame(rows)
    if len(table) != 25 or set(table.model) != set(MODELS):
        raise ValueError("Require all 25 completed model-fold fits")
    table.to_csv(run / "fold_metrics.csv", index=False)
    result = {
        "caveat": "Exploratory only: five overlapping training folds are not independent replicates. Corrected paired t approximates dependence; not confirmatory ANOVA. Repeat grouped CV or use independent external datasets for strong inference.",
        "fold_mean_ci": {},
        "paired_corrected_t_holm": [],
    }
    for name in MODELS:
        vals = table[table.model == name].sort_values("fold").f1.to_numpy()
        ci = (
            stats.t.interval(0.95, 4, loc=vals.mean(), scale=stats.sem(vals))
            if vals.std() > 0
            else (vals.mean(), vals.mean())
        )
        result["fold_mean_ci"][name] = {
            "mean_f1": float(vals.mean()),
            "descriptive_t95": list(ci),
        }
        pred = pd.concat(
            [
                pd.read_csv(run / "models" / f"{name}_fold{f}" / "predictions.csv")
                for f in range(1, 6)
            ]
        ).sort_values("row_id")
        manifest = pd.read_csv(run / "manifest.csv")
        if not np.array_equal(pred.row_id, np.arange(len(manifest))):
            raise ValueError("OOF alignment failure")
        pred.to_csv(run / f"{name}_oof.csv", index=False)
        save(run / f"{name}_oof_metrics.json", metrics(pred.label, pred.probability))
    audit = json.loads((run / "audit.json").read_text())
    ratio = np.mean([f["test"]["rows"] / f["train"]["rows"] for f in audit["folds"]])
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1 :]:
            d = (
                table[table.model == a].sort_values("fold").f1.to_numpy()
                - table[table.model == b].sort_values("fold").f1.to_numpy()
            )
            se = np.sqrt((1 / 5 + ratio) * d.var(ddof=1))
            t = float(d.mean() / se) if se else None
            p = (
                float(2 * stats.t.sf(abs(t), 4))
                if t is not None
                else (1.0 if d.mean() == 0 else None)
            )
            result["paired_corrected_t_holm"].append(
                {
                    "a": a,
                    "b": b,
                    "mean_f1_difference": float(d.mean()),
                    "t": t,
                    "p": p,
                    "corrected95": (
                        d.mean() + np.array([-1, 1]) * stats.t.ppf(0.975, 4) * se
                    ).tolist(),
                }
            )
    valid = sorted(
        [r for r in result["paired_corrected_t_holm"] if r["p"] is not None],
        key=lambda r: r["p"],
    )
    last = 0
    for j, r in enumerate(valid):
        last = max(last, min(1, (len(valid) - j) * r["p"]))
        r["p_holm"] = last
    save(run / "statistics.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "train", "summarize"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.run)
    elif args.action == "train":
        train(args.run, args.epochs, args.batch, args.models, args.smoke)
    else:
        summarize(args.run)
