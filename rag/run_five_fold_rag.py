import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage import generate_triage

DATA_FILE = ROOT / "dataset" / "train_test_network.csv"
OUTPUT_DIR = ROOT / "rag" / "evaluation"
OUTPUT_JSON = OUTPUT_DIR / "five_fold_rag_results.json"
OUTPUT_TXT = OUTPUT_DIR / "five_fold_rag_summary.txt"

NUMERIC = [
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
CATEGORICAL = ["proto", "service", "conn_state"]
FEATURES = NUMERIC + CATEGORICAL
GROUP_COLUMNS = ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]


def make_model():
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                NUMERIC,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def safe_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    required = set(FEATURES + GROUP_COLUMNS + ["label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    X = df[FEATURES]
    y = df["label"].astype(int)

    # Five-tuple grouping is the existing project protocol.
    # It reduces flow-key overlap but does not guarantee temporal/session isolation.
    group_frame = df[GROUP_COLUMNS].astype("string").fillna("<NA>")
    groups = pd.util.hash_pandas_object(group_frame, index=False).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_results = []
    all_y_true = []
    all_y_pred = []
    rag_successes = 0

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        print(f"\n{'=' * 18} FOLD {fold_number}/5 {'=' * 18}")

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = len(train_groups & test_groups)
        if overlap:
            raise RuntimeError(f"Fold {fold_number} has {overlap} overlapping groups.")

        model = make_model()
        start = time.time()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_seconds = time.time() - start

        predictions = model.predict(X.iloc[test_idx]).astype(int)
        truth = y.iloc[test_idx].to_numpy()

        all_y_true.extend(truth.tolist())
        all_y_pred.extend(predictions.tolist())

        fold_metrics = {
            "fold": fold_number,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "group_overlap": overlap,
            "accuracy": float(accuracy_score(truth, predictions)),
            "attack_precision": float(
                precision_score(truth, predictions, zero_division=0)
            ),
            "attack_recall": float(recall_score(truth, predictions, zero_division=0)),
            "attack_f1": float(f1_score(truth, predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
            "confusion_matrix": confusion_matrix(
                truth, predictions, labels=[0, 1]
            ).tolist(),
            "training_seconds": round(train_seconds, 3),
        }

        print(
            f"Detector: F1={fold_metrics['attack_f1']:.4f}, "
            f"balanced accuracy={fold_metrics['balanced_accuracy']:.4f}, "
            f"train={train_seconds:.2f}s"
        )

        # One fixed test record per fold for the expensive LLM evaluation.
        sample_pos = 0
        sample_row_idx = int(test_idx[sample_pos])
        sample = df.iloc[sample_row_idx]
        sample_X = X.iloc[[sample_row_idx]]

        prediction = int(model.predict(sample_X)[0])
        probabilities = model.predict_proba(sample_X)[0]
        classes = list(model.named_steps["classifier"].classes_)
        attack_probability = (
            float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        )

        feature_summary = {name: safe_value(sample[name]) for name in FEATURES}

        query = (
            "Triage this held-out network flow using only the supplied "
            "features and retrieved evidence. Explain what the detector "
            "output suggests, what remains unknown, and which checks "
            "an analyst should perform. Do not claim that a security "
            "incident is confirmed. Flow features: "
            + json.dumps(feature_summary, default=str)
        )

        detector_output = (
            f"Random Forest predicted "
            f"{'attack' if prediction == 1 else 'normal'}; "
            f"estimated attack probability={attack_probability:.4f}"
        )

        rag_result = {
            "status": "failed",
            "error": None,
        }

        try:
            result = generate_triage(
                query=query,
                detector_prediction=detector_output,
                top_k=3,
            )
            rag_result = {
                "status": "success",
                "error": None,
                "held_out_row_index": sample_row_idx,
                "predicted_label": prediction,
                "ground_truth_for_evaluation_only": int(sample["label"]),
                "correct_prediction": (prediction == int(sample["label"])),
                "ground_truth_sent_to_llm": False,
                "estimated_attack_probability": attack_probability,
                "features_sent_to_llm": feature_summary,
                "retrieved_evidence": result["retrieved_evidence"],
                "triage_report": result["triage_report"],
            }
            rag_successes += 1
            print("RAG triage: success")
        except Exception as exc:
            rag_result["error"] = str(exc)
            print(f"RAG triage failed: {exc}")

        fold_results.append(
            {
                "detector_metrics": fold_metrics,
                "rag_sample": rag_result,
            }
        )

    y_true = np.asarray(all_y_true)
    y_pred = np.asarray(all_y_pred)

    # Each original record appears in exactly one test fold.
    overall = {
        "records_evaluated": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "attack_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    result = {
        "experiment": "SOC-RAG Guard five-fold detector plus RAG smoke evaluation",
        "dataset": str(DATA_FILE),
        "folds": 5,
        "grouping": GROUP_COLUMNS,
        "feature_columns": FEATURES,
        "detector": "RandomForestClassifier, 100 trees, seed 42",
        "rag_generation_model": "qwen3.5:4b",
        "embedding_model": "embeddinggemma",
        "rag_samples_per_fold": 1,
        "rag_successful_samples": rag_successes,
        "overall_out_of_fold_detector_metrics": overall,
        "fold_results": fold_results,
        "limitations": [
            "Grouping is based on the five-tuple, not a verified session ID or timestamp.",
            "Five RAG reports are an integration smoke test, not a comprehensive evaluation of generated-report quality.",
            "Citation-ID validity does not establish semantic entailment or factual correctness.",
            "The evidence corpus is small and manually curated.",
            "The detector probability is not necessarily calibrated.",
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        "SOC-RAG Guard: Five-Fold Detector + RAG Evaluation",
        "=" * 54,
        f"Dataset rows evaluated out-of-fold: {overall['records_evaluated']}",
        f"RAG sample successes: {rag_successes}/5",
        "",
        "Overall out-of-fold detector metrics:",
    ]
    for key in (
        "accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "balanced_accuracy",
    ):
        lines.append(f"{key}: {overall[key]:.6f}")
    lines.append(
        f"Confusion matrix [[TN, FP], [FN, TP]]: {overall['confusion_matrix']}"
    )
    lines.append("")
    for item in fold_results:
        metrics = item["detector_metrics"]
        lines.append(
            f"Fold {metrics['fold']}: F1={metrics['attack_f1']:.6f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f}, "
            f"group_overlap={metrics['group_overlap']}, "
            f"RAG={item['rag_sample']['status']}"
        )
    lines += [
        "",
        "Caveat: five-tuple grouping does not guarantee session-level or temporal isolation.",
        "RAG sample success does not establish semantic grounding or report quality.",
    ]
    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== OVERALL OUT-OF-FOLD DETECTOR METRICS ===")
    print(json.dumps(overall, indent=2))
    print(f"\nRAG triage successes: {rag_successes}/5")
    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
