import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from triage import generate_triage

TRAIN_FILE = ROOT / "dataset" / "split_v5_five_tuple_seed42_fold4" / "train.csv"
TEST_FILE = ROOT / "dataset" / "split_v5_five_tuple_seed42_fold4" / "test.csv"
OUTPUT_FILE = ROOT / "rag" / "evaluation" / "detector_rag_demo.json"

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


def main():
    for path in (TRAIN_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required split file not found: {path}\n"
                "Check that the v5 five-tuple split exists."
            )

    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)

    required = set(FEATURES + ["label"])
    for name, frame in (("train", train), ("test", test)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC),
            ("categorical", categorical_pipeline, CATEGORICAL),
        ]
    )

    model = Pipeline(
        [
            ("preprocess", preprocess),
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

    print(f"Training rows: {len(train):,}")
    print(f"Held-out rows: {len(test):,}")
    print("Training Random Forest...")
    start = time.time()
    model.fit(train[FEATURES], train["label"].astype(int))
    print(f"Training completed in {time.time() - start:.2f} seconds.")

    # Select one deterministic held-out row.
    # Ground truth is retained for local evaluation only.
    row_index = 0
    row = test.iloc[row_index]
    X_one = test.loc[[row_index], FEATURES]
    prediction = int(model.predict(X_one)[0])
    probabilities = model.predict_proba(X_one)[0]
    classes = list(model.named_steps["classifier"].classes_)
    attack_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
    ground_truth = int(row["label"])

    # Do not pass the ground-truth label or attack type to the LLM.
    # Include only selected model input features in the investigation query.
    feature_summary = {
        name: (
            None
            if pd.isna(row[name])
            else row[name].item() if hasattr(row[name], "item") else row[name]
        )
        for name in FEATURES
    }

    query = (
        "Investigate this held-out network flow using the available "
        "features. Identify what can and cannot be concluded, and suggest "
        "checks an analyst should perform. Do not assume the prediction "
        "proves an incident. Flow features: " + json.dumps(feature_summary, default=str)
    )

    detector_prediction = (
        f"Random Forest predicted "
        f"{'attack' if prediction == 1 else 'normal'}; "
        f"estimated attack probability={attack_probability:.4f}"
    )

    print("\n=== DETECTOR OUTPUT ===")
    print("Held-out row index:", row_index)
    print("Predicted label:", prediction)
    print("Predicted class:", "attack" if prediction == 1 else "normal")
    print(f"Estimated attack probability: {attack_probability:.4f}")
    print("Ground truth (evaluation only):", ground_truth)
    print("Ground truth passed to LLM: NO")

    print("\nRunning evidence retrieval and LLM triage...")
    result = generate_triage(
        query=query,
        detector_prediction=detector_prediction,
        top_k=3,
    )

    output = {
        "experiment": "Random Forest + evidence-grounded RAG demo",
        "split": "v5_five_tuple_seed42_fold4",
        "held_out_row_index": row_index,
        "detector_prediction": prediction,
        "detector_class": "attack" if prediction == 1 else "normal",
        "estimated_attack_probability": attack_probability,
        "ground_truth_for_evaluation_only": ground_truth,
        "correct_prediction": prediction == ground_truth,
        "ground_truth_sent_to_llm": False,
        "features_sent_to_llm": feature_summary,
        "retrieved_evidence": result["retrieved_evidence"],
        "triage_report": result["triage_report"],
        "model": result["model"],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n=== RAG TRIAGE REPORT ===")
    print(json.dumps(result["triage_report"], indent=2, ensure_ascii=False))
    print("\nPrediction correct:", prediction == ground_truth)
    print("Saved result:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
