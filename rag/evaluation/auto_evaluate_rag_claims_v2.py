import csv
import json
from pathlib import Path
from collections import Counter

BASE = Path.cwd()
JSON_PATH = BASE / "rag/evaluation/five_fold_rag_results.json"
INPUT_CSV = BASE / "rag/evaluation/rag_claim_review_v2.csv"
OUTPUT_CSV = BASE / "rag/evaluation/rag_claim_review_auto_v2.csv"

NA = "Not applicable"
REVIEW = "Needs human review"


def load_json():
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def normalize(value):
    return " ".join(str(value or "").lower().split())


def get_sample(data, fold):
    return data["fold_results"][int(fold) - 1]["rag_sample"]


def check_detector_claim(text, sample):
    """Check a prediction/probability claim against saved report data."""
    claim = normalize(text)
    prediction = sample.get("predicted_label")
    probability = sample.get("estimated_attack_probability")

    if "predicts an attack" in claim or "classified attack" in claim:
        if prediction != 1:
            return "Unsupported", "Saved prediction does not indicate Attack."
        if probability is not None and f"{probability:.4f}" in claim:
            return "Supported", "Prediction and probability match saved record."
        if probability is not None and f"{probability:.2f}" in claim:
            return "Supported", "Prediction and rounded probability match saved record."
        if probability is not None and str(probability) in claim:
            return "Supported", "Prediction and probability match saved record."
        return REVIEW, "Prediction matches; verify probability wording and rounding."

    if claim in {"detector prediction: attack", "detector prediction attack"}:
        return (
            ("Supported", "Saved predicted_label is 1 (Attack).")
            if prediction == 1
            else ("Unsupported", "Saved predicted_label is not 1.")
        )

    return REVIEW, "Claim does not match a recognized deterministic pattern."


def main():
    for path in (JSON_PATH, INPUT_CSV):
        if not path.exists():
            raise FileNotFoundError(path)

    data = load_json()
    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    for extra in ("evaluation_method", "evaluation_notes"):
        if extra not in fields:
            fields.append(extra)

    for row in rows:
        fold = int(row["fold"])
        claim_id = row["claim_id"]
        category = normalize(row.get("claim_category"))
        claim = row.get("claim_text", "")
        sample = get_sample(data, fold)
        features = sample.get("features_sent_to_llm") or {}
        report = sample.get("triage_report") or {}
        notes = []

        # Preserve Fold 1 judgments already reviewed manually.
        if fold == 1 and 1 <= int(claim_id) <= 15:
            row["evaluation_method"] = "Previously reviewed manually"
            row["evaluation_notes"] = (
                "Existing Fold 1 human-reviewed judgment preserved."
            )
            continue

        row["evaluation_method"] = "Automated preliminary check"

        if category in {"detector_output", "attack_interpretation"}:
            status, reason = check_detector_claim(claim, sample)
            row["support_status"] = status
            row["citation_correct"] = NA
            notes.append(reason)

        elif category == "observed_feature":
            # Compare only explicit field/value mentions that can be verified.
            text = normalize(claim)
            checks = {
                "proto": features.get("proto"),
                "conn_state": features.get("conn_state"),
                "service": features.get("service"),
                "src_bytes": features.get("src_bytes"),
                "dst_bytes": features.get("dst_bytes"),
                "dns_qclass": features.get("dns_qclass"),
                "dns_qtype": features.get("dns_qtype"),
                "dns_rcode": features.get("dns_rcode"),
                "http_request_body_len": features.get("http_request_body_len"),
                "http_response_body_len": features.get("http_response_body_len"),
            }
            matched = []
            for key, value in checks.items():
                if key in text and value is not None and str(value).lower() in text:
                    matched.append(f"{key}={value}")
            if matched:
                row["support_status"] = "Supported"
                notes.append("Verified fields: " + ", ".join(matched))
            else:
                row["support_status"] = REVIEW
                notes.append("Could not verify explicit field/value pairs.")
            row["citation_correct"] = NA

        elif category == "missing_information":
            text = normalize(claim)
            if "payload" in text and any(
                term in text for term in ("zero", "length", "bytes", "field")
            ):
                row["support_status"] = "Partially supported"
                notes.append(
                    "Zero-valued HTTP/DNS fields do not prove payload was never captured."
                )
            else:
                row["support_status"] = REVIEW
                notes.append(
                    "Check the claim against the feature list and available telemetry."
                )
            row["citation_correct"] = NA

        elif category == "general_guidance":
            citations = report.get("evidence_citations", [])
            retrieved = {
                item.get("source_id"): normalize(item.get("text", ""))
                for item in sample.get("retrieved_evidence", [])
                if item.get("source_id")
            }
            source_id = row.get("source_id", "").strip()

            citation_ids = {c.get("source_id") for c in citations if c.get("source_id")}

            if (
                source_id
                and source_id in retrieved
                and source_id in citation_ids
                and retrieved[source_id]
            ):
                # This confirms source availability, not semantic entailment.
                row["citation_correct"] = REVIEW
                row["support_status"] = REVIEW
                notes.append(
                    "Source was retrieved and cited; semantic support requires review."
                )
            else:
                row["citation_correct"] = "No"
                row["support_status"] = REVIEW
                notes.append(
                    "Source is missing from retrieved evidence or report citations."
                )

        else:
            # Recommendations and limitations need claim-specific review.
            row["support_status"] = REVIEW
            row["citation_correct"] = NA
            notes.append(
                "Requires claim-specific review; automated matching is insufficient."
            )

        row["evaluation_notes"] = " ".join(notes)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Output written to: {OUTPUT_CSV}")
    print(f"Rows processed: {len(rows)}")

    print("\nSupport status:")
    for name, count in Counter(r.get("support_status", "") for r in rows).items():
        print(f"  {name}: {count}")

    print("\nCitation status:")
    for name, count in Counter(r.get("citation_correct", "") for r in rows).items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
