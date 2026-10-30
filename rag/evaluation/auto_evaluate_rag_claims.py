import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
JSON_PATH = ROOT / "rag/evaluation/five_fold_rag_results.json"
CSV_PATH = ROOT / "rag/evaluation/rag_claim_review_v2.csv"
OUT_PATH = ROOT / "rag/evaluation/rag_claim_review_auto.csv"

NOT_APPLICABLE = "Not applicable"
REVIEW = "Needs human review"

# Conservative rules: automatic matches are preliminary, not proof.
SOURCE_RULES = {
    "CISA-IR-001": [
        "validate alerts against telemetry",
        "automated detections",
        "preserve useful records",
        "document decisions",
        "unverified conclusion",
        "evidence is incomplete",
    ],
    "NIST-IR-001": [
        "investigative lead",
        "not proof of a confirmed",
        "distinguish confirmed findings",
        "hypotheses",
        "preserve relevant evidence",
        "document observations",
    ],
    "MITRE-ATTACK-001": [
        "specific att&ck technique",
        "supporting evidence",
        "classifier prediction alone",
        "technique occurred",
    ],
}


def norm(value):
    return " ".join((value or "").lower().split())


def get_fold_data(data, fold_number):
    folds = data.get("fold_results", [])
    if fold_number < 1 or fold_number > len(folds):
        return {}, {}
    item = folds[fold_number - 1]
    sample = item.get("rag_sample", {})
    report = sample.get("triage_report", {})
    return sample, report


def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Missing JSON: {JSON_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")

    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    extra_fields = ["evaluation_method", "evaluation_notes"]
    output_fields = fieldnames + [col for col in extra_fields if col not in fieldnames]

    # Preserve existing human-reviewed judgments from v2.
    for row in rows:
        row["evaluation_method"] = "Conservative automated preliminary check"
        row["evaluation_notes"] = ""

        fold_number = int(row["fold"])
        sample, report = get_fold_data(data, fold_number)
        retrieved = sample.get("retrieved_evidence", [])
        citations = report.get("evidence_citations", [])

        retrieved_by_id = {
            item.get("source_id"): item for item in retrieved if item.get("source_id")
        }
        citation_ids = {
            item.get("source_id") for item in citations if item.get("source_id")
        }

        claim_id = row.get("claim_id", "")
        category = norm(row.get("claim_category"))
        claim = norm(row.get("claim_text"))
        source_id = (row.get("source_id") or "").strip()
        notes = []

        # Do not overwrite judgments that were manually reviewed in v2.
        # These Fold 1 judgments were assessed in the current conversation.
        if fold_number == 1 and claim_id in {str(i) for i in range(1, 16)}:
            row["evaluation_method"] = "Previously reviewed manually"
            row["evaluation_notes"] = (
                "Preserved the existing Fold 1 judgment; not re-evaluated automatically."
            )
            continue

        # A zero-valued feature is not proof that payload was never captured.
        zero_payload_overclaim = (
            ("payload" in claim or "application-layer" in claim)
            and any(
                term in claim
                for term in (
                    "no payload",
                    "payload content is available",
                    "no application-layer traffic was captured",
                    "preventing analysis",
                )
            )
            and any(
                term in claim
                for term in (
                    "zero",
                    "length",
                    "bytes",
                    "fields",
                )
            )
        )
        if zero_payload_overclaim:
            row["support_status"] = "Partially supported"
            row["citation_correct"] = NOT_APPLICABLE
            notes.append(
                "Zero-valued fields do not establish that payload was never captured "
                "or is unavailable from other telemetry."
            )
            row["evaluation_notes"] = " ".join(notes)
            continue

        # Claims about saved model/report output do not need external citations.
        if category in {"detector_output", "observed_feature", "missing_information"}:
            row["citation_correct"] = NOT_APPLICABLE
            notes.append(
                "Data/report claim: verify against the saved fold record and feature list."
            )
            if row.get("support_status") not in {
                "Supported",
                "Partially supported",
                "Unsupported",
            }:
                row["support_status"] = REVIEW
            row["evaluation_notes"] = " ".join(notes)
            continue

        # Citation validation: a source ID must exist in both retrieved evidence
        # and the report's citation list to be considered attached to this report.
        if source_id and source_id != NOT_APPLICABLE:
            if source_id not in retrieved_by_id:
                row["citation_correct"] = "No"
                row["support_status"] = REVIEW
                notes.append(
                    f"{source_id} is not present in this fold's retrieved evidence."
                )
            elif source_id not in citation_ids:
                row["citation_correct"] = REVIEW
                notes.append(
                    f"{source_id} was retrieved but is not listed in the report citations."
                )
            else:
                evidence = norm(retrieved_by_id[source_id].get("text", ""))
                phrases = SOURCE_RULES.get(source_id, [])
                topic_match = any(phrase in evidence for phrase in phrases)
                claim_has_topic = any(
                    token in claim
                    for token in (
                        "telemetry",
                        "investigative lead",
                        "not proof",
                        "technique",
                        "supporting evidence",
                        "preserve",
                        "document",
                        "unverified",
                        "confirmed incident",
                        "automated detection",
                        "hypotheses",
                    )
                )
                if topic_match and claim_has_topic:
                    row["citation_correct"] = "Yes"
                    notes.append(
                        "Preliminary topic match between claim and retrieved evidence; "
                        "semantic entailment still requires human review."
                    )
                    row["support_status"] = REVIEW
                else:
                    row["citation_correct"] = REVIEW
                    row["support_status"] = REVIEW
                    notes.append(
                        "Source exists, but automated checks cannot establish semantic support."
                    )
        else:
            row["citation_correct"] = NOT_APPLICABLE
            if category in {
                "general_guidance",
                "recommended_action",
                "limitation",
                "attack_interpretation",
            }:
                row["support_status"] = REVIEW
                notes.append(
                    "No source assigned; review whether this claim needs a citation "
                    "or is a data-derived observation."
                )

        row["evaluation_notes"] = " ".join(notes)

    with OUT_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nOutput: {OUT_PATH}")
    print(f"Rows processed: {len(rows)}")
    print("\nSupport status totals:")
    for status, count in Counter(r.get("support_status", "") for r in rows).items():
        print(f"  {status}: {count}")

    print("\nCitation correctness totals:")
    for status, count in Counter(r.get("citation_correct", "") for r in rows).items():
        print(f"  {status}: {count}")

    print("\nRows still needing review:")
    for row in rows:
        if row.get("support_status") == REVIEW or row.get("citation_correct") == REVIEW:
            print(
                f"  Fold {row['fold']}, Claim {row['claim_id']}: "
                f"{row['claim_text']}"
            )


if __name__ == "__main__":
    main()
