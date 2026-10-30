import csv
import json
import re
from pathlib import Path

root = Path("rag/evaluation")
json_path = root / "five_fold_rag_results.json"
csv_path = root / "rag_claim_review.csv"

if not json_path.exists() or not csv_path.exists():
    raise SystemExit(
        "Missing JSON results or claim-review CSV. Run from the project root."
    )

data = json.loads(json_path.read_text(encoding="utf-8"))
with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def get_text(obj, *keys):
    for key in keys:
        val = obj.get(key)
        if isinstance(val, str):
            return val
    return ""


fold_map = {}
for i, fold in enumerate(data.get("fold_results", []), start=1):
    fold_num = fold.get("fold", i)
    sample = fold.get("rag_sample") or {}
    report = sample.get("triage_report") or {}
    evidence = sample.get("retrieved_evidence") or []
    fold_map[str(fold_num)] = {
        "report": report,
        "evidence": evidence,
        "evidence_text": " ".join(
            norm(e.get("text", "")) for e in evidence if isinstance(e, dict)
        ),
        "source_ids": {
            str(e.get("source_id", "")).strip() for e in evidence if isinstance(e, dict)
        },
    }


def evaluate(row):
    fold = fold_map.get(str(row.get("fold", "")))
    if not fold:
        return "Needs human review", "Needs human review", "Fold evidence not found."

    claim = norm(row.get("claim_text", ""))
    category = norm(row.get("claim_category", ""))
    report = fold["report"]
    evidence_text = fold["evidence_text"]
    evidence = fold["evidence"]
    source_id = str(row.get("source_id", "")).strip()
    notes = []

    # Citation correctness: only check whether the referenced source was retrieved.
    # Presence alone does not prove semantic support.
    if source_id in ("", "not applicable", "none", "null"):
        citation = "Not applicable"
    elif source_id in fold["source_ids"]:
        citation = "Needs human review"
        notes.append("Source was retrieved; verify that it supports this exact claim.")
    else:
        citation = "No"
        notes.append(
            "Referenced source ID was not found in this fold's retrieved evidence."
        )

    # Exact feature/model fields can be checked against the corresponding report.
    if category == "detector_output":
        expected = norm(get_text(report, "detector_prediction"))
        if expected and expected in claim:
            status = "Supported"
            notes.append("Prediction matches the report field.")
        else:
            status = "Needs human review"
            notes.append(
                "Check the prediction and probability against the detector output."
            )

    elif category == "observed_feature":
        summary = norm(get_text(report, "evidence_summary"))
        if claim and summary and claim in summary:
            status = "Supported"
            notes.append("Claim matches the report's evidence summary.")
        else:
            status = "Needs human review"
            notes.append(
                "Compare against the original flow features; the summary alone may be insufficient."
            )

    elif category in ("general_guidance", "recommended_action"):
        if claim and claim in evidence_text:
            status = "Supported"
            notes.append("Claim text appears verbatim in retrieved evidence.")
            if source_id == "Not applicable":
                citation = "Needs human review"
        else:
            status = "Needs human review"
            notes.append(
                "Semantic support requires checking the claim against this fold's retrieved source text."
            )

    elif category == "missing_information":
        missing = report.get("missing_information") or []
        exact_items = [norm(x) for x in missing if isinstance(x, str)]
        if claim in exact_items:
            status = "Supported"
            notes.append("Claim matches the report's missing-information field.")
        else:
            status = "Needs human review"
            notes.append("Verify against the available features and original record.")

    elif category == "limitation":
        limitations = report.get("limitations") or []
        exact_items = [norm(x) for x in limitations if isinstance(x, str)]
        if claim in exact_items:
            status = "Supported"
            notes.append("Claim matches the report's limitation field.")
        else:
            status = "Needs human review"
            notes.append("Verify the limitation against the evaluation design.")

    else:
        status = "Needs human review"
        notes.append(
            "Check factual accuracy against this fold's report, evidence, and original record."
        )

    # A model-generated interpretation is never automatically accepted as fact.
    if category == "attack_interpretation":
        status = "Needs human review"
        notes.append("Do not treat the model's interpretation as a confirmed incident.")

    return status, citation, " ".join(notes)


updated = 0
for row in rows:
    status, citation, notes = evaluate(row)
    row["support_status"] = status
    row["citation_correct"] = citation
    existing = row.get("notes", "").strip()
    row["notes"] = (existing + " " + notes).strip()
    updated += 1

columns = [
    "fold",
    "claim_id",
    "claim_text",
    "claim_category",
    "source_id",
    "support_status",
    "citation_correct",
    "notes",
]
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"Updated {updated} rows in {csv_path}")
print("\nPreliminary support-status counts:")
for status in sorted(set(r["support_status"] for r in rows)):
    print(f"{status}: {sum(r['support_status'] == status for r in rows)}")

print("\nCitation-status counts:")
for status in sorted(set(r["citation_correct"] for r in rows)):
    print(f"{status}: {sum(r['citation_correct'] == status for r in rows)}")

print(
    "\nImportant: these are preliminary checks, not a completed human-validated evaluation."
)
