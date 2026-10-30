import json
import requests

from retriever import retrieve

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen3.5:4b"

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},
        "detector_prediction": {"type": "string"},
        "evidence_summary": {"type": "string"},
        "evidence_citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["source_id", "reason"],
            },
        },
        "missing_information": {"type": "array", "items": {"type": "string"}},
        "recommended_checks": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "assessment",
        "detector_prediction",
        "evidence_summary",
        "evidence_citations",
        "missing_information",
        "recommended_checks",
        "limitations",
    ],
}

SYSTEM_PROMPT = """
You are an evidence-grounded SOC triage assistant.

Treat the detector output as a prediction, not confirmed ground truth.
Use retrieved sources only for claims about incident-response guidance.
Do not invent logs, indicators, affected assets, or events.
State explicitly when telemetry is missing.
Recommend validation steps; do not claim that checks were performed.
Do not conclude that an attack is confirmed from a model prediction alone.
Cite only the supplied source IDs.
Return an object matching the supplied JSON schema.
"""


def call_model(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "stream": False,
            "format": REPORT_SCHEMA,
            "messages": messages,
            "options": {"temperature": 0},
        },
        timeout=240,
    )
    response.raise_for_status()

    content = response.json().get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response.")

    try:
        report = json.loads(content)
    except json.JSONDecodeError:
        # Retry once with the malformed response explicitly rejected.
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid JSON. "
                    "Return a corrected JSON object matching the schema. "
                    "Do not add Markdown fences or text outside the JSON."
                ),
            },
        ]

        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "stream": False,
                "format": REPORT_SCHEMA,
                "messages": retry_messages,
                "options": {"temperature": 0},
            },
            timeout=240,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        try:
            report = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama returned invalid JSON after one retry. "
                f"Response excerpt: {content[:800]}"
            ) from exc

    return report


def validate_report(report, allowed_ids):
    required = REPORT_SCHEMA["required"]
    if not isinstance(report, dict):
        raise RuntimeError("The report is not a JSON object.")

    missing = [key for key in required if key not in report]
    if missing:
        raise RuntimeError(f"Report is missing fields: {missing}")

    for key in ("assessment", "detector_prediction", "evidence_summary"):
        if not isinstance(report[key], str):
            raise RuntimeError(f"Field {key!r} must be a string.")

    for key in (
        "evidence_citations",
        "missing_information",
        "recommended_checks",
        "limitations",
    ):
        if not isinstance(report[key], list):
            raise RuntimeError(f"Field {key!r} must be an array.")

    invalid_ids = []
    for citation in report["evidence_citations"]:
        if not isinstance(citation, dict):
            raise RuntimeError("A citation is not a JSON object.")
        source_id = citation.get("source_id")
        if source_id not in allowed_ids:
            invalid_ids.append(source_id)

    if invalid_ids:
        raise RuntimeError(f"Unretrieved citation IDs were returned: {invalid_ids}")


def generate_triage(query, detector_prediction="not_provided", top_k=3):
    evidence = retrieve(query, top_k=top_k)

    allowed_ids = [item["source_id"] for item in evidence]

    evidence_text = "\n\n".join(
        f"Source ID: {item['source_id']}\n"
        f"Title: {item['title']}\n"
        f"URL: {item['source_url']}\n"
        f"Evidence: {item['text']}"
        for item in evidence
    )

    user_prompt = f"""
Investigation query:
{query}

Detector output:
{detector_prediction}

Retrieved evidence:
{evidence_text}

Permitted citation IDs:
{json.dumps(allowed_ids)}

Assess the situation cautiously. The query and detector output do not
prove an incident occurred. Do not claim to have inspected telemetry
beyond the information supplied.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    report = call_model(messages)
    validate_report(report, set(allowed_ids))

    return {
        "query": query,
        "retrieved_evidence": evidence,
        "triage_report": report,
        "model": CHAT_MODEL,
    }


def main():
    query = (
        "A network detector flags unusual connections. "
        "What should an analyst validate before declaring an incident?"
    )

    result = generate_triage(query=query, detector_prediction="not_provided", top_k=3)

    print("\n=== RETRIEVED EVIDENCE ===")
    for item in result["retrieved_evidence"]:
        print(f"{item['source_id']} | " f"similarity={item['similarity']:.4f}")

    print("\n=== GENERATED TRIAGE REPORT ===")
    print(json.dumps(result["triage_report"], indent=2, ensure_ascii=False))

    output_path = "rag/evaluation/triage_demo.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
