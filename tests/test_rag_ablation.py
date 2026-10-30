"""Offline contract tests only; synthetic reports are NOT research results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
import json
import tempfile
import types
import unittest
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import pandas as pd
from protocol import NUM, CAT, sha
from rag_ablation import generate, score


class Response:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        pass

    def json(self):
        return self.value


class AblationTests(unittest.TestCase):
    def test_paired_inputs_and_unreviewed_score_refusal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            run.mkdir()
            kb = root / "rag/knowledge_base"
            kb.mkdir(parents=True)
            (kb / "cybersecurity_evidence.json").write_text("[]")
            data = root / "data.csv"
            pd.DataFrame(
                [
                    {
                        **{c: 1 for c in NUM},
                        **{c: "tcp" for c in CAT},
                        "label": i % 2,
                        "type": "hidden",
                    }
                    for i in range(5)
                ]
            ).to_csv(data, index=False)
            (run / "protocol.json").write_text(
                json.dumps({"data": str(data), "data_sha256": sha(data)})
            )
            pd.DataFrame(
                {"row_id": range(5), "fold": range(1, 6), "group": range(5)}
            ).to_csv(run / "manifest.csv", index=False)
            for fold in range(1, 6):
                directory = run / "models" / f"cnn_fold{fold}"
                directory.mkdir(parents=True)
                pd.DataFrame({"row_id": [fold - 1], "probability": [0.4]}).to_csv(
                    directory / "predictions.csv", index=False
                )
            retriever = types.SimpleNamespace(
                retrieve=lambda query, top_k: [
                    {"source_id": "S1", "text": "Validate alerts."}
                ]
            )
            payloads = []

            def post(url, **kwargs):
                payload = kwargs["json"]
                if url.endswith("/generate"):
                    return Response({})
                payloads.append(payload)
                return Response(
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "claims": [
                                        {
                                            "text": "The detector is uncertain.",
                                            "source_ids": [],
                                        }
                                    ],
                                    "recommended_checks": ["Validate alerts."],
                                }
                            )
                        }
                    }
                )

            with (
                patch.dict("sys.modules", {"retriever": retriever}),
                patch(
                    "rag_ablation.requests.get", return_value=Response({"models": []})
                ),
                patch("rag_ablation.requests.post", side_effect=post),
                redirect_stdout(io.StringIO()),
            ):
                generate(run, root, 1, "mock-only")
            self.assertEqual(len(payloads), 10)
            for i in range(0, 10, 2):
                a, b = [
                    json.loads(p["messages"][1]["content"]) for p in payloads[i : i + 2]
                ]
                self.assertEqual(a["query"], b["query"])
                self.assertNotEqual(a["evidence"], b["evidence"])
                self.assertNotIn("hidden", a["query"])
                self.assertNotIn("label", a["query"])
                self.assertEqual(payloads[i]["options"], payloads[i + 1]["options"])
            automatic = json.loads(
                (run / "rag_ablation/automatic_metrics.json").read_text()
            )
            self.assertIsNone(automatic["semantic_groundedness"])
            with self.assertRaises(ValueError):
                score(run)


if __name__ == "__main__":
    unittest.main()
