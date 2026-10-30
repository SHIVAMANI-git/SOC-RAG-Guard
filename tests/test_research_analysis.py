import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
import unittest
import tempfile
import numpy as np
import pandas as pd
from research_analysis import rm_anova, source_feature
from reviewer_agreement import main as compare_reviewers


class ResearchAnalysisTests(unittest.TestCase):
    def test_anova_preserves_affine_scale_block_offsets_and_model_order(self):
        a = np.random.default_rng(2).normal(size=(8, 5))
        original = rm_anova(a)
        for changed in [a * 3 + 20, a + np.arange(8)[:, None], a[:, ::-1]]:
            b = rm_anova(changed)
            for key in ["F", "p_GG", "greenhouse_geisser_epsilon"]:
                self.assertAlmostEqual(original[key], b[key], places=10)
        self.assertGreaterEqual(original["greenhouse_geisser_epsilon"], 0.25)
        self.assertLessEqual(original["greenhouse_geisser_epsilon"], 1)

    def test_anova_zero_treatment_effect(self):
        a = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]], float)
        result = rm_anova(a)
        self.assertEqual(result["F"], 0)
        self.assertEqual(result["p_GG"], 1)

    def test_feature_names_preserve_underscored_names(self):
        self.assertEqual(source_feature("cat__conn_state_S0"), "conn_state")
        self.assertEqual(source_feature("num__src_ip_bytes"), "src_ip_bytes")
        with self.assertRaises(ValueError):
            source_feature("unknown")

    def test_human_agreement_refuses_blank_or_mismatched_reviews(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            a = p / "a.csv"
            b = p / "b.csv"
            frame = pd.DataFrame(
                {
                    "blind_id": ["a", "b"],
                    "claim_index": [0, 0],
                    "support_status": ["supported", "partial"],
                    "reviewer": ["Reviewer 1", "Reviewer 1"],
                }
            )
            frame.to_csv(a, index=False)
            frame.assign(reviewer="Reviewer 2").to_csv(b, index=False)
            compare_reviewers(a, b, p / "agreement.json")
            frame.assign(support_status="").to_csv(b, index=False)
            with self.assertRaises(ValueError):
                compare_reviewers(a, b, p / "invalid.json")
            frame.assign(blind_id=["a", "c"]).to_csv(b, index=False)
            with self.assertRaises(ValueError):
                compare_reviewers(a, b, p / "invalid.json")


if __name__ == "__main__":
    unittest.main()
