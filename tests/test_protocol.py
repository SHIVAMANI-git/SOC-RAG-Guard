import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
import unittest
import numpy as np
import pandas as pd
import torch
from protocol import FEATURES, NUM, CAT, MODELS, Network, groups_for, normalize


class ProtocolTests(unittest.TestCase):
    def test_transitive_group_isolation(self):
        # Rows 0/1 share input signature; rows 1/2 share reverse connection.
        rows = []
        for i in range(4):
            r = {c: float(i) for c in NUM}
            r.update({c: "tcp" for c in CAT})
            r.update(src_ip=f"a{i}", src_port=1, dst_ip=f"b{i}", dst_port=2)
            rows.append(r)
        df = pd.DataFrame(rows)
        df.loc[1, FEATURES] = df.loc[0, FEATURES]
        df.loc[2, ["src_ip", "src_port", "dst_ip", "dst_port"]] = ["b1", 2, "a1", 1]
        g, _, _ = groups_for(df, normalize(df))
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[1], g[2])
        self.assertNotEqual(g[2], g[3])

    def test_all_architectures_batch_shapes_gradients_and_reload(self):
        for kind in MODELS:
            model = Network(kind, 21)
            x = torch.randn(3, 21)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (3,))
            logits.sum().backward()
            self.assertTrue(
                all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
            )
            other = Network(kind, 21)
            other.load_state_dict(model.state_dict())
            model.eval()
            other.eval()
            torch.testing.assert_close(model(x), other(x))

    def test_normalization_excludes_labels_and_handles_infinity(self):
        df = pd.DataFrame(
            [
                {
                    **{c: np.inf for c in NUM},
                    **{c: None for c in CAT},
                    "label": 1,
                    "type": "attack",
                }
            ]
        )
        x = normalize(df)
        self.assertEqual(list(x), FEATURES)
        self.assertTrue(x[NUM].isna().all().all())


if __name__ == "__main__":
    unittest.main()
