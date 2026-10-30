"""Batch inference for the selected tabular flow detector."""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import torch
from protocol import Network, normalize, predict


def main(bundle, input_path, output):
    if input_path.resolve() == output.resolve():
        raise ValueError("Output must differ from input")
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_csv(input_path, low_memory=False)
    pre = joblib.load(bundle / "preprocessor.joblib")
    ck = torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)
    model = Network(ck["kind"], ck["input_dim"])
    model.load_state_dict(ck["state_dict"])
    torch.set_num_threads(4)
    p = predict(model, pre.transform(normalize(frame)).astype("float32"), "cpu")
    pd.DataFrame(
        {
            "row_id": range(len(frame)),
            "attack_probability": p,
            "prediction": (p >= 0.5).astype(int),
        }
    ).to_csv(output, index=False)
    print(
        f"Saved {len(frame)} predictions to {output}. Predictions require analyst validation."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.bundle, a.input, a.output)
