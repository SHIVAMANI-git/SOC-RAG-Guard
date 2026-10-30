"""Coverage checks and inter-reviewer agreement for completed review sheets."""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from protocol import save


def main(first, second, out):
    a = pd.read_csv(first).fillna("")
    b = pd.read_csv(second).fillna("")
    cols = ["blind_id", "claim_index"]
    if a.duplicated(cols).any() or b.duplicated(cols).any():
        raise ValueError("Duplicate review rows")
    if set(map(tuple, a[cols].values)) != set(map(tuple, b[cols].values)):
        raise ValueError("Review sheets cover different claims")
    allowed = {"supported", "partial", "unsupported", "unverifiable"}
    for frame in [a, b]:
        if not frame.support_status.isin(allowed).all() or (frame.reviewer == "").any():
            raise ValueError(
                "Every claim requires an actual identified reviewer and completed support judgment"
            )
    data = a.merge(b, on=cols, suffixes=("_1", "_2"), validate="one_to_one")
    same = data.support_status_1 == data.support_status_2
    data[~same].to_csv(out.with_name(out.stem + "_adjudication.csv"), index=False)
    save(
        out,
        {
            "claims": len(data),
            "support_agreement": float(same.mean()),
            "support_cohen_kappa": float(
                cohen_kappa_score(data.support_status_1, data.support_status_2)
            ),
            "disagreements": int((~same).sum()),
            "note": "Disagreements require adjudication before final quality scoring. No automatic human-label filling.",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.first, a.second, a.out)
