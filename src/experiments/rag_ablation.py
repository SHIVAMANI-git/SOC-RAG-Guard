"""Matched RAG generation, claim-review scoring, and paired inference."""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from protocol import save, sha, FEATURES

SYSTEM = "You are a cautious SOC triage assistant. Treat detector output as uncertain, not ground truth. Use only supplied telemetry and evidence for factual claims. Do not invent events or claim checks were performed. Give atomic claims with source_ids, and separate recommended checks. Cite only supplied IDs. Return JSON with claims (list of objects with text and source_ids) and recommended_checks (list of strings)."
SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "source_ids"],
            },
        },
        "recommended_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims", "recommended_checks"],
}


def generate(run, root, n, model, sample_ids=None):
    sys.path.insert(0, str(root / "rag"))
    import retriever

    out = run / "rag_ablation"
    out.mkdir(exist_ok=True)
    retriever.CACHE_FILE = out / "embeddings_cache.json"
    config = json.loads((run / "protocol.json").read_text())
    if sha(config["data"]) != config["data_sha256"]:
        raise ValueError("Dataset changed")
    df = pd.read_csv(config["data"], low_memory=False)
    manifest = pd.read_csv(run / "manifest.csv")
    # Predeclared CNN; never select detector or samples based on observed correctness.
    predictions = pd.concat(
        [
            pd.read_csv(run / "models" / f"cnn_fold{f}" / "predictions.csv")
            for f in range(1, 6)
        ]
    ).set_index("row_id")
    rng = np.random.default_rng(42)
    ids = []
    for fold in range(1, 6):
        pool = manifest[manifest.fold == fold].drop_duplicates("group")
        ids.extend(
            rng.choice(
                pool.row_id.to_numpy(), min(n, len(pool)), replace=False
            ).tolist()
        )
    if sample_ids is not None:
        ids = [int(i) for i in sample_ids]
        if (
            len(ids) != len(set(ids))
            or manifest.set_index("row_id").loc[ids, "group"].duplicated().any()
        ):
            raise ValueError("Samples must have distinct row and group IDs")
    kb = root / "rag" / "knowledge_base" / "cybersecurity_evidence.json"
    tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=10)
    tags.raise_for_status()
    settings = {
        "sample_ids": ids,
        "detector": "cnn fixed before comparison",
        "per_fold": n,
        "generation_model": model,
        "model_inventory": tags.json(),
        "temperature": 0,
        "seed": 42,
        "num_predict": 512,
        "kb_sha256": sha(kb),
        "prompt": SYSTEM,
        "schema": SCHEMA,
        "evaluation": "paired identical out-of-fold telemetry; only supplied evidence differs; randomized arm order; semantic review pending",
        "latency": "end-to-end including retrieval; excludes model warmup; generation also timed separately",
    }
    if (out / "settings.json").exists():
        old = json.loads((out / "settings.json").read_text())
        for key in ["sample_ids", "generation_model", "kb_sha256", "prompt", "schema"]:
            if old[key] != settings[key]:
                raise ValueError("Ablation configuration changed")
    save(out / "settings.json", settings)
    (out / "frozen_corpus.json").write_bytes(kb.read_bytes())
    # Load generation model and embedding model outside timed trials.
    warm = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": model, "prompt": "", "keep_alive": "30m"},
        timeout=240,
    )
    warm.raise_for_status()
    retriever.retrieve("network incident validation", top_k=3)
    for row_id in ids:
        flow = df.loc[row_id, FEATURES].to_dict()
        probability = float(predictions.loc[row_id, "probability"])
        query = (
            "Assess this network flow and recommend validation checks. Telemetry: "
            + json.dumps(flow)
            + " Detector attack probability: "
            + str(probability)
        )
        for arm in rng.permutation(["with_rag", "without_rag"]):
            path = out / f"{row_id}_{arm}.json"
            if path.exists():
                continue
            start = time.perf_counter()
            evidence = retriever.retrieve(query, 3) if arm == "with_rag" else []
            retrieval_seconds = time.perf_counter() - start
            prompt = json.dumps({"query": query, "evidence": evidence})
            genstart = time.perf_counter()
            raw = None
            try:
                response = requests.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={
                        "model": model,
                        "stream": False,
                        "think": False,
                        "format": SCHEMA,
                        "keep_alive": "30m",
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "options": {"temperature": 0, "seed": 42, "num_predict": 512},
                    },
                    timeout=240,
                )
                response.raise_for_status()
                raw = response.json()
                content = raw["message"]["content"]
                report = json.loads(content)
                if not isinstance(report.get("claims"), list) or not report["claims"]:
                    raise ValueError("No claims returned")
                for claim in report["claims"]:
                    if not isinstance(claim.get("text"), str) or not isinstance(
                        claim.get("source_ids"), list
                    ):
                        raise ValueError("Invalid claim schema")
                allowed = {e["source_id"] for e in evidence}
                cites = [s for c in report["claims"] for s in c["source_ids"]]
                result = {
                    "status": "success",
                    "report": report,
                    "citation_count": len(cites),
                    "valid_citation_count": sum(s in allowed for s in cites),
                    "cited_claim_count": sum(
                        bool(c["source_ids"]) for c in report["claims"]
                    ),
                    "generation_seconds": time.perf_counter() - genstart,
                    "server_total_duration_ns": raw.get("total_duration"),
                    "raw_response": raw,
                }
            except Exception as exc:
                result = {
                    "status": "error",
                    "error": str(exc),
                    "generation_seconds": time.perf_counter() - genstart,
                    "raw_response": raw,
                }
            save(
                path,
                dict(
                    result,
                    row_id=row_id,
                    arm=arm,
                    fold=int(manifest.loc[row_id, "fold"]),
                    group=int(manifest.loc[row_id, "group"]),
                    query=query,
                    evidence=evidence,
                    retrieval_seconds=retrieval_seconds,
                    total_seconds=time.perf_counter() - start,
                ),
            )
            print(row_id, arm, result["status"], flush=True)
    review = []
    keys = []
    results = []
    for row_id in ids:
        for arm in ["with_rag", "without_rag"]:
            item = json.loads((out / f"{row_id}_{arm}.json").read_text())
            results.append(item)
            if item["status"] != "success":
                continue
            blind = uuid.uuid5(uuid.NAMESPACE_URL, str(row_id) + arm).hex[:12]
            keys.append({"blind_id": blind, "row_id": row_id, "arm": arm})
            for index, claim in enumerate(item["report"]["claims"]):
                review.append(
                    {
                        "blind_id": blind,
                        "claim_index": index,
                        "claim": claim["text"],
                        "source_ids": json.dumps(claim["source_ids"]),
                        "support_status": "",
                        "citation_entails": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )
    review_path = out / "claim_review.csv"
    if not review_path.exists():
        pd.DataFrame(review).sample(frac=1, random_state=42).to_csv(
            review_path, index=False
        )
    pd.DataFrame(keys).to_csv(out / "review_key.csv", index=False)
    latency = pd.DataFrame(results).pivot(
        index="row_id", columns="arm", values="total_seconds"
    )
    delta = (latency.with_rag - latency.without_rag).to_numpy()
    boot_rng = np.random.default_rng(42)
    boot = [
        boot_rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)
    ]
    save(
        out / "automatic_metrics.json",
        {
            "semantic_groundedness": None,
            "unsupported_claim_rate": None,
            "reason": "Must adjudicate atomic claims against telemetry and frozen reference corpus. Citation ID validity is NOT entailment.",
            "latency_paired_difference": {
                "with_minus_without_seconds": float(delta.mean()),
                "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "includes_failed_attempts": True,
            },
            "arms": {
                arm: {
                    "successes": sum(
                        r["status"] == "success" for r in results if r["arm"] == arm
                    ),
                    "attempts": sum(r["arm"] == arm for r in results),
                    "mean_total_seconds": float(
                        np.mean(
                            [r["total_seconds"] for r in results if r["arm"] == arm]
                        )
                    ),
                    "citation_count": sum(
                        r.get("citation_count", 0) for r in results if r["arm"] == arm
                    ),
                    "valid_citation_count": sum(
                        r.get("valid_citation_count", 0)
                        for r in results
                        if r["arm"] == arm
                    ),
                }
                for arm in ["with_rag", "without_rag"]
            },
        },
    )


def score(run, automated=False, assistant=False):
    out = run / "rag_ablation"
    claims = pd.read_csv(
        out
        / (
            "assistant_claim_review.csv"
            if assistant
            else "automated_claim_review.csv" if automated else "claim_review.csv"
        )
    ).fillna("")
    keys = pd.read_csv(out / "review_key.csv")
    data = claims.merge(keys, on="blind_id", validate="many_to_one")
    expected = set()
    for _, key in keys.iterrows():
        item = json.loads((out / f"{key.row_id}_{key.arm}.json").read_text())
        expected.update((key.blind_id, i) for i in range(len(item["report"]["claims"])))
    actual = list(zip(data.blind_id, data.claim_index))
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Review must cover every generated claim exactly once")
    allowed = {"supported", "partial", "unsupported", "unverifiable"}
    if not data.support_status.isin(allowed).all() or (data.reviewer == "").any():
        raise ValueError("All claims require support status and identified reviewer")
    cited = data.source_ids.apply(lambda value: bool(json.loads(value)))
    if not data.loc[cited, "citation_entails"].isin(["yes", "no"]).all():
        raise ValueError("Cited claims require citation_entails yes/no")
    citation_summary = {}
    for arm, group in data[cited].groupby("arm"):
        citation_summary[arm] = {
            "cited_claims": len(group),
            "claim_citation_entailment_precision": float(
                (group.citation_entails == "yes").mean()
            ),
        }
    rows = []
    for (row_id, arm), group in data.groupby(["row_id", "arm"]):
        rows.append(
            {
                "row_id": row_id,
                "arm": arm,
                "groundedness": float((group.support_status == "supported").mean()),
                "unsupported_rate": float(
                    (group.support_status == "unsupported").mean()
                ),
                "partial_rate": float((group.support_status == "partial").mean()),
                "not_fully_supported_rate": float(
                    (group.support_status != "supported").mean()
                ),
                "unverifiable_rate": float(
                    (group.support_status == "unverifiable").mean()
                ),
            }
        )
    prefix = "assistant" if assistant else "automated" if automated else "human"
    samples = pd.DataFrame(rows)
    samples.to_csv(out / f"{prefix}_sample_metrics.csv", index=False)
    results = {
        "citation_entailment": citation_summary,
        "review_mode": prefix,
        "warning": "Assistant and automated reviews are provisional, not independent human validation. Assistant review is non-blind. Intervals conditional on sampled cases.",
        "arm_means": samples.groupby("arm")[
            [
                "groundedness",
                "unsupported_rate",
                "partial_rate",
                "not_fully_supported_rate",
                "unverifiable_rate",
            ]
        ]
        .mean()
        .to_dict("index"),
    }
    rng = np.random.default_rng(42)
    for metric in [
        "groundedness",
        "unsupported_rate",
        "partial_rate",
        "not_fully_supported_rate",
        "unverifiable_rate",
    ]:
        pair = samples.pivot(index="row_id", columns="arm", values=metric).dropna()
        d = (pair.with_rag - pair.without_rag).to_numpy()
        if len(d) < 2:
            raise ValueError("Need at least two completed pairs")
        boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
        signs = rng.choice([-1, 1], size=(10000, len(d)))
        null = np.abs((signs * d).mean(axis=1))
        p_value = float((1 + np.sum(null >= abs(d.mean()) - 1e-12)) / (len(null) + 1))
        results[metric] = {
            "paired_samples": len(d),
            "with_rag_complete_pair_mean": float(pair.with_rag.mean()),
            "without_rag_complete_pair_mean": float(pair.without_rag.mean()),
            "with_minus_without": float(d.mean()),
            "paired_sample_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "paired_sign_flip_p_exploratory": p_value,
        }
    order = sorted(
        [
            results[k]
            for k in [
                "groundedness",
                "unsupported_rate",
                "partial_rate",
                "not_fully_supported_rate",
                "unverifiable_rate",
            ]
        ],
        key=lambda r: r["paired_sign_flip_p_exploratory"],
    )
    last = 0
    for j, r in enumerate(order):
        last = max(last, min(1, (len(order) - j) * r["paired_sign_flip_p_exploratory"]))
        r["p_holm"] = last
    settings = json.loads((out / "settings.json").read_text())
    reward_rows = []
    for row_id in settings["sample_ids"]:
        for arm in ["with_rag", "without_rag"]:
            item = json.loads((out / f"{row_id}_{arm}.json").read_text())
            found = samples[(samples.row_id == row_id) & (samples.arm == arm)]
            reward_rows.append(
                {
                    "row_id": row_id,
                    "arm": arm,
                    "reward": (
                        float(found.groundedness.iloc[0])
                        if item["status"] == "success" and len(found) == 1
                        else 0.0
                    ),
                    "generation_success": item["status"] == "success",
                }
            )
    reward = pd.DataFrame(reward_rows)
    paired = reward.pivot(index="row_id", columns="arm", values="reward")
    d = (paired.with_rag - paired.without_rag).to_numpy()
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(5000)]
    results["failure_adjusted_supported_claim_reward"] = {
        "definition": "Supported-claim fraction for valid reports; zero reward for generation failures. This is a usability-adjusted reward, not a claim fraction on failed reports.",
        "attempted_pairs": len(paired),
        "arm_means": reward.groupby("arm").reward.mean().to_dict(),
        "with_minus_without": float(d.mean()),
        "bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
    }
    save(out / f"{prefix}_summary.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["generate", "score"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--per-fold", type=int, default=2)
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--automated", action="store_true")
    p.add_argument("--assistant", action="store_true")
    args = p.parse_args()
    if args.action == "generate":
        generate(args.run, args.root, args.per_fold, args.model)
    else:
        score(args.run, args.automated, args.assistant)
