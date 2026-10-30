"""Experiment reports, figures, and reproducibility archives."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from protocol import MODELS, sha, save


def compile_report(run, out, code):
    rag = out / "extended_rag/rag_ablation"
    automatic = json.loads((rag / "automatic_metrics.json").read_text())
    quality = json.loads((rag / "automated_summary.json").read_text())
    selection = json.loads((out / "model_selection.json").read_text())
    anova = json.loads((out / "anova.json").read_text())
    intervals = json.loads((run / "cluster_bootstrap.json").read_text())
    complete = quality["groundedness"]
    reward = quality["failure_adjusted_supported_claim_reward"]
    rows = []
    for kind in MODELS:
        scores = json.loads((run / f"{kind}_oof_metrics.json").read_text())
        ci = intervals["f1_cluster95"][kind]
        rows.append(
            {
                "model": kind,
                **scores,
                "F1_cluster95_low": ci[0],
                "F1_cluster95_high": ci[1],
            }
        )
    pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)
    table = pd.read_csv(rag / "automated_sample_metrics.csv")
    recommendations = pd.read_csv(rag / "automated_recommendation_review.csv")
    rec_summary = (
        recommendations.groupby("arm")
        .verdict.value_counts()
        .unstack(fill_value=0)
        .to_dict("index")
    )
    save(
        out / "recommendation_audit.json",
        {"mode": "automated proxy, not human validation", "counts": rec_summary},
    )
    plt.figure(figsize=(8, 5))
    labels = ["with_rag", "without_rag"]
    means = [
        complete["with_rag_complete_pair_mean"],
        complete["without_rag_complete_pair_mean"],
    ]
    plt.bar(labels, means, color=["#277a9c", "#c27a48"])
    plt.ylim(0, 1)
    plt.ylabel("Fully supported claim fraction")
    plt.title(f'Automated proxy: {complete["paired_samples"]} complete report pairs')
    plt.tight_layout()
    plt.savefig(out / "figures/rag_groundedness.png", dpi=160)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        [automatic["arms"][a]["mean_total_seconds"] for a in labels],
        color=["#277a9c", "#c27a48"],
    )
    plt.ylabel("Mean end-to-end seconds")
    plt.title("RAG latency, all attempted reports")
    plt.tight_layout()
    plt.savefig(out / "figures/rag_latency.png", dpi=160)
    plt.close()
    effects = (
        np.array([complete["with_minus_without"], reward["with_minus_without"]]) * 100
    )
    ci = np.array([complete["paired_sample_bootstrap95"], reward["bootstrap95"]]) * 100
    plt.figure(figsize=(9, 4))
    plt.errorbar(
        effects,
        [1, 0],
        xerr=np.vstack([effects - ci[:, 0], ci[:, 1] - effects]),
        fmt="o",
        capsize=5,
        color="#267a9c",
    )
    plt.axvline(0, color="black", linestyle="--")
    plt.yticks(
        [1, 0], ["Valid report pairs only", "All pairs, failures receive zero reward"]
    )
    plt.xlabel(
        "With minus without RAG (percentage points), paired bootstrap 95% interval"
    )
    plt.title("Automated quality proxy: independent human validation pending")
    plt.tight_layout()
    plt.savefig(out / "figures/rag_effect_intervals.png", dpi=160)
    plt.close()
    comparison = [
        "| Model | Accuracy | Attack F1 | ROC-AUC | Conditional cluster F1 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        comparison.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | [{r['F1_cluster95_low']:.4f}, {r['F1_cluster95_high']:.4f}] |"
        )
    importance = pd.read_csv(out / "feature_importance.csv")
    fidelity = pd.read_csv(out / "lime_fidelity.csv")
    chosen = selection["selected"]["model"]
    top = (
        importance[(importance.model == chosen) & (importance.method == "SHAP")]
        .sort_values("absolute_contribution", ascending=False)
        .head(5)
        .feature.tolist()
    )
    qtable = ["| Measure | With RAG | Without RAG |", "|---|---:|---:|"]
    qtable.append(
        f"| Valid generations / attempts | {automatic['arms']['with_rag']['successes']} / 50 | {automatic['arms']['without_rag']['successes']} / 50 |"
    )
    qtable.append(
        f"| Mean latency, seconds, all attempts | {automatic['arms']['with_rag']['mean_total_seconds']:.3f} | {automatic['arms']['without_rag']['mean_total_seconds']:.3f} |"
    )
    for metric, title in [
        ("groundedness", "Fully supported claim fraction"),
        ("partial_rate", "Partially supported fraction"),
        ("unsupported_rate", "Fully unsupported fraction"),
        ("not_fully_supported_rate", "Not fully supported fraction"),
    ]:
        q = quality[metric]
        qtable.append(
            f"| {title}, complete pairs, automated | {q['with_rag_complete_pair_mean']:.2%} | {q['without_rag_complete_pair_mean']:.2%} |"
        )
    qtable.append(
        f"| Failure-adjusted supported-claim reward | {reward['arm_means']['with_rag']:.2%} | {reward['arm_means']['without_rag']:.2%} |"
    )
    for arm in labels:
        value = automatic["arms"][arm]
        assert value["attempts"] == 50
    sample = json.loads((out / "extended_rag/sampling.json").read_text())
    manifest = pd.read_csv(run / "manifest.csv").set_index("row_id")
    ids = sample["row_ids"]
    assert (
        len(ids) == len(set(ids)) == 50 and manifest.loc[ids, "group"].nunique() == 50
    )
    assert not set(manifest.loc[ids, "group"]) & set(sample["excluded_pilot_groups"])
    # Every included review row must correspond to an actual successful report claim.
    reviewed = pd.read_csv(rag / "automated_claim_review.csv")
    keys = pd.read_csv(rag / "review_key.csv")
    audit = reviewed.merge(keys, on="blind_id", validate="many_to_one")
    for (row_id, arm), group in audit.groupby(["row_id", "arm"]):
        item = json.loads((rag / f"{row_id}_{arm}.json").read_text())
        assert item["status"] == "success"
        assert set(group.claim_index) == set(range(len(item["report"]["claims"])))
    splits = np.load(run / "splits.npz")
    explanation_count = 0
    for kind in MODELS:
        for fold in range(1, 6):
            original = run / "models" / f"{kind}_fold{fold}"
            copied = out / "explanations/models" / f"{kind}_fold{fold}"
            assert sha(original / "model.pt") == sha(copied / "model.pt")
            exp = np.load(copied / "shap.npz")
            assert len(exp["row_ids"]) == 20 and np.isfinite(exp["values"]).all()
            assert set(exp["row_ids"]).issubset(set(splits[f"test_{fold}"])) and set(
                exp["background_ids"]
            ).issubset(set(splits[f"train_{fold}"]))
            lime = json.loads((copied / "lime.json").read_text())
            assert {x["row_id"] for x in lime} == set(exp["row_ids"])
            assert all(np.isfinite(x["local_surrogate_r2"]) for x in lime)
            explanation_count += len(lime)
    assert explanation_count == 500
    raw_pairs = []
    for row_id in ids:
        a = json.loads((rag / f"{row_id}_with_rag.json").read_text())
        b = json.loads((rag / f"{row_id}_without_rag.json").read_text())
        assert a["query"] == b["query"] and not b["evidence"]
        raw_pairs.append((a, b))
    lines = [
        "# SOC-RAG Guard — research evaluation report",
        "",
        "Date: 30 October 2026. All Python execution used the repository `.venv`. This report presents observed results; it does not assume that RAG must improve quality.",
        "",
        "## Completion checklist",
        "",
        "| Requested work | Delivered |",
        "|---|---|",
        "| Five deep models | CNN, LSTM, CNN-LSTM, BiLSTM, Transformer; 25 shared grouped outer-fold fits |",
        "| SHAP and LIME | Each method: 100 held-out examples per architecture, 500 total; raw values, original-feature summaries and plots |",
        "| Finalize models | All comparison checkpoints retained; validation-selected Transformer refitted on all data and tested for reload/inference |",
        "| ANOVA | Fold-block repeated-measures ANOVA, Greenhouse-Geisser correction, explicitly exploratory |",
        "| t-test, p-values, intervals | CV-dependence-corrected paired t, Holm correction, fold intervals, group-bootstrap intervals; paired RAG intervals and exploratory sign-flip tests |",
        "| ROC and confusion matrices | All 25 fold plots plus pooled ROC and combined confusion matrices |",
        "| With/without RAG | 50 fresh matched group-sampled flows, 100 generation attempts; actual failures, latency, citations, automated quality and recommendation audits |",
        "| Independently prove RAG quality enhancement | Not established by this automated study; human review materials supplied and limitations reported |",
        "",
        "## Neural comparison",
        "",
        *comparison,
        "",
        "The input is an ordered vector of 17 tabular flow features, not a chronological traffic sequence. IPs, ports, target label and attack type are excluded from detector inputs. Exact feature signatures and bidirectional connection groups are kept together transitively. All preprocessing fits inside the training partition. The dataset has 211,043 rows; severe group imbalance causes unequal folds and limits inference. Existing Random Forest values remain historical baselines only.",
        "",
        f'Exploratory F1 repeated-measures ANOVA: F(4,16)={anova["results"]["f1"]["F"]:.4f}; uncorrected p={anova["results"]["f1"]["p_uncorrected"]:.6f}; Greenhouse-Geisser p={anova["results"]["f1"]["p_GG"]:.6f}. Overlapping training folds violate independent-replicate assumptions; the correction for sphericity does not resolve CV dependence. Corrected paired F1 comparisons in the original run found no Holm-adjusted significant winner. These results do not prove equivalence.',
        "",
        f'Final selection: {chosen}, mean inner-validation BCE={selection["selected"]["mean_inner_validation_BCE"]:.6f}, refit for {selection["selected"]["final_epochs"]} epochs on all rows. Selection used validation scores, not outer test ranking. The selection rule was formalized after the initial experiment and is not preregistered. The final refit has no new independent test performance estimate.',
        "",
        "## Feature explanations",
        "",
        f"Top five {chosen} SHAP features in the fixed held-out sample: "
        + ", ".join(top)
        + ".",
        "",
        "Categorical one-hot contributions are summed back to their original feature before taking absolute importance. LIME aggregation includes its selected top-10 encoded terms; absent terms are treated as zero and cancellation can affect grouped ranks. These are sample summaries, not population-level or causal effects. Perturbations can violate one-hot/correlation structure. LIME fidelity must be checked before interpreting an example: mean local R2 is "
        + ", ".join(
            f"{k}={v:.3f}" for k, v in fidelity.groupby("model").local_R2.mean().items()
        )
        + ". Low-fidelity examples are illustrative only.",
        "",
        "## Extended RAG comparison",
        "",
        "Ten new distinct groups per fold, sampled with seed 20261030 before generating reports, excluding all original pilot groups. Each arm receives identical held-out telemetry and the same fixed CNN OOF probability. The model, system prompt, schema, temperature, seed and 512-token cap are shared; arm order is randomized. Only reference evidence differs. Results describe a uniform group sample with the first row representing each selected group, not a row-weighted deployment population.",
        "",
        *qtable,
        "",
        f'Complete-pair automated groundedness difference (with minus without): {complete["with_minus_without"]:.4f}; paired bootstrap 95% interval {complete["paired_sample_bootstrap95"]}; exploratory sign-flip p={complete["paired_sign_flip_p_exploratory"]:.6f}, Holm p={complete["p_holm"]:.6f}. Only {complete["paired_samples"]} pairs have two valid reports. Complete-pair quality is subject to survivor bias; the separate failure-adjusted reward retains all 50 pairs.',
        "",
        f'All-pair failure-adjusted reward difference: {reward["with_minus_without"]:.4f}; paired bootstrap 95% interval {reward["bootstrap95"]}. A failed report receives zero usability reward, not an invented unsupported-claim fraction.',
        "",
        "Quality figures are automated proxies from the same Qwen model family as the generator. They are not independent human adjudication. The revised judge separates telemetry support from reference entailment; raw decisions and validation rules are saved. The original failed pilot judge remains preserved. Citation entailment is assessed only for reference text that was actually retrieved; no-RAG field-name references are not supplied corpus IDs, so ID compliance is not a neutral standalone measure of factual quality.",
        "",
        "Some reports hit the shared token cap and returned incomplete JSON. These attempts are counted as failures and retained with raw responses. Both arms were tested under the same cap. No failed generation was selectively repaired.",
        "",
        "The frozen corpus has only three local summaries, and top-k=3 returns the entire corpus. This experiment measures evidence inclusion, not retrieval ranking. It does not independently establish the original documents' correctness. Source revisions, environment shifts and external traffic remain untested. RAG does not alter detector decisions, so detector classification accuracy is unchanged by design.",
        "",
        "A recommendation-specific automated audit is also provided, because unsupported specificity may occur outside the claims array. Human validation should review both sections.",
        "",
        "## Independent review and reproducibility",
        "",
        "Two blank reviewer sheets and a blinded report packet are in extended_rag/rag_ablation/. Actual human labels are intentionally unfilled. reviewer_agreement.py checks coverage and computes agreement/kappa after real reviews, producing a disagreement sheet for adjudication. No human agreement statistic has been invented.",
        "",
        "Reproduction commands and final-model inference are in RUNBOOK.md. Tests are recorded in tests.txt. Original grouped CV files remain under ../comparison_v1/. Artifact hashes and the research artifact archive accompany this report.",
        "",
        "Statistical reference: https://www.statsmodels.org/stable/generated/statsmodels.stats.anova.AnovaRM.html ; dependence-aware comparison reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html .",
    ]
    lines += [
        "",
        "Automated audit caution: spot-checking found a false-positive pattern in which a request to verify threat intelligence was judged unjustified because intelligence was not already supplied. Such recommendations can be reasonable validation steps. Claim support can also be overgenerous for compound interpretations. Both support and recommendation verdict counts are diagnostic estimates, not validated quality measurements. Review the original texts before making a scientific claim from those counts.",
        "",
        "The RAG experiment intentionally retains the prespecified CNN OOF detector for comparability. It does not evaluate the new full-data Transformer refit plus RAG as an independently tested end-to-end deployed system.",
    ]
    (out / "RESEARCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(
        out / "verification.json",
        {
            "models_compared": 5,
            "grouped_fits": 25,
            "SHAP_explanations": 500,
            "LIME_explanations": 500,
            "RAG_attempted_pairs": 50,
            "distinct_RAG_groups": 50,
            "pilot_groups_excluded": True,
            "paired_queries_identical": True,
            "reviewed_claims": len(reviewed),
            "human_review_completed": False,
            "final_model_reload": json.loads(
                (out / "final_model/verification.json").read_text()
            ),
        },
    )
    text = """# Reproduction and review

Use D:\\PROJECTS\\SOC-RAG-Guard\\.venv\\Scripts\\python.exe for every command below. Run from the repository root.

```powershell
python src/experiments/research_analysis.py prepare_explanations --run results/comparison_v1 --out results/research_evaluation_v1
python src/experiments/explain.py --run results/research_evaluation_v1/explanations --count 20
python src/experiments/research_analysis.py analysis --run results/comparison_v1 --out results/research_evaluation_v1
python src/experiments/research_analysis.py finalize --run results/comparison_v1 --out results/research_evaluation_v1
python src/experiments/extended_rag.py generate --source results/comparison_v1 --out results/research_evaluation_v1/extended_rag
python src/experiments/extended_rag.py judge --source results/comparison_v1 --out results/research_evaluation_v1/extended_rag
python src/experiments/generate_report.py --run results/comparison_v1 --out results/research_evaluation_v1 --code src/experiments
python -m unittest discover -s tests -p "test_*.py"
```

`prepare_explanations` requires a new destination; completed RAG generations and audits resume from saved outputs. Original CV training instructions are in src/experiments/README.md. Do not mix reruns with different settings. The original dataset is excluded from the submission archive; its SHA256 is recorded in protocol.json. The original code, package environment, seed, indices and artifacts are retained. GPU/CPU numerical differences are possible.

```powershell
python src/experiments/predict_final.py --bundle results/research_evaluation_v1/final_model --input your_flow_features.csv --output new_predictions.csv
```

Input must contain the 17 features listed in protocol.py; target labels are not required. The output must be a new file. Predictions are investigative leads. The final refit has no new independent test score.

For human review, give each reviewer human_review_packet.json, frozen_corpus.json, its own human_reviewer_N.csv, and the rubric in automated_audit_v2/settings.json. Withhold review_key.csv and all automated labels. Citation content can still reveal the arm. Review all factual claims and recommendations. Independent reviewers must decide support_status, citation_entails and record their identity and rationale. Compare the two completed sheets using reviewer_agreement.py, adjudicate disagreements, and then copy real adjudicated labels into claim_review.csv to run rag_ablation.py score. Do not label automated or assistant judgments as human.
"""
    (out / "RUNBOOK.md").write_text(text, encoding="utf-8")
    (rag / "HUMAN_REVIEW_GUIDE.md").write_text(
        """# Independent review guide

Review independently before comparing labels. Keep automated judgments, arm keys and other reviewers' answers out of view. Citation content may still reveal the condition.

Use the supplied telemetry and frozen corpus as the reference standard. Groundedness is support by these inputs, not proof of a real incident. Supported: every factual part is supported. Partial: at least one part is supported and at least one part is not. Unsupported: asserted without evidential basis or contradicted. Unverifiable: too ambiguous to decide. Record a reason for each decision. Compound claims should be judged conservatively or explicitly split and reconciled before scoring.

Literal byte/packet values can be supported by telemetry. Zero payload bytes do not establish zero IP bytes. Undefined state abbreviations cannot be expanded using assumed meanings under this evidence-only protocol. A detector score alone does not establish an attack or explain its cause.

Citation entailment is separate: a general guidance document cannot establish a particular flow's numeric score. Use yes/no for cited claims, na for claims with no citation. Field-name references are not supplied corpus source IDs, but can still point to supported telemetry facts. Do not confuse source-ID compliance with factual support.

A suggestion to CHECK whether malicious activity exists does not assert that it exists. Missing telemetry can justify requesting it. Flag unsupported specificity when a particular technique, indicator or incident is assumed as fact without evidence. Audit factual assertions embedded inside recommendations as well as the claims array. Record recommendation judgments separately; the supplied automatic labels are not validated human judgments.

Fill support_status, citation_entails, reviewer and notes in your own CSV. Reviewer must identify the actual person; leave unresolved items explicitly unresolved. After both reviews, use reviewer_agreement.py and adjudicate disagreements. Do not replace failed generation attempts or silently remove them from success-rate reporting.
""",
        encoding="utf-8",
    )
    shutil.copy2(run / "comparison_roc.png", out / "figures/comparison_roc.png")
    save(out / "code_hashes.json", {p.name: sha(p) for p in code.glob("*.py")})
    hashes = {
        str(p.relative_to(out)): sha(p)
        for p in out.rglob("*")
        if p.is_file()
        and p.name not in ["artifact_hashes.json", "research_artifacts.zip"]
    }
    save(out / "artifact_hashes.json", hashes)
    with zipfile.ZipFile(
        out / "research_artifacts.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in out.rglob("*"):
            if p.is_file() and p.name != "research_artifacts.zip":
                archive.write(p, str(p.relative_to(out)))
        for p in code.glob("*.py"):
            archive.write(p, "reproduction_code/" + p.name)
        for name in [
            "fold_metrics.csv",
            "statistics.json",
            "cluster_bootstrap.json",
            "protocol.json",
            "audit.json",
            "environment.txt",
            "manifest.csv",
            "splits.npz",
        ]:
            archive.write(run / name, "original_comparison/" + name)
        for kind in MODELS:
            archive.write(
                run / f"{kind}_oof.csv", "original_comparison/" + f"{kind}_oof.csv"
            )
            for fold in range(1, 6):
                archive.write(
                    run / "models" / f"{kind}_fold{fold}" / "history.json",
                    f"original_comparison/history/{kind}_fold{fold}.json",
                )
    print("Report, verification and submission archive completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--code", type=Path, required=True)
    a = p.parse_args()
    compile_report(a.run, a.out, a.code)
