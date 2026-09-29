import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

from trustprotkg.cross_protein import (
    apply_split_strategy,
    run_cross_protein_experiment,
)
from trustprotkg.evaluation import compute_grouped_classification_metrics


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CONFIG = ROOT / "experiments" / "configs" / "v0.6_all_leave_one_protein_summary.yaml"
TP53_CONFIG = ROOT / "experiments" / "configs" / "v0.6_leave_tp53_out.yaml"


def _toy_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "protein_id": ["TP53_HUMAN", "TP53_HUMAN", "BRCA1_HUMAN", "PTEN_HUMAN"],
            "split": ["train", "test", "train", "test"],
        }
    )


def test_apply_split_strategy_supports_leave_one_protein_out():
    split = apply_split_strategy(
        _toy_feature_table(),
        strategy="leave_one_protein_out",
        holdout_protein_id="TP53_HUMAN",
    )

    tp53_rows = split[split["protein_id"] == "TP53_HUMAN"]
    other_rows = split[split["protein_id"] != "TP53_HUMAN"]
    assert set(tp53_rows["eval_split"]) == {"test"}
    assert set(other_rows["eval_split"]) == {"train"}


def test_apply_split_strategy_supports_user_defined_split():
    user_split = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "split": ["train", "train", "test", "test"],
        }
    )
    split = apply_split_strategy(
        _toy_feature_table(),
        strategy="user_defined",
        user_split=user_split,
    )

    assert split.set_index("variant_id").loc["v3", "eval_split"] == "test"
    assert split.set_index("variant_id").loc["v1", "eval_split"] == "train"


def test_grouped_classification_metrics_report_nan_auroc_for_one_class_groups():
    predictions = pd.DataFrame(
        {
            "protein_id": ["TP53_HUMAN", "TP53_HUMAN", "PTEN_HUMAN", "PTEN_HUMAN"],
            "label_binary": [1, 1, 0, 1],
            "prob_pathogenic": [0.8, 0.7, 0.2, 0.9],
        }
    )

    grouped = compute_grouped_classification_metrics(
        predictions,
        group_columns=["protein_id"],
        label_column="label_binary",
    ).set_index("protein_id")

    assert math.isnan(grouped.loc["TP53_HUMAN", "auroc"])
    assert grouped.loc["PTEN_HUMAN", "auroc"] == 1.0


def test_cross_protein_summary_generates_grouped_outputs_and_report():
    result = run_cross_protein_experiment(SUMMARY_CONFIG)

    assert result.run_count == 4
    for path in [
        result.metrics,
        result.predictions,
        result.per_protein_metrics,
        result.failure_modes,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["benchmark_config"] == "configs\\real_snapshot_benchmark.yaml"
    assert metadata["dataset_summary"] == {
        "families": 0,
        "proteins": 3,
        "variants": 9,
    }

    metrics = pd.read_csv(result.metrics)
    assert set(metrics["split_strategy"]) == {
        "existing_split",
        "leave_one_protein_out",
    }
    assert set(metrics["holdout_protein_id"].dropna()) >= {
        "BRCA1_HUMAN",
        "PTEN_HUMAN",
        "TP53_HUMAN",
    }
    assert set(metrics["feature_set"]) == {
        "structure_only",
        "kg_only",
        "structure_plus_kg",
        "no_domain",
        "no_structural_contacts",
    }

    per_protein = pd.read_csv(result.per_protein_metrics)
    assert {"accuracy", "precision", "recall", "f1", "auroc"}.issubset(
        per_protein.columns
    )
    assert set(per_protein["protein_id"]) == {"BRCA1_HUMAN", "PTEN_HUMAN", "TP53_HUMAN"}
    one_class_rows = per_protein[per_protein["n_test"] < 3]
    assert one_class_rows["auroc"].isna().any()

    failure_modes = pd.read_csv(result.failure_modes)
    assert {
        "false_positives",
        "false_negatives",
        "low_confidence_structure_cases",
        "sparse_kg_context_cases",
        "missing_domain_context_cases",
    }.issubset(failure_modes.columns)
    assert failure_modes["low_confidence_structure_cases"].sum() >= 0

    report = result.report.read_text(encoding="utf-8")
    for section in [
        "## Split Strategy Summary",
        "## Random Split Performance",
        "## Leave-One-Protein-Out Performance",
        "## Per-Protein Metrics",
        "## Feature-Set Robustness Across Proteins",
        "## Failure-Mode Summary",
    ]:
        assert section in report


def test_cross_protein_cli_executes_single_holdout_config():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustprotkg.cross_protein",
            "--config",
            str(TP53_CONFIG),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TrustProtKG cross-protein experiment complete" in completed.stdout
    metadata_path = ROOT / "experiments" / "runs" / "v0.6_leave_tp53_out" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["split_strategy"] == "leave_one_protein_out"
    assert metadata["split_runs"][0]["holdout_protein_id"] == "TP53_HUMAN"







































