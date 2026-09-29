import json
from pathlib import Path

import pandas as pd

from trustprotkg.cross_protein import apply_split_strategy, run_cross_protein_experiment
from trustprotkg.explanation_quality import run_explanation_quality_evaluation
from trustprotkg.ingestion import run_ingestion, verify_manifest


ROOT = Path(__file__).resolve().parents[1]
INGESTION_CONFIG = ROOT / "configs" / "family_snapshot_ingestion.yaml"
MANIFEST = ROOT / "data" / "snapshots" / "v0.8_family" / "manifest.csv"
CROSS_CONFIG = ROOT / "experiments" / "configs" / "v0.8_family_generalization.yaml"
QUALITY_CONFIG = ROOT / "experiments" / "configs" / "v0.8_family_explanation_quality.yaml"


def test_v08_family_snapshot_manifest_and_ingestion_outputs_family_metadata():
    verification = verify_manifest(MANIFEST)

    assert verification["exists"].all()
    assert verification["checksum_ok"].all()
    assert verification["row_count_ok"].all()
    assert "protein_family_metadata" in set(verification["source_type"])

    result = run_ingestion(INGESTION_CONFIG)
    assert result.warnings == []
    assert result.normalized_files["protein_families"].exists()

    proteins = pd.read_csv(result.normalized_files["proteins"])
    variants = pd.read_csv(result.normalized_files["variants"])
    families = pd.read_csv(result.normalized_files["protein_families"])
    assert proteins.shape[0] == 6
    assert variants.shape[0] == 18
    assert set(proteins["protein_family"]) == {
        "tumor_suppressor",
        "receptor_tyrosine_kinase",
        "ras_gtpase",
    }
    assert families.shape[0] == 3

    validation = json.loads(result.validation_summary.read_text(encoding="utf-8"))[
        "validation"
    ]
    assert validation["protein_family_count"] == 3
    assert validation["variant_residue_mapping_rate"] == 1.0
    assert validation["variant_ca_coverage_rate"] == 1.0


def test_leave_one_family_split_strategy_holds_out_family():
    feature_table = pd.DataFrame(
        {
            "variant_id": ["v1", "v2", "v3", "v4"],
            "protein_id": ["P1", "P2", "P3", "P4"],
            "protein_family": ["family_a", "family_a", "family_b", "family_b"],
            "split": ["train", "train", "test", "test"],
        }
    )

    split = apply_split_strategy(
        feature_table,
        strategy="leave_one_family_out",
        holdout_family_id="family_a",
    )

    assert set(split[split["protein_family"] == "family_a"]["eval_split"]) == {"test"}
    assert set(split[split["protein_family"] == "family_b"]["eval_split"]) == {"train"}


def test_v08_family_generalization_outputs_protein_and_family_splits():
    run_ingestion(INGESTION_CONFIG)
    result = run_cross_protein_experiment(CROSS_CONFIG)

    assert result.run_count == 10
    for path in [
        result.metrics,
        result.predictions,
        result.per_protein_metrics,
        result.per_family_metrics,
        result.failure_modes,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"] == {
        "families": 3,
        "proteins": 6,
        "variants": 18,
    }

    metrics = pd.read_csv(result.metrics)
    assert set(metrics["split_strategy"]) == {
        "existing_split",
        "leave_one_protein_out",
        "leave_one_family_out",
    }
    assert set(metrics["holdout_family_id"].dropna()) >= {
        "tumor_suppressor",
        "receptor_tyrosine_kinase",
        "ras_gtpase",
    }

    per_family = pd.read_csv(result.per_family_metrics)
    assert set(per_family["protein_family"]) == {
        "tumor_suppressor",
        "receptor_tyrosine_kinase",
        "ras_gtpase",
    }
    report = result.report.read_text(encoding="utf-8")
    assert "## Leave-One-Family-Out Performance" in report
    assert "## Per-Family Metrics" in report


def test_v08_family_explanation_quality_summarizes_families_and_cards():
    run_ingestion(INGESTION_CONFIG)
    result = run_explanation_quality_evaluation(QUALITY_CONFIG)

    assert result.variant_count == 18
    assert result.card_count == 3
    assert result.quality_by_family.exists()

    quality_by_family = pd.read_csv(result.quality_by_family)
    assert set(quality_by_family["protein_family"]) == {
        "tumor_suppressor",
        "receptor_tyrosine_kinase",
        "ras_gtpase",
    }

    quality_by_context = pd.read_csv(result.quality_by_context)
    assert "leave_one_family_out" in set(quality_by_context["split_strategy"])
    assert not quality_by_context["mean_explanation_quality_score"].isna().any()

    cards = [
        json.loads(line)
        for line in result.explanation_cards_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert {card["variant_id"] for card in cards} == {
        "V_TP53_R273H",
        "V_EGFR_L858R",
        "V_KRAS_G12D",
    }

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["dataset_summary"]["families"] == 3
    assert metadata["dataset_summary"]["variants"] == 18

    report = result.report.read_text(encoding="utf-8")
    assert "## Quality By Protein Family" in report
    assert "leave_one_family_out" in report







































