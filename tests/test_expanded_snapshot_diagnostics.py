import json
from pathlib import Path

import pandas as pd

from trustprotkg.ingestion import run_ingestion
from trustprotkg.snapshot_diagnostics import run_snapshot_diagnostics


ROOT = Path(__file__).resolve().parents[1]
INGESTION_CONFIG = ROOT / "configs" / "expanded_snapshot_ingestion.yaml"
DIAGNOSTICS_CONFIG = ROOT / "experiments" / "configs" / "v1.2_snapshot_diagnostics.yaml"
NORMALIZED_DIR = ROOT / "data" / "normalized" / "v1.2_expanded"
RUN_DIR = ROOT / "experiments" / "runs" / "v1.2_snapshot_diagnostics"


def test_v12_expanded_snapshot_ingestion_normalizes_larger_fixture():
    result = run_ingestion(INGESTION_CONFIG)

    assert result.output_dir == NORMALIZED_DIR
    assert result.warnings == []
    assert result.benchmark_config.exists()
    assert result.validation_summary.exists()
    assert result.audit_report is not None
    assert result.audit_report.exists()

    metadata = json.loads(result.validation_summary.read_text(encoding="utf-8"))
    validation = metadata["validation"]
    assert validation["variant_count"] == 24
    assert validation["protein_count"] == 8
    assert validation["protein_family_count"] == 4
    assert validation["positive_labels"] == 16
    assert validation["negative_labels"] == 8
    assert validation["variant_residue_mapping_rate"] == 1.0
    assert validation["variant_ca_coverage_rate"] == 1.0
    assert len(metadata["manifest"]) == 17
    assert all(row["checksum_ok"] and row["row_count_ok"] for row in metadata["manifest"])

    proteins = pd.read_csv(NORMALIZED_DIR / "proteins.csv")
    assert {"AKT1_HUMAN", "BRAF_HUMAN"}.issubset(set(proteins["protein_id"]))
    assert "serine_threonine_kinase" in set(proteins["protein_family"])


def test_v12_snapshot_diagnostics_reports_label_leakage_and_provenance_checks():
    run_ingestion(INGESTION_CONFIG)
    result = run_snapshot_diagnostics(DIAGNOSTICS_CONFIG)

    assert result.variant_count == 24
    assert result.protein_count == 8
    assert result.family_count == 4
    for path in [
        result.label_diagnostics,
        result.leakage_diagnostics,
        result.provenance_diagnostics,
        result.curation_summary,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    metadata = json.loads((RUN_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_summary"]["variants"] == 24
    assert metadata["dataset_summary"]["proteins"] == 8
    assert metadata["dataset_summary"]["families"] == 4
    assert metadata["dataset_summary"]["provenance_failures"] == 0
    assert metadata["dataset_summary"]["leakage_warnings"] == 2

    labels = pd.read_csv(RUN_DIR / "label_diagnostics.csv")
    overall = labels[
        (labels["context_type"] == "overall") & (labels["context_id"] == "all")
    ].iloc[0]
    assert int(overall["positive_labels"]) == 16
    assert int(overall["negative_labels"]) == 8
    assert bool(overall["has_both_classes"])

    leakage = pd.read_csv(RUN_DIR / "split_leakage_diagnostics.csv")
    assert {"random_split", "protein", "family"}.issubset(
        set(leakage["context_type"])
    )
    assert (leakage["check_name"] == "leave_one_family_out_holdout").any()

    provenance = pd.read_csv(RUN_DIR / "provenance_diagnostics.csv")
    assert (provenance["status"] == "fail").sum() == 0
    assert {
        "variants",
        "go_annotations",
        "domains",
        "pathways",
        "disease_associations",
    }.issubset(set(provenance["table_name"]))

    report = (RUN_DIR / "snapshot_diagnostics_report.md").read_text(encoding="utf-8")
    for section in [
        "## Label Balance Diagnostics",
        "## Leakage-Aware Split Diagnostics",
        "## Provenance Completeness Diagnostics",
        "## Curation And Dropped-Row Summary",
    ]:
        assert section in report
















