import json
from pathlib import Path

import pandas as pd

from trustprotkg.experiment import run_experiment
from trustprotkg.ingestion import load_manifest, run_ingestion, verify_manifest


ROOT = Path(__file__).resolve().parents[1]
INGESTION_CONFIG = ROOT / "configs" / "snapshot_ingestion.yaml"
MANIFEST = ROOT / "data" / "snapshots" / "v0.4" / "manifest.csv"
EXPERIMENT_CONFIG = ROOT / "experiments" / "configs" / "v0.4_snapshot.yaml"


def test_manifest_verification_checks_checksum_and_row_count(tmp_path):
    verification = verify_manifest(MANIFEST)

    assert verification["exists"].all()
    assert verification["checksum_ok"].all()
    assert verification["row_count_ok"].all()
    assert {
        "protein_metadata",
        "structure_index",
        "variant_assertions",
        "go_annotations",
        "domains",
        "pathways",
        "disease_associations",
    }.issubset(set(verification["source_type"]))

    bad_manifest = load_manifest(MANIFEST)
    bad_manifest.loc[0, "checksum"] = "0" * 64
    bad_path = tmp_path / "bad_manifest.csv"
    bad_manifest.to_csv(bad_path, index=False)
    bad_verification = verify_manifest(bad_path)

    assert not bool(bad_verification.loc[0, "checksum_ok"])
    assert bool(bad_verification.loc[0, "row_count_ok"])


def test_snapshot_ingestion_normalizes_sources_and_writes_provenance_report():
    result = run_ingestion(INGESTION_CONFIG)

    assert result.benchmark_config.exists()
    assert result.provenance_report.exists()
    assert result.validation_summary.exists()
    assert result.warnings == []
    for path in result.normalized_files.values():
        assert path.exists()

    variants = pd.read_csv(result.normalized_files["variants"])
    assert variants.shape[0] == 9
    assert set(variants["protein_id"]) == {"TP53_HUMAN", "BRCA1_HUMAN", "PTEN_HUMAN"}
    assert {"source_db", "evidence_type", "confidence"}.issubset(variants.columns)

    summary = json.loads(result.validation_summary.read_text(encoding="utf-8"))
    validation = summary["validation"]
    assert validation["variant_residue_mapping_rate"] == 1.0
    assert validation["variant_ca_coverage_rate"] == 1.0
    assert validation["positive_labels"] == 4
    assert validation["negative_labels"] == 5

    report = result.provenance_report.read_text(encoding="utf-8")
    assert "## Source Snapshot Manifest" in report
    assert "## Transformations" in report
    assert "## Validation Summary" in report
    assert "No validation warnings." in report


def test_snapshot_experiment_runs_on_normalized_benchmark():
    run_ingestion(INGESTION_CONFIG)
    result = run_experiment(EXPERIMENT_CONFIG)

    assert result.node_count == 54
    assert result.edge_count == 166
    metrics = pd.read_csv(result.metrics)
    assert set(metrics["feature_set"]) == {
        "structure_only",
        "kg_only",
        "structure_plus_kg",
        "no_domain",
        "no_structural_contacts",
    }

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["benchmark_config"] == "configs\\snapshot_benchmark.yaml"
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"]["variants"] == 9
    assert metadata["input_hashes"]["variants"]

    report = result.report.read_text(encoding="utf-8")
    assert "TrustProtKG v0.4 snapshot-ingestion experiment" in report
    assert "## Ablation Table" in report







































