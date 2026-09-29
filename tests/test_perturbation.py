import json
from pathlib import Path

import pandas as pd

from trustprotkg.config import PipelineConfig
from trustprotkg.ingestion import run_ingestion
from trustprotkg.models import EdgeType
from trustprotkg.perturbation import (
    PerturbationConfig,
    apply_graph_perturbation,
    build_perturbation_specs,
    run_perturbation_experiment,
)
from trustprotkg.pipeline import _build_graph


ROOT = Path(__file__).resolve().parents[1]
INGESTION_CONFIG = ROOT / "configs" / "family_snapshot_ingestion.yaml"
BENCHMARK_CONFIG = ROOT / "configs" / "family_snapshot_benchmark.yaml"
PERTURBATION_CONFIG = ROOT / "experiments" / "configs" / "v0.9_perturbation.yaml"


def _family_graph():
    run_ingestion(INGESTION_CONFIG)
    return _build_graph(PipelineConfig.from_file(BENCHMARK_CONFIG))


def test_apply_graph_perturbation_removes_structural_contacts_without_mutating_base():
    graph = _family_graph()
    config = PerturbationConfig.from_file(PERTURBATION_CONFIG)

    contact_edges_before = sum(
        1
        for _source, _target, _key, attrs in graph.edges(keys=True, data=True)
        if attrs.get("edge_type") == EdgeType.STRUCTURAL_CONTACT.value
    )
    perturbation_spec = [
        item
        for item in build_perturbation_specs(config.perturbation_names, 0.25)
        if item.name == "remove_structural_contacts"
    ][0]

    perturbed, summary = apply_graph_perturbation(graph, perturbation_spec)
    contact_edges_after = sum(
        1
        for _source, _target, _key, attrs in perturbed.edges(keys=True, data=True)
        if attrs.get("edge_type") == EdgeType.STRUCTURAL_CONTACT.value
    )
    contact_edges_base_after = sum(
        1
        for _source, _target, _key, attrs in graph.edges(keys=True, data=True)
        if attrs.get("edge_type") == EdgeType.STRUCTURAL_CONTACT.value
    )

    assert contact_edges_before > 0
    assert contact_edges_after == 0
    assert contact_edges_base_after == contact_edges_before
    assert summary["edges_removed"] == contact_edges_before


def test_v09_perturbation_run_writes_deltas_robustness_and_cards():
    run_ingestion(INGESTION_CONFIG)
    result = run_perturbation_experiment(PERTURBATION_CONFIG)

    assert result.perturbation_count == 8
    assert result.split_run_count == 10
    assert result.variant_count == 18
    assert result.card_count == 3
    for path in [
        result.variant_quality_scores,
        result.metrics,
        result.predictions,
        result.prediction_deltas,
        result.quality_deltas,
        result.protein_robustness,
        result.family_robustness,
        result.before_after_cards_jsonl,
        result.before_after_cards_markdown,
        result.metadata,
        result.report,
    ]:
        assert path.exists()

    metrics = pd.read_csv(result.metrics)
    assert set(metrics["perturbation"]) == {
        "baseline",
        "remove_structural_contacts",
        "remove_domain_edges",
        "remove_go_edges",
        "remove_pathway_edges",
        "remove_disease_edges",
        "remove_source_db_interpro",
        "weaken_source_db_go",
    }
    assert set(metrics["split_strategy"]) == {
        "existing_split",
        "leave_one_protein_out",
        "leave_one_family_out",
    }

    prediction_deltas = pd.read_csv(result.prediction_deltas)
    assert {"delta_accuracy", "delta_f1", "delta_auroc"}.issubset(
        prediction_deltas.columns
    )

    quality_deltas = pd.read_csv(result.quality_deltas)
    assert {
        "delta_explanation_quality_score",
        "delta_confidence_weighted_explanation_quality_score",
        "delta_structural_contact_plausibility_score",
        "delta_domain_support_score",
    }.issubset(quality_deltas.columns)
    structural_rows = quality_deltas[
        quality_deltas["perturbation"] == "remove_structural_contacts"
    ]
    assert (structural_rows["contact_count"] == 0).all()
    assert (structural_rows["delta_structural_contact_plausibility_score"] < 0).any()

    go_weaken_rows = quality_deltas[
        quality_deltas["perturbation"] == "weaken_source_db_go"
    ]
    assert (go_weaken_rows["delta_evidence_confidence_mean"] < 0).any()
    assert (
        go_weaken_rows["delta_confidence_weighted_explanation_quality_score"] < 0
    ).any()

    family_robustness = pd.read_csv(result.family_robustness)
    assert set(family_robustness["protein_family"]) == {
        "tumor_suppressor",
        "receptor_tyrosine_kinase",
        "ras_gtpase",
    }

    cards = [
        json.loads(line)
        for line in result.before_after_cards_jsonl.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {card["variant_id"] for card in cards} == {
        "V_TP53_R273H",
        "V_EGFR_L858R",
        "V_KRAS_G12D",
    }
    assert all(card["comparisons"] for card in cards)

    metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"]["perturbations"] == 8
    assert metadata["dataset_summary"]["families"] == 3

    report = result.report.read_text(encoding="utf-8")
    for section in [
        "## Perturbation Plan",
        "## Prediction Metric Deltas",
        "## Explanation-Quality Deltas",
        "## Protein Robustness Summary",
        "## Family Robustness Summary",
        "## Before/After Case Cards",
        "## Limitations",
    ]:
        assert section in report







































