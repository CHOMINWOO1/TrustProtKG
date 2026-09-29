import json
from pathlib import Path

import pandas as pd

from trustprotkg.config import PipelineConfig
from trustprotkg.graph_baselines import (
    GRAPH_FEATURES,
    extract_graph_feature_table,
    run_graph_baseline_experiment,
)
from trustprotkg.pipeline import _build_graph


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = ROOT / "configs" / "family_snapshot_benchmark.yaml"
GRAPH_CONFIG = ROOT / "experiments" / "configs" / "v1.1_graph_baselines.yaml"
RUN_DIR = ROOT / "experiments" / "runs" / "v1.1_graph_baselines"


def test_graph_feature_table_contains_pagerank_metapath_and_topology_features():
    benchmark_config = PipelineConfig.from_file(BENCHMARK_CONFIG)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)

    table = extract_graph_feature_table(graph, variants)

    assert table.shape[0] == 18
    assert set(GRAPH_FEATURES).issubset(table.columns)
    assert table["graph_ppr_residue"].gt(0).all()
    assert table["graph_ppr_biomedical_mass"].gt(0).all()
    assert table["graph_residue_to_domain_score"].between(0, 1).all()
    assert table["graph_residue_degree"].gt(0).all()
    assert table["graph_component_size"].gt(1).all()


def test_v11_graph_baseline_experiment_writes_graph_aware_outputs():
    result = run_graph_baseline_experiment(GRAPH_CONFIG)

    assert result.perturbation_count == 8
    assert result.split_run_count == 10
    assert result.feature_set_count == 6
    assert result.variant_count == 18

    expected_paths = [
        RUN_DIR / "graph_variant_features.csv",
        RUN_DIR / "graph_baseline_metrics.csv",
        RUN_DIR / "graph_baseline_predictions.csv",
        RUN_DIR / "graph_vs_transparent.csv",
        RUN_DIR / "graph_perturbation_deltas.csv",
        RUN_DIR / "graph_robustness_summary.csv",
        RUN_DIR / "graph_feature_summary.csv",
        RUN_DIR / "metadata.json",
        RUN_DIR / "graph_baseline_report.md",
    ]
    for path in expected_paths:
        assert path.exists()

    metadata = json.loads((RUN_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["trustprotkg_version"] == "41.0.0"
    assert metadata["dataset_summary"]["metric_rows"] == 480
    assert metadata["dataset_summary"]["prediction_rows"] == 2064
    assert metadata["dataset_summary"]["split_runs"] == 10
    assert metadata["dataset_summary"]["perturbations"] == 8

    metrics = pd.read_csv(RUN_DIR / "graph_baseline_metrics.csv")
    assert metrics.shape[0] == 480
    assert set(metrics["split_strategy"]) == {
        "existing_split",
        "leave_one_protein_out",
        "leave_one_family_out",
    }
    assert {
        "transparent_structure_plus_kg",
        "graph_pagerank",
        "graph_metapath",
        "graph_topology",
        "graph_all",
        "transparent_plus_graph",
    }.issubset(set(metrics["feature_set"]))

    comparison = pd.read_csv(RUN_DIR / "graph_vs_transparent.csv")
    assert "delta_vs_transparent_f1" in comparison.columns
    assert "transparent_plus_graph" in set(comparison["feature_set"])

    robustness = pd.read_csv(RUN_DIR / "graph_robustness_summary.csv")
    assert not robustness.empty
    assert "mean_delta_from_unperturbed_f1" in robustness.columns

    feature_summary = pd.read_csv(RUN_DIR / "graph_feature_summary.csv")
    assert set(GRAPH_FEATURES) == set(feature_summary["feature"])

    report = (RUN_DIR / "graph_baseline_report.md").read_text(encoding="utf-8")
    for section in [
        "## Graph-Derived Feature Families",
        "## Graph-Aware Versus Transparent Baseline",
        "## Perturbation Robustness",
        "## Graph Feature Summary",
    ]:
        assert section in report







































