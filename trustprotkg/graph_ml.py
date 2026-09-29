"""Lightweight deterministic graph-embedding baseline for TrustProtKG."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from trustprotkg.config import PipelineConfig, load_simple_yaml
from trustprotkg.cross_protein import apply_split_strategy
from trustprotkg.evaluation import run_baseline_evaluation
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.models import variant_node_id
from trustprotkg.pipeline import _build_graph


TRANSPARENT_FEATURES = [
    "residue_confidence",
    "num_structural_contacts",
    "avg_contact_distance",
    "is_in_domain",
    "num_go_pathway_disease_edges",
]


def _resolve_path(path_value: str | Path, base_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or not (base_path.parent / path).exists():
        return cwd_path
    return base_path.parent / path


def _relative(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path_obj)


def _project_version() -> str:
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    with pyproject.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return None if math.isnan(numeric) else numeric
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass(frozen=True)
class GraphMLConfig:
    path: Path
    name: str
    benchmark_config: Path
    random_seed: int
    timestamp: str
    embedding_dimensions: int
    strategy: str
    include_random_split: bool
    protein_column: str
    family_column: str
    split_column: str
    train_value: str
    test_value: str
    label_column: str
    run_dir: Path
    feature_table: Path
    metrics: Path
    predictions: Path
    comparison: Path
    embedding_summary: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "GraphMLConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        embedding = raw.get("embedding", {})
        split = raw.get("split", {})
        label = raw.get("label", {})
        outputs = raw.get("outputs", {})
        run_dir = _resolve_path(outputs["run_dir"], config_path)

        def out_path(key: str, default_name: str) -> Path:
            value = Path(str(outputs.get(key, default_name)))
            return value if value.is_absolute() else run_dir / value

        return cls(
            path=config_path,
            name=str(experiment["name"]),
            benchmark_config=_resolve_path(experiment["benchmark_config"], config_path),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            embedding_dimensions=int(embedding.get("dimensions", 8)),
            strategy=str(split.get("strategy", "all_family_aware")),
            include_random_split=bool(split.get("include_random_split", True)),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            label_column=str(label.get("column", "label_binary")),
            run_dir=run_dir,
            feature_table=out_path("feature_table", "graph_ml_variant_features.csv"),
            metrics=out_path("metrics", "graph_ml_metrics.csv"),
            predictions=out_path("predictions", "graph_ml_predictions.csv"),
            comparison=out_path("comparison", "graph_ml_vs_transparent.csv"),
            embedding_summary=out_path("embedding_summary", "graph_embedding_summary.csv"),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "graph_ml_report.md"),
        )


@dataclass(frozen=True)
class GraphMLResult:
    run_dir: Path
    feature_table: Path
    metrics: Path
    predictions: Path
    comparison: Path
    embedding_summary: Path
    metadata: Path
    report: Path
    variant_count: int
    split_run_count: int
    feature_set_count: int


def _simple_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for source, target, attrs in graph.edges(data=True):
        weight = float(attrs.get("confidence", 1.0) or 1.0)
        if simple.has_edge(source, target):
            simple[source][target]["weight"] += weight
        else:
            simple.add_edge(source, target, weight=weight)
    return simple


def _spectral_embeddings(
    graph: nx.MultiDiGraph,
    *,
    dimensions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple = _simple_graph(graph)
    nodes = sorted(simple.nodes)
    n_nodes = len(nodes)
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for source, target, attrs in simple.edges(data=True):
        i = node_index[source]
        j = node_index[target]
        weight = float(attrs.get("weight", 1.0))
        adjacency[i, j] += weight
        adjacency[j, i] += weight
    adjacency += np.eye(n_nodes, dtype=float)
    degree = adjacency.sum(axis=1)
    degree[degree == 0.0] = 1.0
    normalized = adjacency / np.sqrt(np.outer(degree, degree))
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    active = order[: max(1, min(dimensions, n_nodes))]
    values = eigenvalues[active]
    vectors = eigenvectors[:, active] * values
    columns = [f"graph_ml_emb_{index + 1:02d}" for index in range(vectors.shape[1])]
    embeddings = pd.DataFrame(vectors, columns=columns)
    embeddings.insert(0, "node_id", nodes)
    summary = pd.DataFrame(
        {
            "dimension": columns,
            "eigenvalue": values,
            "mean": vectors.mean(axis=0),
            "std": vectors.std(axis=0),
            "min": vectors.min(axis=0),
            "max": vectors.max(axis=0),
        }
    )
    return embeddings, summary


def _variant_embedding_features(
    feature_table: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    table = feature_table.copy()
    table["node_id"] = table["variant_id"].map(lambda value: variant_node_id(str(value)))
    merged = table.merge(embeddings, on="node_id", how="left").drop(columns=["node_id"])
    embedding_columns = [column for column in embeddings.columns if column != "node_id"]
    merged[embedding_columns] = merged[embedding_columns].fillna(0.0)
    return merged


def _attach_protein_family(
    feature_table: pd.DataFrame,
    benchmark_config: PipelineConfig,
    family_column: str,
) -> pd.DataFrame:
    if family_column in feature_table.columns:
        return feature_table
    proteins = pd.read_csv(benchmark_config.proteins)
    if family_column not in proteins.columns:
        return feature_table
    return feature_table.merge(
        proteins[["protein_id", family_column]],
        on="protein_id",
        how="left",
    )


def _split_runs(feature_table: pd.DataFrame, config: GraphMLConfig) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if config.include_random_split:
        runs.append(
            {
                "run_name": "random_split",
                "strategy": "existing_split",
                "holdout_protein_id": None,
                "holdout_family_id": None,
            }
        )
    for protein_id in sorted(feature_table[config.protein_column].dropna().unique()):
        runs.append(
            {
                "run_name": f"leave_{protein_id}_out",
                "strategy": "leave_one_protein_out",
                "holdout_protein_id": str(protein_id),
                "holdout_family_id": None,
            }
        )
    for family_id in sorted(feature_table[config.family_column].dropna().unique()):
        runs.append(
            {
                "run_name": f"leave_family_{family_id}_out",
                "strategy": "leave_one_family_out",
                "holdout_protein_id": None,
                "holdout_family_id": str(family_id),
            }
        )
    return runs


def _evaluate_runs(
    feature_table: pd.DataFrame,
    config: GraphMLConfig,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for run in _split_runs(feature_table, config):
        split_table = apply_split_strategy(
            feature_table,
            strategy=run["strategy"],
            protein_column=config.protein_column,
            split_column=config.split_column,
            train_value=config.train_value,
            test_value=config.test_value,
            holdout_protein_id=run["holdout_protein_id"],
            holdout_family_id=run["holdout_family_id"],
            family_column=config.family_column,
        )
        evaluation = run_baseline_evaluation(
            split_table,
            label_column=config.label_column,
            split_column="eval_split",
            train_value=config.train_value,
            test_value=config.test_value,
            feature_sets=feature_sets,
        )
        evaluation.metrics.insert(0, "run_name", run["run_name"])
        evaluation.metrics.insert(1, "split_strategy", run["strategy"])
        evaluation.metrics.insert(2, "holdout_protein_id", run["holdout_protein_id"] or "")
        evaluation.metrics.insert(3, "holdout_family_id", run["holdout_family_id"] or "")
        evaluation.predictions.insert(0, "run_name", run["run_name"])
        evaluation.predictions.insert(1, "split_strategy", run["strategy"])
        evaluation.predictions.insert(2, "holdout_protein_id", run["holdout_protein_id"] or "")
        evaluation.predictions.insert(3, "holdout_family_id", run["holdout_family_id"] or "")
        metric_frames.append(evaluation.metrics)
        prediction_frames.append(evaluation.predictions)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _comparison(metrics: pd.DataFrame, reference: str = "transparent_structure_plus_kg") -> pd.DataFrame:
    reference_rows = metrics[metrics["feature_set"] == reference][
        ["run_name", "accuracy", "f1", "auroc"]
    ].rename(
        columns={
            "accuracy": "reference_accuracy",
            "f1": "reference_f1",
            "auroc": "reference_auroc",
        }
    )
    comparison = metrics.merge(reference_rows, on="run_name", how="left")
    comparison["delta_accuracy_vs_transparent"] = (
        comparison["accuracy"] - comparison["reference_accuracy"]
    )
    comparison["delta_f1_vs_transparent"] = comparison["f1"] - comparison["reference_f1"]
    comparison["delta_auroc_vs_transparent"] = (
        comparison["auroc"] - comparison["reference_auroc"]
    )
    return comparison[
        [
            "run_name",
            "split_strategy",
            "feature_set",
            "n_train",
            "n_test",
            "accuracy",
            "f1",
            "auroc",
            "delta_accuracy_vs_transparent",
            "delta_f1_vs_transparent",
            "delta_auroc_vs_transparent",
        ]
    ].sort_values(["run_name", "feature_set"])


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]]
    if frame.empty:
        return "_No rows._"
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].round(3)
    lines = [
        "| " + " | ".join(rounded.columns) + " |",
        "| " + " | ".join(["---"] * len(rounded.columns)) + " |",
    ]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def _write_report(
    config: GraphMLConfig,
    *,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    embedding_summary: pd.DataFrame,
) -> Path:
    grouped = (
        comparison.groupby("feature_set", as_index=False)
        .agg(
            n_runs=("run_name", "nunique"),
            mean_f1=("f1", "mean"),
            mean_delta_f1_vs_transparent=("delta_f1_vs_transparent", "mean"),
            mean_auroc=("auroc", "mean"),
        )
        .sort_values("feature_set")
    )
    lines = [
        f"# {config.name}",
        "",
        "## Summary",
        "",
        f"- Metric rows: `{metrics.shape[0]}`",
        f"- Embedding dimensions: `{config.embedding_dimensions}`",
        "",
        "## Mean Performance By Feature Set",
        "",
        _markdown_table(grouped),
        "",
        "## Embedding Spectrum",
        "",
        _markdown_table(embedding_summary, ["dimension", "eigenvalue", "std", "min", "max"]),
        "",
        "## Interpretation Guardrail",
        "",
        "This is a compact deterministic graph-embedding baseline for method exploration only. It should be interpreted beside transparent features, evidence diagnostics, citation curation, and the HTML evidence-card report.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")
    return config.report


def run_graph_ml_experiment(config_path: str | Path) -> GraphMLResult:
    config = GraphMLConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    feature_table = _attach_protein_family(
        feature_table,
        benchmark_config,
        config.family_column,
    )
    embeddings, embedding_summary = _spectral_embeddings(
        graph,
        dimensions=config.embedding_dimensions,
    )
    feature_table = _variant_embedding_features(feature_table, embeddings)
    embedding_columns = [column for column in feature_table.columns if column.startswith("graph_ml_emb_")]
    feature_sets = {
        "transparent_structure_plus_kg": TRANSPARENT_FEATURES,
        "spectral_graph_embedding": embedding_columns,
        "transparent_plus_spectral": TRANSPARENT_FEATURES + embedding_columns,
    }
    metrics, predictions = _evaluate_runs(feature_table, config, feature_sets)
    comparison = _comparison(metrics)

    config.feature_table.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(config.feature_table, index=False)
    metrics.to_csv(config.metrics, index=False)
    predictions.to_csv(config.predictions, index=False)
    comparison.to_csv(config.comparison, index=False)
    embedding_summary.to_csv(config.embedding_summary, index=False)
    _write_report(
        config,
        metrics=metrics,
        comparison=comparison,
        embedding_summary=embedding_summary,
    )
    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "benchmark_config": _relative(config.benchmark_config),
        "trustprotkg_version": _project_version(),
        "dataset_summary": {
            "variants": int(feature_table["variant_id"].nunique()),
            "graph_nodes": int(graph.number_of_nodes()),
            "graph_edges": int(graph.number_of_edges()),
            "embedding_dimensions": len(embedding_columns),
            "split_runs": len(_split_runs(feature_table, config)),
            "feature_sets": len(feature_sets),
            "metric_rows": int(metrics.shape[0]),
            "prediction_rows": int(predictions.shape[0]),
        },
        "feature_sets": feature_sets,
        "outputs": {
            "feature_table": _relative(config.feature_table),
            "metrics": _relative(config.metrics),
            "predictions": _relative(config.predictions),
            "comparison": _relative(config.comparison),
            "embedding_summary": _relative(config.embedding_summary),
            "report": _relative(config.report),
        },
    }
    config.metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return GraphMLResult(
        run_dir=config.run_dir,
        feature_table=config.feature_table,
        metrics=config.metrics,
        predictions=config.predictions,
        comparison=config.comparison,
        embedding_summary=config.embedding_summary,
        metadata=config.metadata,
        report=config.report,
        variant_count=int(feature_table["variant_id"].nunique()),
        split_run_count=len(_split_runs(feature_table, config)),
        feature_set_count=len(feature_sets),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TrustProtKG lightweight graph ML baseline."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v2.0_graph_ml.yaml",
        help="Path to a TrustProtKG graph ML config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_graph_ml_experiment(args.config)
    print("TrustProtKG graph ML baseline complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Variants: {result.variant_count}")
    print(f"Split runs: {result.split_run_count}")
    print(f"Feature sets: {result.feature_set_count}")
    print(f"Metrics: {result.metrics}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
