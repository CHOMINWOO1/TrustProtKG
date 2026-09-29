"""Graph-aware baseline models for TrustProtKG v1.1."""

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
from trustprotkg.features import (
    _incident_edges,
    _located_residue_for_variant,
    _other_node,
    extract_variant_feature_table,
)
from trustprotkg.models import EdgeType, NodeType, protein_node_id, variant_node_id
from trustprotkg.perturbation import apply_graph_perturbation, build_perturbation_specs
from trustprotkg.pipeline import _build_graph


TRANSPARENT_FEATURES = [
    "residue_confidence",
    "num_structural_contacts",
    "avg_contact_distance",
    "is_in_domain",
    "num_go_pathway_disease_edges",
]

GRAPH_PAGERANK_FEATURES = [
    "graph_ppr_residue",
    "graph_ppr_protein",
    "graph_ppr_domain_mass",
    "graph_ppr_go_mass",
    "graph_ppr_pathway_mass",
    "graph_ppr_disease_mass",
    "graph_ppr_biomedical_mass",
]

GRAPH_METAPATH_FEATURES = [
    "graph_residue_domain_edge_count",
    "graph_residue_contact_domain_count",
    "graph_protein_go_edge_count",
    "graph_protein_pathway_edge_count",
    "graph_protein_disease_edge_count",
    "graph_residue_to_domain_score",
    "graph_residue_to_go_score",
    "graph_residue_to_pathway_score",
    "graph_residue_to_disease_score",
]

GRAPH_TOPOLOGY_FEATURES = [
    "graph_residue_degree",
    "graph_residue_clustering",
    "graph_residue_betweenness",
    "graph_residue_pagerank",
    "graph_protein_degree",
    "graph_protein_pagerank",
    "graph_residue_neighbor_degree_mean",
    "graph_component_size",
]

GRAPH_FEATURES = (
    GRAPH_PAGERANK_FEATURES + GRAPH_METAPATH_FEATURES + GRAPH_TOPOLOGY_FEATURES
)

DEFAULT_FEATURE_SETS = {
    "transparent_structure_plus_kg": TRANSPARENT_FEATURES,
    "graph_pagerank": GRAPH_PAGERANK_FEATURES,
    "graph_metapath": GRAPH_METAPATH_FEATURES,
    "graph_topology": GRAPH_TOPOLOGY_FEATURES,
    "graph_all": GRAPH_FEATURES,
    "transparent_plus_graph": TRANSPARENT_FEATURES + GRAPH_FEATURES,
}

FEATURE_SET_FAMILIES = {
    "transparent_structure_plus_kg": "transparent",
    "graph_pagerank": "graph_only",
    "graph_metapath": "graph_only",
    "graph_topology": "graph_only",
    "graph_all": "graph_only",
    "transparent_plus_graph": "transparent_plus_graph",
}


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


def _split_csv(value: Any) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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
class GraphBaselineConfig:
    path: Path
    name: str
    benchmark_config: Path
    random_seed: int
    timestamp: str
    strategy: str
    include_random_split: bool
    protein_column: str
    family_column: str
    split_column: str
    train_value: str
    test_value: str
    label_column: str
    feature_sets: dict[str, list[str]]
    perturbation_names: list[str]
    weaken_confidence_scale: float
    transparent_reference_feature_set: str
    run_dir: Path
    feature_table: Path
    metrics: Path
    predictions: Path
    graph_vs_transparent: Path
    perturbation_deltas: Path
    robustness_summary: Path
    graph_feature_summary: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "GraphBaselineConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        split = raw.get("split", {})
        label = raw.get("label", {})
        feature_sets_raw = raw.get("feature_sets", {})
        perturbations = raw.get("perturbations", {})
        comparison = raw.get("comparison", {})
        outputs = raw.get("outputs", {})
        run_dir = _resolve_path(outputs["run_dir"], config_path)

        def out_path(key: str, default_name: str) -> Path:
            value = outputs.get(key, default_name)
            path_value = Path(str(value))
            if path_value.is_absolute():
                return path_value
            return run_dir / path_value

        feature_sets = (
            {
                str(name): _split_csv(columns)
                for name, columns in feature_sets_raw.items()
            }
            if feature_sets_raw
            else dict(DEFAULT_FEATURE_SETS)
        )

        return cls(
            path=config_path,
            name=str(experiment["name"]),
            benchmark_config=_resolve_path(experiment["benchmark_config"], config_path),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            strategy=str(split.get("strategy", "all_family_aware")),
            include_random_split=bool(split.get("include_random_split", True)),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            label_column=str(label.get("column", "label_binary")),
            feature_sets=feature_sets,
            perturbation_names=_split_csv(
                perturbations.get(
                    "names",
                    "baseline,remove_structural_contacts,remove_domain_edges,"
                    "remove_go_edges,remove_pathway_edges,remove_disease_edges,"
                    "remove_source_db_interpro,weaken_source_db_go",
                )
            ),
            weaken_confidence_scale=float(
                perturbations.get("weaken_confidence_scale", 0.25)
            ),
            transparent_reference_feature_set=str(
                comparison.get(
                    "transparent_reference_feature_set",
                    "transparent_structure_plus_kg",
                )
            ),
            run_dir=run_dir,
            feature_table=out_path("feature_table", "graph_variant_features.csv"),
            metrics=out_path("metrics", "graph_baseline_metrics.csv"),
            predictions=out_path("predictions", "graph_baseline_predictions.csv"),
            graph_vs_transparent=out_path(
                "graph_vs_transparent", "graph_vs_transparent.csv"
            ),
            perturbation_deltas=out_path(
                "perturbation_deltas", "graph_perturbation_deltas.csv"
            ),
            robustness_summary=out_path(
                "robustness_summary", "graph_robustness_summary.csv"
            ),
            graph_feature_summary=out_path(
                "graph_feature_summary", "graph_feature_summary.csv"
            ),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "graph_baseline_report.md"),
        )


@dataclass(frozen=True)
class GraphBaselineResult:
    run_dir: Path
    feature_table: Path
    metrics: Path
    predictions: Path
    graph_vs_transparent: Path
    perturbation_deltas: Path
    robustness_summary: Path
    graph_feature_summary: Path
    metadata: Path
    report: Path
    perturbation_count: int
    split_run_count: int
    feature_set_count: int
    variant_count: int


def _undirected_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for source, target, attrs in graph.edges(data=True):
        if simple.has_edge(source, target):
            simple[source][target]["multiplicity"] += 1
        else:
            simple.add_edge(source, target, multiplicity=1, edge_type=attrs.get("edge_type"))
    return simple


def _node_ids_by_type(graph: nx.Graph | nx.MultiDiGraph, node_type: str) -> list[str]:
    return [
        node_id
        for node_id, attrs in graph.nodes(data=True)
        if attrs.get("node_type") == node_type
    ]


def _node_type(graph: nx.Graph | nx.MultiDiGraph, node_id: str) -> str:
    return str(graph.nodes[node_id].get("node_type", ""))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pagerank(
    graph: nx.Graph,
    *,
    alpha: float = 0.85,
    personalization: dict[str, float] | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-8,
) -> dict[str, float]:
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    node_index = {node_id: index for index, node_id in enumerate(nodes)}
    n_nodes = len(nodes)

    if personalization is None:
        teleport = np.full(n_nodes, 1.0 / n_nodes, dtype=float)
    else:
        teleport = np.array(
            [float(personalization.get(node_id, 0.0)) for node_id in nodes],
            dtype=float,
        )
        total = float(teleport.sum())
        teleport = (
            teleport / total
            if total > 0.0
            else np.full(n_nodes, 1.0 / n_nodes, dtype=float)
        )

    scores = np.full(n_nodes, 1.0 / n_nodes, dtype=float)
    degrees = np.array([float(graph.degree(node_id)) for node_id in nodes])
    dangling_indices = np.where(degrees == 0.0)[0]

    for _ in range(max_iter):
        previous = scores.copy()
        scores = (1.0 - alpha) * teleport
        if dangling_indices.size:
            scores += alpha * float(previous[dangling_indices].sum()) * teleport
        for node_id in nodes:
            index = node_index[node_id]
            degree = degrees[index]
            if degree == 0.0:
                continue
            share = alpha * previous[index] / degree
            for neighbor in graph.neighbors(node_id):
                scores[node_index[neighbor]] += share
        if np.abs(scores - previous).sum() < n_nodes * tol:
            break

    return {node_id: float(scores[index]) for node_id, index in node_index.items()}


def _reciprocal_min_distance(
    graph: nx.Graph,
    source: str,
    targets: list[str],
) -> float:
    if source not in graph or not targets:
        return 0.0
    lengths = nx.single_source_shortest_path_length(graph, source, cutoff=4)
    distances = [lengths[target] for target in targets if target in lengths]
    if not distances:
        return 0.0
    return 1.0 / (1.0 + float(min(distances)))


def _incident_count_by_type(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    edge_type: EdgeType,
    neighbor_type: NodeType,
) -> int:
    count = 0
    for source, target, _key, attrs in _incident_edges(graph, node_id):
        if attrs.get("edge_type") != edge_type.value:
            continue
        neighbor = _other_node(source, target, node_id)
        if graph.nodes[neighbor].get("node_type") == neighbor_type.value:
            count += 1
    return count


def _protein_context_count(
    graph: nx.MultiDiGraph,
    protein_id: str,
    *,
    edge_type: EdgeType,
    neighbor_type: NodeType,
) -> int:
    return _incident_count_by_type(
        graph,
        protein_node_id(protein_id),
        edge_type=edge_type,
        neighbor_type=neighbor_type,
    )


def _residue_in_domain(graph: nx.MultiDiGraph, residue_id: str) -> bool:
    for source, target, _key, attrs in _incident_edges(graph, residue_id):
        if attrs.get("edge_type") != EdgeType.LOCATED_AT.value:
            continue
        neighbor = _other_node(source, target, residue_id)
        if graph.nodes[neighbor].get("node_type") == NodeType.DOMAIN.value:
            return True
    return False


def _contact_domain_count(graph: nx.MultiDiGraph, residue_id: str) -> int:
    contact_residues: set[str] = set()
    for source, target, _key, attrs in _incident_edges(graph, residue_id):
        if attrs.get("edge_type") == EdgeType.STRUCTURAL_CONTACT.value:
            neighbor = _other_node(source, target, residue_id)
            if graph.nodes[neighbor].get("node_type") == NodeType.RESIDUE.value:
                contact_residues.add(neighbor)
    return sum(1 for node_id in contact_residues if _residue_in_domain(graph, node_id))


def _component_size(graph: nx.Graph, node_id: str) -> int:
    if node_id not in graph:
        return 0
    return int(len(nx.node_connected_component(graph, node_id)))


def _graph_context(graph: nx.MultiDiGraph) -> dict[str, Any]:
    simple = _undirected_graph(graph)
    centrality = nx.betweenness_centrality(simple, normalized=True)
    pagerank = _pagerank(simple, alpha=0.85)
    clustering = nx.clustering(simple)
    node_types = {
        NodeType.DOMAIN.value: _node_ids_by_type(simple, NodeType.DOMAIN.value),
        NodeType.GO_TERM.value: _node_ids_by_type(simple, NodeType.GO_TERM.value),
        NodeType.PATHWAY.value: _node_ids_by_type(simple, NodeType.PATHWAY.value),
        NodeType.DISEASE.value: _node_ids_by_type(simple, NodeType.DISEASE.value),
    }
    return {
        "simple": simple,
        "centrality": centrality,
        "pagerank": pagerank,
        "clustering": clustering,
        "node_types": node_types,
    }


def extract_graph_features_for_variant(
    graph: nx.MultiDiGraph,
    variant_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Extract graph-derived features for one variant."""

    simple: nx.Graph = context["simple"]
    centrality: dict[str, float] = context["centrality"]
    pagerank: dict[str, float] = context["pagerank"]
    clustering: dict[str, float] = context["clustering"]
    node_types: dict[str, list[str]] = context["node_types"]

    variant_node = variant_node_id(variant_id)
    residue_node = _located_residue_for_variant(graph, variant_id)
    protein_id = str(graph.nodes[residue_node]["protein_id"])
    protein_node = protein_node_id(protein_id)

    if variant_node in simple:
        personalization = {node_id: 0.0 for node_id in simple.nodes}
        personalization[variant_node] = 1.0
        ppr = _pagerank(simple, alpha=0.85, personalization=personalization)
    else:
        ppr = {node_id: 0.0 for node_id in simple.nodes}

    neighbor_degrees = [
        float(simple.degree(neighbor))
        for neighbor in simple.neighbors(residue_node)
    ] if residue_node in simple else []

    domain_nodes = node_types[NodeType.DOMAIN.value]
    go_nodes = node_types[NodeType.GO_TERM.value]
    pathway_nodes = node_types[NodeType.PATHWAY.value]
    disease_nodes = node_types[NodeType.DISEASE.value]
    biomedical_nodes = domain_nodes + go_nodes + pathway_nodes + disease_nodes

    return {
        "variant_id": variant_id,
        "graph_ppr_residue": float(ppr.get(residue_node, 0.0)),
        "graph_ppr_protein": float(ppr.get(protein_node, 0.0)),
        "graph_ppr_domain_mass": float(sum(ppr.get(node_id, 0.0) for node_id in domain_nodes)),
        "graph_ppr_go_mass": float(sum(ppr.get(node_id, 0.0) for node_id in go_nodes)),
        "graph_ppr_pathway_mass": float(sum(ppr.get(node_id, 0.0) for node_id in pathway_nodes)),
        "graph_ppr_disease_mass": float(sum(ppr.get(node_id, 0.0) for node_id in disease_nodes)),
        "graph_ppr_biomedical_mass": float(sum(ppr.get(node_id, 0.0) for node_id in biomedical_nodes)),
        "graph_residue_domain_edge_count": _incident_count_by_type(
            graph,
            residue_node,
            edge_type=EdgeType.LOCATED_AT,
            neighbor_type=NodeType.DOMAIN,
        ),
        "graph_residue_contact_domain_count": _contact_domain_count(graph, residue_node),
        "graph_protein_go_edge_count": _protein_context_count(
            graph,
            protein_id,
            edge_type=EdgeType.HAS_FUNCTION,
            neighbor_type=NodeType.GO_TERM,
        ),
        "graph_protein_pathway_edge_count": _protein_context_count(
            graph,
            protein_id,
            edge_type=EdgeType.PARTICIPATES_IN,
            neighbor_type=NodeType.PATHWAY,
        ),
        "graph_protein_disease_edge_count": _protein_context_count(
            graph,
            protein_id,
            edge_type=EdgeType.ASSOCIATED_WITH,
            neighbor_type=NodeType.DISEASE,
        ),
        "graph_residue_to_domain_score": _reciprocal_min_distance(simple, residue_node, domain_nodes),
        "graph_residue_to_go_score": _reciprocal_min_distance(simple, residue_node, go_nodes),
        "graph_residue_to_pathway_score": _reciprocal_min_distance(simple, residue_node, pathway_nodes),
        "graph_residue_to_disease_score": _reciprocal_min_distance(simple, residue_node, disease_nodes),
        "graph_residue_degree": float(simple.degree(residue_node)) if residue_node in simple else 0.0,
        "graph_residue_clustering": float(clustering.get(residue_node, 0.0)),
        "graph_residue_betweenness": float(centrality.get(residue_node, 0.0)),
        "graph_residue_pagerank": float(pagerank.get(residue_node, 0.0)),
        "graph_protein_degree": float(simple.degree(protein_node)) if protein_node in simple else 0.0,
        "graph_protein_pagerank": float(pagerank.get(protein_node, 0.0)),
        "graph_residue_neighbor_degree_mean": _mean(neighbor_degrees),
        "graph_component_size": _component_size(simple, residue_node),
    }


def extract_graph_feature_table(
    graph: nx.MultiDiGraph,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    context = _graph_context(graph)
    rows = [
        extract_graph_features_for_variant(graph, str(row["variant_id"]), context)
        for _, row in variants.iterrows()
    ]
    return pd.DataFrame(rows)


def _feature_table_for_graph(
    graph: nx.MultiDiGraph,
    benchmark_config: PipelineConfig,
    config: GraphBaselineConfig,
) -> pd.DataFrame:
    variants = pd.read_csv(benchmark_config.variants)
    transparent = extract_variant_feature_table(graph, variants)
    graph_features = extract_graph_feature_table(graph, variants)
    feature_table = transparent.merge(graph_features, on="variant_id", how="left")
    proteins = pd.read_csv(benchmark_config.proteins)
    if config.family_column in proteins.columns:
        feature_table = feature_table.merge(
            proteins[[config.protein_column, config.family_column]],
            on=config.protein_column,
            how="left",
        )
    return feature_table


def _split_runs(config: GraphBaselineConfig, feature_table: pd.DataFrame) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    if config.strategy in {"all_family_aware", "all_leave_one_protein"}:
        if config.include_random_split:
            runs.append(
                {
                    "run_name": "random_split",
                    "strategy": "existing_split",
                    "holdout_protein_id": "",
                    "holdout_family_id": "",
                }
            )
        for protein_id in sorted(feature_table[config.protein_column].unique()):
            runs.append(
                {
                    "run_name": f"leave_{protein_id}_out",
                    "strategy": "leave_one_protein_out",
                    "holdout_protein_id": str(protein_id),
                    "holdout_family_id": "",
                }
            )
        if config.strategy == "all_family_aware":
            for family_id in sorted(feature_table[config.family_column].unique()):
                runs.append(
                    {
                        "run_name": f"leave_family_{family_id}_out",
                        "strategy": "leave_one_family_out",
                        "holdout_protein_id": "",
                        "holdout_family_id": str(family_id),
                    }
                )
    elif config.strategy == "existing_split":
        runs.append(
            {
                "run_name": "random_split",
                "strategy": "existing_split",
                "holdout_protein_id": "",
                "holdout_family_id": "",
            }
        )
    else:
        raise ValueError(f"Unsupported split strategy: {config.strategy}")
    return runs


def _feature_set_family(feature_set: str) -> str:
    return FEATURE_SET_FAMILIES.get(
        feature_set,
        "transparent_plus_graph" if "transparent" in feature_set else "graph_only",
    )


def _evaluate_feature_table(
    feature_table: pd.DataFrame,
    config: GraphBaselineConfig,
    split_runs: list[dict[str, str]],
    perturbation_name: str,
    perturbation_type: str,
    perturbation_target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for run in split_runs:
        split_table = apply_split_strategy(
            feature_table,
            strategy=run["strategy"],
            protein_column=config.protein_column,
            split_column=config.split_column,
            train_value=config.train_value,
            test_value=config.test_value,
            holdout_protein_id=run["holdout_protein_id"] or None,
            holdout_family_id=run["holdout_family_id"] or None,
            family_column=config.family_column,
        )
        evaluation = run_baseline_evaluation(
            split_table,
            label_column=config.label_column,
            split_column="eval_split",
            train_value=config.train_value,
            test_value=config.test_value,
            feature_sets=config.feature_sets,
        )
        metrics = evaluation.metrics.copy()
        metrics.insert(0, "perturbation", perturbation_name)
        metrics.insert(1, "perturbation_type", perturbation_type)
        metrics.insert(2, "perturbation_target", perturbation_target)
        metrics.insert(3, "run_name", run["run_name"])
        metrics.insert(4, "split_strategy", run["strategy"])
        metrics.insert(5, "holdout_protein_id", run["holdout_protein_id"])
        metrics.insert(6, "holdout_family_id", run["holdout_family_id"])
        metrics["feature_set_family"] = metrics["feature_set"].map(_feature_set_family)
        metric_frames.append(metrics)

        predictions = evaluation.predictions.copy()
        predictions.insert(0, "perturbation", perturbation_name)
        predictions.insert(1, "perturbation_type", perturbation_type)
        predictions.insert(2, "perturbation_target", perturbation_target)
        predictions.insert(3, "run_name", run["run_name"])
        predictions.insert(4, "split_strategy", run["strategy"])
        predictions.insert(5, "holdout_protein_id", run["holdout_protein_id"])
        predictions.insert(6, "holdout_family_id", run["holdout_family_id"])
        predictions["feature_set_family"] = predictions["feature_set"].map(
            _feature_set_family
        )
        if config.family_column in split_table.columns:
            predictions = predictions.merge(
                split_table[["variant_id", config.family_column]].drop_duplicates(),
                on="variant_id",
                how="left",
            )
        prediction_frames.append(predictions)
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _delta_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for key in keys:
        if key in output.columns:
            output[key] = output[key].fillna("").astype(str)
    return output


def _graph_vs_transparent(
    metrics: pd.DataFrame,
    *,
    reference_feature_set: str,
) -> pd.DataFrame:
    keys = [
        "perturbation",
        "run_name",
        "split_strategy",
        "holdout_protein_id",
        "holdout_family_id",
    ]
    numeric_columns = ["accuracy", "precision", "recall", "f1", "auroc"]
    metrics_for_merge = _delta_keys(metrics, keys)
    reference = metrics_for_merge[
        metrics_for_merge["feature_set"] == reference_feature_set
    ][keys + numeric_columns].rename(
        columns={column: f"transparent_{column}" for column in numeric_columns}
    )
    compared = metrics_for_merge.merge(reference, on=keys, how="left")
    for column in numeric_columns:
        compared[f"delta_vs_transparent_{column}"] = (
            compared[column] - compared[f"transparent_{column}"]
        )
    return compared[compared["feature_set"] != reference_feature_set].copy()


def _perturbation_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "run_name",
        "split_strategy",
        "holdout_protein_id",
        "holdout_family_id",
        "feature_set",
    ]
    numeric_columns = ["accuracy", "precision", "recall", "f1", "auroc"]
    metrics_for_merge = _delta_keys(metrics, keys)
    baseline = metrics_for_merge[metrics_for_merge["perturbation"] == "baseline"][
        keys + numeric_columns
    ].rename(columns={column: f"baseline_{column}" for column in numeric_columns})
    merged = metrics_for_merge.merge(baseline, on=keys, how="left")
    for column in numeric_columns:
        merged[f"delta_from_unperturbed_{column}"] = (
            merged[column] - merged[f"baseline_{column}"]
        )
    return merged


def _robustness_summary(
    graph_vs_transparent: pd.DataFrame,
    perturbation_deltas: pd.DataFrame,
) -> pd.DataFrame:
    comparison = (
        graph_vs_transparent.groupby(["perturbation", "feature_set"], dropna=False)
        .agg(
            feature_set_family=("feature_set_family", "first"),
            n_contexts=("run_name", "nunique"),
            mean_delta_vs_transparent_accuracy=(
                "delta_vs_transparent_accuracy",
                "mean",
            ),
            mean_delta_vs_transparent_f1=("delta_vs_transparent_f1", "mean"),
            mean_delta_vs_transparent_auroc=("delta_vs_transparent_auroc", "mean"),
        )
        .reset_index()
    )
    perturb = (
        perturbation_deltas.groupby(["perturbation", "feature_set"], dropna=False)
        .agg(
            mean_delta_from_unperturbed_accuracy=(
                "delta_from_unperturbed_accuracy",
                "mean",
            ),
            mean_delta_from_unperturbed_f1=("delta_from_unperturbed_f1", "mean"),
            mean_delta_from_unperturbed_auroc=("delta_from_unperturbed_auroc", "mean"),
        )
        .reset_index()
    )
    return comparison.merge(perturb, on=["perturbation", "feature_set"], how="left").sort_values(
        ["perturbation", "feature_set"]
    )


def _graph_feature_summary(feature_table: pd.DataFrame) -> pd.DataFrame:
    baseline = feature_table[feature_table["perturbation"] == "baseline"]
    rows: list[dict[str, Any]] = []
    for column in GRAPH_FEATURES:
        values = pd.to_numeric(baseline[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "nonzero_fraction": float((values.fillna(0.0) != 0.0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        existing = [column for column in columns if column in frame.columns]
        frame = frame[existing]
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    config: GraphBaselineConfig,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    graph_vs_transparent: pd.DataFrame,
    perturbation_deltas: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    graph_feature_summary: pd.DataFrame,
) -> None:
    split_summary = (
        metrics.groupby(["split_strategy", "feature_set"], dropna=False)
        .agg(
            feature_set_family=("feature_set_family", "first"),
            n_contexts=("run_name", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_f1=("f1", "mean"),
            mean_auroc=("auroc", "mean"),
        )
        .reset_index()
        .sort_values(["split_strategy", "feature_set"])
    )
    comparison_summary = (
        graph_vs_transparent.groupby(["feature_set", "feature_set_family"], dropna=False)
        .agg(
            mean_delta_vs_transparent_accuracy=(
                "delta_vs_transparent_accuracy",
                "mean",
            ),
            mean_delta_vs_transparent_f1=("delta_vs_transparent_f1", "mean"),
            mean_delta_vs_transparent_auroc=("delta_vs_transparent_auroc", "mean"),
        )
        .reset_index()
        .sort_values("feature_set")
    )
    perturbation_summary = (
        perturbation_deltas[perturbation_deltas["perturbation"] != "baseline"]
        .groupby(["perturbation", "feature_set"], dropna=False)
        .agg(
            mean_delta_from_unperturbed_accuracy=(
                "delta_from_unperturbed_accuracy",
                "mean",
            ),
            mean_delta_from_unperturbed_f1=("delta_from_unperturbed_f1", "mean"),
            mean_delta_from_unperturbed_auroc=("delta_from_unperturbed_auroc", "mean"),
        )
        .reset_index()
        .sort_values(["perturbation", "feature_set"])
    )
    lines = [
        f"# {config.name}",
        "",
        "## Run Metadata",
        "",
        f"- Timestamp: `{metadata['timestamp']}`",
        f"- Random seed: `{metadata['random_seed']}`",
        f"- Benchmark config: `{metadata['benchmark_config']}`",
        f"- TrustProtKG version: `{metadata['trustprotkg_version']}`",
        f"- Perturbation contexts: `{metadata['dataset_summary']['perturbations']}`",
        f"- Split runs: `{metadata['dataset_summary']['split_runs']}`",
        f"- Feature sets: `{metadata['dataset_summary']['feature_sets']}`",
        "",
        "## Graph-Derived Feature Families",
        "",
        "- `graph_pagerank`: personalized PageRank mass around each variant.",
        "- `graph_metapath`: residue-domain/function/pathway/disease path and metapath counts.",
        "- `graph_topology`: compact degree, centrality, clustering, and component summaries.",
        "- `transparent_plus_graph`: existing transparent residue/KG features plus all graph-derived features.",
        "",
        "## Split Context Performance",
        "",
        _markdown_table(
            split_summary,
            [
                "split_strategy",
                "feature_set",
                "feature_set_family",
                "n_contexts",
                "mean_accuracy",
                "mean_f1",
                "mean_auroc",
            ],
        ),
        "",
        "## Graph-Aware Versus Transparent Baseline",
        "",
        _markdown_table(
            comparison_summary,
            [
                "feature_set",
                "feature_set_family",
                "mean_delta_vs_transparent_accuracy",
                "mean_delta_vs_transparent_f1",
                "mean_delta_vs_transparent_auroc",
            ],
        ),
        "",
        "## Perturbation Robustness",
        "",
        _markdown_table(
            perturbation_summary,
            [
                "perturbation",
                "feature_set",
                "mean_delta_from_unperturbed_accuracy",
                "mean_delta_from_unperturbed_f1",
                "mean_delta_from_unperturbed_auroc",
            ],
        ),
        "",
        "## Graph Feature Summary",
        "",
        _markdown_table(
            graph_feature_summary,
            ["feature", "mean", "std", "min", "max", "nonzero_fraction"],
        ),
        "",
        "## Outputs",
        "",
        f"- Feature table: `{_relative(config.feature_table)}`",
        f"- Metrics: `{_relative(config.metrics)}`",
        f"- Graph-vs-transparent comparison: `{_relative(config.graph_vs_transparent)}`",
        f"- Perturbation deltas: `{_relative(config.perturbation_deltas)}`",
        f"- Robustness summary: `{_relative(config.robustness_summary)}`",
        "",
        "## Limitations",
        "",
        "- Graph-aware features are deterministic shallow summaries, not a trained graph neural network.",
        "- The compact benchmark is too small for strong claims about model superiority.",
        "- Perturbation contexts test sensitivity to controlled edge edits, not causal biological removal.",
        "- Graph features can encode evidence availability, so they should be interpreted alongside provenance and ablations.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def run_graph_baseline_experiment(config_path: str | Path) -> GraphBaselineResult:
    config = GraphBaselineConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    base_graph = _build_graph(benchmark_config)
    perturbation_specs = build_perturbation_specs(
        config.perturbation_names,
        config.weaken_confidence_scale,
    )

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    perturbation_summaries: list[dict[str, Any]] = []
    baseline_feature_table: pd.DataFrame | None = None
    split_runs: list[dict[str, str]] | None = None

    for spec in perturbation_specs:
        graph, summary = apply_graph_perturbation(base_graph, spec)
        feature_table = _feature_table_for_graph(graph, benchmark_config, config)
        feature_table.insert(0, "perturbation", spec.name)
        if spec.name == "baseline":
            baseline_feature_table = feature_table.copy()
            split_runs = _split_runs(config, feature_table)
        if split_runs is None:
            raise ValueError("Baseline perturbation must run before other perturbations.")

        eval_table = feature_table.drop(columns=["perturbation"])
        metrics, predictions = _evaluate_feature_table(
            eval_table,
            config,
            split_runs,
            perturbation_name=spec.name,
            perturbation_type=spec.perturbation_type,
            perturbation_target=spec.target,
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        feature_frames.append(feature_table)
        perturbation_summaries.append(summary)

    if baseline_feature_table is None or split_runs is None:
        raise ValueError("Graph baseline config must include the baseline perturbation.")

    all_features = pd.concat(feature_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    graph_vs_transparent = _graph_vs_transparent(
        metrics,
        reference_feature_set=config.transparent_reference_feature_set,
    )
    perturbation_deltas = _perturbation_deltas(metrics)
    robustness_summary = _robustness_summary(graph_vs_transparent, perturbation_deltas)
    graph_feature_summary = _graph_feature_summary(all_features)

    all_features.to_csv(config.feature_table, index=False)
    metrics.to_csv(config.metrics, index=False)
    predictions.to_csv(config.predictions, index=False)
    graph_vs_transparent.to_csv(config.graph_vs_transparent, index=False)
    perturbation_deltas.to_csv(config.perturbation_deltas, index=False)
    robustness_summary.to_csv(config.robustness_summary, index=False)
    graph_feature_summary.to_csv(config.graph_feature_summary, index=False)

    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "benchmark_config": _relative(config.benchmark_config),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "split_strategy": config.strategy,
        "split_runs": split_runs,
        "feature_sets": config.feature_sets,
        "feature_set_families": {
            name: _feature_set_family(name) for name in config.feature_sets
        },
        "graph_feature_columns": GRAPH_FEATURES,
        "transparent_reference_feature_set": config.transparent_reference_feature_set,
        "perturbations": perturbation_summaries,
        "graph_summary": {
            "nodes": int(base_graph.number_of_nodes()),
            "edges": int(base_graph.number_of_edges()),
        },
        "dataset_summary": {
            "perturbations": len(perturbation_specs),
            "split_runs": len(split_runs),
            "feature_sets": len(config.feature_sets),
            "variants": int(baseline_feature_table["variant_id"].nunique()),
            "proteins": int(baseline_feature_table[config.protein_column].nunique()),
            "families": int(baseline_feature_table[config.family_column].nunique()),
            "metric_rows": int(metrics.shape[0]),
            "prediction_rows": int(predictions.shape[0]),
        },
        "outputs": {
            "feature_table": _relative(config.feature_table),
            "metrics": _relative(config.metrics),
            "predictions": _relative(config.predictions),
            "graph_vs_transparent": _relative(config.graph_vs_transparent),
            "perturbation_deltas": _relative(config.perturbation_deltas),
            "robustness_summary": _relative(config.robustness_summary),
            "graph_feature_summary": _relative(config.graph_feature_summary),
            "report": _relative(config.report),
        },
    }
    config.metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        config,
        metadata,
        metrics,
        graph_vs_transparent,
        perturbation_deltas,
        robustness_summary,
        graph_feature_summary,
    )

    return GraphBaselineResult(
        run_dir=config.run_dir,
        feature_table=config.feature_table,
        metrics=config.metrics,
        predictions=config.predictions,
        graph_vs_transparent=config.graph_vs_transparent,
        perturbation_deltas=config.perturbation_deltas,
        robustness_summary=config.robustness_summary,
        graph_feature_summary=config.graph_feature_summary,
        metadata=config.metadata,
        report=config.report,
        perturbation_count=len(perturbation_specs),
        split_run_count=len(split_runs),
        feature_set_count=len(config.feature_sets),
        variant_count=int(baseline_feature_table["variant_id"].nunique()),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TrustProtKG graph-aware baseline modeling."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v1.1_graph_baselines.yaml",
        help="Path to a TrustProtKG graph baseline config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_graph_baseline_experiment(args.config)
    print("TrustProtKG graph-aware baseline experiment complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Perturbations: {result.perturbation_count}")
    print(f"Split runs: {result.split_run_count}")
    print(f"Feature sets: {result.feature_set_count}")
    print(f"Variants: {result.variant_count}")
    print(f"Metrics: {result.metrics}")
    print(f"Graph-vs-transparent: {result.graph_vs_transparent}")
    print(f"Robustness summary: {result.robustness_summary}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
