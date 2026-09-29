"""Command-line pipelines for TrustProtKG demos and benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trustprotkg.config import PipelineConfig
from trustprotkg.evaluation import run_baseline_evaluation
from trustprotkg.export import export_graphml, export_jsonl
from trustprotkg.explanations import export_variant_explanations
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.kg import build_benchmark_knowledge_graph, build_knowledge_graph


@dataclass(frozen=True)
class PipelineResult:
    graph_jsonl: Path
    graphml: Path
    variant_features: Path
    baseline_metrics: Path | None
    baseline_predictions: Path | None
    explanations: Path | None
    node_count: int
    edge_count: int


def _build_graph(config: PipelineConfig):
    if config.structures is not None:
        return build_benchmark_knowledge_graph(
            proteins_path=config.proteins,
            variants_path=config.variants,
            go_annotations_path=config.go_annotations,
            disease_associations_path=config.disease_associations,
            domains_path=config.domains,
            pathways_path=config.pathways,
            structures_path=config.structures,
            contact_threshold_angstrom=config.contact_threshold_angstrom,
            default_structure_source_db=config.structure_source_db,
            created_at=config.created_at,
            derived_from=config.derived_from,
        )
    if config.structure_file is None or config.structure_id is None:
        raise ValueError("Single-structure configs require structure_file and structure_id.")
    return build_knowledge_graph(
        proteins_path=config.proteins,
        variants_path=config.variants,
        go_annotations_path=config.go_annotations,
        disease_associations_path=config.disease_associations,
        domains_path=config.domains,
        pathways_path=config.pathways,
        structure_file=config.structure_file,
        protein_id=config.protein_id,
        structure_id=config.structure_id,
        contact_threshold_angstrom=config.contact_threshold_angstrom,
        structure_source_db=config.structure_source_db,
        created_at=config.created_at,
        derived_from=config.derived_from,
    )


def run_pipeline(config_path: str | Path) -> PipelineResult:
    config = PipelineConfig.from_file(config_path)
    graph = _build_graph(config)

    export_jsonl(graph, config.graph_jsonl)
    export_graphml(graph, config.graphml)

    variants = pd.read_csv(config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    config.variant_features.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(config.variant_features, index=False)

    if config.baseline_metrics is not None or config.baseline_predictions is not None:
        evaluation = run_baseline_evaluation(feature_table)
        if config.baseline_metrics is not None:
            config.baseline_metrics.parent.mkdir(parents=True, exist_ok=True)
            evaluation.metrics.to_csv(config.baseline_metrics, index=False)
        if config.baseline_predictions is not None:
            config.baseline_predictions.parent.mkdir(parents=True, exist_ok=True)
            evaluation.predictions.to_csv(config.baseline_predictions, index=False)

    if config.explanations is not None:
        export_variant_explanations(graph, feature_table, config.explanations)

    return PipelineResult(
        graph_jsonl=config.graph_jsonl,
        graphml=config.graphml,
        variant_features=config.variant_features,
        baseline_metrics=config.baseline_metrics,
        baseline_predictions=config.baseline_predictions,
        explanations=config.explanations,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TrustProtKG demo pipeline.")
    parser.add_argument(
        "--config",
        default="configs/demo.yaml",
        help="Path to a TrustProtKG YAML-style config file.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_pipeline(args.config)
    print(
        "TrustProtKG pipeline complete: "
        f"{result.node_count} nodes, {result.edge_count} edges"
    )
    print(f"JSONL: {result.graph_jsonl}")
    print(f"GraphML: {result.graphml}")
    print(f"Variant features: {result.variant_features}")
    if result.baseline_metrics is not None:
        print(f"Baseline metrics: {result.baseline_metrics}")
    if result.baseline_predictions is not None:
        print(f"Baseline predictions: {result.baseline_predictions}")
    if result.explanations is not None:
        print(f"Explanations: {result.explanations}")


if __name__ == "__main__":
    main()
