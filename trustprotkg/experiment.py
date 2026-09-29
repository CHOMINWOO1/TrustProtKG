"""Paper-style experiment runner for TrustProtKG v0.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from trustprotkg.config import PipelineConfig, load_simple_yaml
from trustprotkg.evaluation import EvaluationResult, run_baseline_evaluation
from trustprotkg.export import export_graphml, export_jsonl
from trustprotkg.explanations import explain_variant, export_variant_explanations
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.pipeline import _build_graph


def _resolve_path(path_value: str | Path, base_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or not (base_path.parent / path).exists():
        return cwd_path
    return base_path.parent / path


def _split_csv(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    name: str
    benchmark_config: Path
    random_seed: int
    timestamp: str
    label_column: str
    split_column: str
    train_value: str
    test_value: str
    metrics: list[str]
    feature_sets: dict[str, list[str]]
    run_dir: Path
    graph_jsonl: Path
    graphml: Path
    feature_table: Path
    metrics_output: Path
    predictions: Path
    error_analysis: Path
    explanations: Path
    explanation_summaries: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        split = raw.get("split", {})
        label = raw.get("label", {})
        metrics = raw.get("metrics", {})
        feature_sets = raw.get("feature_sets", {})
        outputs = raw.get("outputs", {})

        run_dir = _resolve_path(outputs["run_dir"], config_path)

        def out_path(key: str, default_name: str) -> Path:
            value = outputs.get(key, default_name)
            path_value = Path(str(value))
            if path_value.is_absolute():
                return path_value
            return run_dir / path_value

        return cls(
            path=config_path,
            name=str(experiment["name"]),
            benchmark_config=_resolve_path(experiment["benchmark_config"], config_path),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(
                experiment.get(
                    "timestamp",
                    datetime.now(UTC).replace(microsecond=0).isoformat(),
                )
            ),
            label_column=str(label.get("column", "label_binary")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            metrics=_split_csv(metrics.get("names", "accuracy,precision,recall,f1,auroc")),
            feature_sets={
                str(name): _split_csv(columns)
                for name, columns in feature_sets.items()
            },
            run_dir=run_dir,
            graph_jsonl=out_path("graph_jsonl", "graph.jsonl"),
            graphml=out_path("graphml", "graph.graphml"),
            feature_table=out_path("feature_table", "variant_features.csv"),
            metrics_output=out_path("metrics", "metrics.csv"),
            predictions=out_path("predictions", "predictions.csv"),
            error_analysis=out_path("error_analysis", "error_analysis.csv"),
            explanations=out_path("explanations", "variant_explanations.jsonl"),
            explanation_summaries=out_path(
                "explanation_summaries", "explanation_summaries.jsonl"
            ),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "report.md"),
        )


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    metadata: Path
    report: Path
    metrics: Path
    predictions: Path
    error_analysis: Path
    explanation_summaries: Path
    node_count: int
    edge_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_version() -> str:
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    with pyproject.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _input_hashes(benchmark_config: PipelineConfig) -> dict[str, str]:
    named_paths = {
        "proteins": benchmark_config.proteins,
        "variants": benchmark_config.variants,
        "go_annotations": benchmark_config.go_annotations,
        "disease_associations": benchmark_config.disease_associations,
        "domains": benchmark_config.domains,
        "pathways": benchmark_config.pathways,
    }
    if benchmark_config.structures is not None:
        named_paths["structures"] = benchmark_config.structures
        structures = pd.read_csv(benchmark_config.structures)
        for _, row in structures.iterrows():
            structure_file = Path(str(row["structure_file"]))
            if not structure_file.is_absolute():
                structure_file = Path.cwd() / structure_file
            named_paths[f"structure:{row['structure_id']}"] = structure_file
    elif benchmark_config.structure_file is not None:
        named_paths["structure"] = benchmark_config.structure_file

    return {name: _sha256(path) for name, path in named_paths.items()}


def _graph_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    node_type_counts: dict[str, int] = {}
    edge_type_counts: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = str(attrs.get("node_type", "unknown"))
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    for _, _, attrs in graph.edges(data=True):
        edge_type = str(attrs.get("edge_type", "unknown"))
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
    }


def _dataset_summary(feature_table: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "variants": int(feature_table.shape[0]),
        "proteins": int(feature_table["protein_id"].nunique()),
        "train_variants": int(
            (feature_table[config.split_column] == config.train_value).sum()
        ),
        "test_variants": int(
            (feature_table[config.split_column] == config.test_value).sum()
        ),
        "positive_variants": int(feature_table[config.label_column].astype(int).sum()),
        "negative_variants": int(
            (feature_table[config.label_column].astype(int) == 0).sum()
        ),
    }


def _feature_summary(
    feature_table: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    numeric_columns = sorted({column for columns in feature_sets.values() for column in columns})
    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        series = feature_table[column]
        if series.dtype == bool:
            series = series.astype(int)
        series = pd.to_numeric(series, errors="coerce")
        rows.append(
            {
                "feature": column,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def _error_analysis(predictions: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows = predictions.copy()
    rows["error_type"] = "correct"
    false_positive = (rows["predicted_label"] == 1) & (rows[label_column] == 0)
    false_negative = (rows["predicted_label"] == 0) & (rows[label_column] == 1)
    rows.loc[false_positive, "error_type"] = "false_positive"
    rows.loc[false_negative, "error_type"] = "false_negative"
    return rows[rows["error_type"] != "correct"].reset_index(drop=True)


def _explanation_summary_record(
    graph: nx.MultiDiGraph,
    feature_table: pd.DataFrame,
    prediction_row: pd.Series,
    primary_feature_set: str,
) -> dict[str, Any]:
    variant_id = str(prediction_row["variant_id"])
    feature_row = feature_table[feature_table["variant_id"] == variant_id].iloc[0].to_dict()
    explanation = explain_variant(graph, variant_id, feature_row, max_contacts=3)
    evidence = explanation["evidence"]
    return {
        "variant_id": variant_id,
        "protein_id": str(prediction_row["protein_id"]),
        "feature_set": primary_feature_set,
        "true_label": int(prediction_row["label_binary"]),
        "predicted_label": int(prediction_row["predicted_label"]),
        "prob_pathogenic": float(prediction_row["prob_pathogenic"]),
        "residue": explanation["residue"],
        "feature_snapshot": {
            "residue_confidence": feature_row.get("residue_confidence"),
            "num_structural_contacts": feature_row.get("num_structural_contacts"),
            "avg_contact_distance": feature_row.get("avg_contact_distance"),
            "is_in_domain": feature_row.get("is_in_domain"),
            "num_go_pathway_disease_edges": feature_row.get(
                "num_go_pathway_disease_edges"
            ),
        },
        "supporting_evidence": {
            "domain_membership": evidence["domain_membership"],
            "nearest_structural_contacts": evidence["structural_contacts"],
            "protein_context": evidence["protein_context"],
        },
    }


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        frame = frame[columns]
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    config: ExperimentConfig,
    metadata: dict[str, Any],
    dataset_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    feature_summary: pd.DataFrame,
    evaluation: EvaluationResult,
    error_analysis: pd.DataFrame,
    explanation_summaries: list[dict[str, Any]],
) -> None:
    ablation_columns = [
        "feature_set",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
    ]
    selected_explanations = explanation_summaries[:3]
    lines = [
        f"# {config.name}",
        "",
        "## Run Metadata",
        "",
        f"- Timestamp: `{metadata['timestamp']}`",
        f"- Random seed: `{metadata['random_seed']}`",
        f"- Experiment config: `{metadata['config_path']}`",
        f"- Benchmark config: `{metadata['benchmark_config']}`",
        f"- TrustProtKG version: `{metadata['trustprotkg_version']}`",
        "",
        "## Dataset Summary",
        "",
        _markdown_table(pd.DataFrame([dataset_summary])),
        "",
        "## Graph Summary",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "nodes": graph_summary["nodes"],
                        "edges": graph_summary["edges"],
                    }
                ]
            )
        ),
        "",
        "Node type counts:",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"node_type": key, "count": value}
                    for key, value in sorted(graph_summary["node_type_counts"].items())
                ]
            )
        ),
        "",
        "Edge type counts:",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"edge_type": key, "count": value}
                    for key, value in sorted(graph_summary["edge_type_counts"].items())
                ]
            )
        ),
        "",
        "## Feature Summary",
        "",
        _markdown_table(feature_summary),
        "",
        "## Metric Table",
        "",
        _markdown_table(evaluation.metrics),
        "",
        "## Ablation Table",
        "",
        _markdown_table(evaluation.metrics, ablation_columns),
        "",
        "## Error Analysis",
        "",
        _markdown_table(
            error_analysis,
            [
                "variant_id",
                "protein_id",
                "feature_set",
                "clinical_label",
                "label_binary",
                "prob_pathogenic",
                "predicted_label",
                "error_type",
            ],
        ),
        "",
        "## Selected Variant Explanations",
        "",
    ]
    for summary in selected_explanations:
        evidence = summary["supporting_evidence"]
        domain_count = len(evidence["domain_membership"])
        contact_count = len(evidence["nearest_structural_contacts"])
        context_count = len(evidence["protein_context"])
        lines.extend(
            [
                f"### {summary['variant_id']}",
                "",
                f"- Prediction: `{summary['prob_pathogenic']:.3f}` pathogenic probability",
                f"- True/predicted label: `{summary['true_label']}` / `{summary['predicted_label']}`",
                f"- Residue confidence: `{summary['feature_snapshot']['residue_confidence']}`",
                f"- Domain evidence items: `{domain_count}`",
                f"- Nearest structural contacts summarized: `{contact_count}`",
                f"- GO/pathway/disease evidence items: `{context_count}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "",
            "- The benchmark is deliberately small and toy-data based.",
            "- Labels are curated for software validation rather than clinical use.",
            "- The logistic baseline is meant as a transparent sanity check, not a competitive pathogenicity predictor.",
            "- pLDDT-like residue confidence is read from B-factor fields in local PDB fragments.",
            "- Real experiments should use versioned external snapshots and protein-family aware splits.",
            "",
        ]
    )
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config_path: str | Path) -> ExperimentResult:
    config = ExperimentConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    evaluation = run_baseline_evaluation(
        feature_table,
        label_column=config.label_column,
        split_column=config.split_column,
        train_value=config.train_value,
        test_value=config.test_value,
        feature_sets=config.feature_sets,
    )
    errors = _error_analysis(evaluation.predictions, config.label_column)

    export_jsonl(graph, config.graph_jsonl)
    export_graphml(graph, config.graphml)
    feature_table.to_csv(config.feature_table, index=False)
    evaluation.metrics.to_csv(config.metrics_output, index=False)
    evaluation.predictions.to_csv(config.predictions, index=False)
    errors.to_csv(config.error_analysis, index=False)
    export_variant_explanations(graph, feature_table, config.explanations)

    primary_feature_set = (
        "structure_plus_kg"
        if "structure_plus_kg" in config.feature_sets
        else next(iter(config.feature_sets))
    )
    primary_predictions = evaluation.predictions[
        evaluation.predictions["feature_set"] == primary_feature_set
    ].copy()
    explanation_summaries = [
        _explanation_summary_record(graph, feature_table, row, primary_feature_set)
        for _, row in primary_predictions.iterrows()
    ]
    _write_jsonl(explanation_summaries, config.explanation_summaries)

    dataset_summary = _dataset_summary(feature_table, config)
    graph_summary = _graph_summary(graph)
    feature_summary = _feature_summary(feature_table, config.feature_sets)
    metadata = {
        "timestamp": config.timestamp,
        "config_path": str(config.path.relative_to(Path.cwd())),
        "benchmark_config": str(config.benchmark_config.relative_to(Path.cwd())),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "input_hashes": _input_hashes(benchmark_config),
        "dataset_summary": dataset_summary,
        "graph_summary": graph_summary,
        "feature_sets": config.feature_sets,
        "metrics": config.metrics,
        "outputs": {
            "graph_jsonl": str(config.graph_jsonl.relative_to(Path.cwd())),
            "graphml": str(config.graphml.relative_to(Path.cwd())),
            "feature_table": str(config.feature_table.relative_to(Path.cwd())),
            "metrics": str(config.metrics_output.relative_to(Path.cwd())),
            "predictions": str(config.predictions.relative_to(Path.cwd())),
            "error_analysis": str(config.error_analysis.relative_to(Path.cwd())),
            "explanations": str(config.explanations.relative_to(Path.cwd())),
            "explanation_summaries": str(
                config.explanation_summaries.relative_to(Path.cwd())
            ),
            "report": str(config.report.relative_to(Path.cwd())),
        },
    }
    config.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        config,
        metadata,
        dataset_summary,
        graph_summary,
        feature_summary,
        evaluation,
        errors,
        explanation_summaries,
    )

    return ExperimentResult(
        run_dir=config.run_dir,
        metadata=config.metadata,
        report=config.report,
        metrics=config.metrics_output,
        predictions=config.predictions,
        error_analysis=config.error_analysis,
        explanation_summaries=config.explanation_summaries,
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a TrustProtKG experiment.")
    parser.add_argument(
        "--config",
        default="experiments/configs/v0.3.yaml",
        help="Path to a TrustProtKG experiment config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_experiment(args.config)
    print(
        "TrustProtKG experiment complete: "
        f"{result.node_count} nodes, {result.edge_count} edges"
    )
    print(f"Run dir: {result.run_dir}")
    print(f"Metadata: {result.metadata}")
    print(f"Metrics: {result.metrics}")
    print(f"Predictions: {result.predictions}")
    print(f"Error analysis: {result.error_analysis}")
    print(f"Explanation summaries: {result.explanation_summaries}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
