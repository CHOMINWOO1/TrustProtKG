"""External validation snapshot scaffold for TrustProtKG graph baselines."""

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
from trustprotkg.evaluation import run_baseline_evaluation
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.graph_ml import (
    TRANSPARENT_FEATURES,
    _attach_protein_family,
    _spectral_embeddings,
    _variant_embedding_features,
)
from trustprotkg.ingestion import run_ingestion
from trustprotkg.pipeline import _build_graph
from trustprotkg.snapshot_diagnostics import run_snapshot_diagnostics


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
class ExternalValidationConfig:
    path: Path
    name: str
    random_seed: int
    timestamp: str
    training_ingestion_config: Path
    validation_ingestion_config: Path
    validation_diagnostics_config: Path
    training_benchmark_config: Path
    validation_benchmark_config: Path
    internal_graph_ml_metrics: Path
    embedding_dimensions: int
    protein_column: str
    family_column: str
    split_column: str
    train_value: str
    test_value: str
    label_column: str
    run_dir: Path
    training_feature_table: Path
    validation_feature_table: Path
    metrics: Path
    predictions: Path
    comparison: Path
    feature_shift_summary: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "ExternalValidationConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        inputs = raw.get("input", {})
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
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            training_ingestion_config=_resolve_path(
                inputs["training_ingestion_config"],
                config_path,
            ),
            validation_ingestion_config=_resolve_path(
                inputs["validation_ingestion_config"],
                config_path,
            ),
            validation_diagnostics_config=_resolve_path(
                inputs["validation_diagnostics_config"],
                config_path,
            ),
            training_benchmark_config=_resolve_path(
                inputs["training_benchmark_config"],
                config_path,
            ),
            validation_benchmark_config=_resolve_path(
                inputs["validation_benchmark_config"],
                config_path,
            ),
            internal_graph_ml_metrics=_resolve_path(
                inputs["internal_graph_ml_metrics"],
                config_path,
            ),
            embedding_dimensions=int(embedding.get("dimensions", 8)),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            label_column=str(label.get("column", "label_binary")),
            run_dir=run_dir,
            training_feature_table=out_path(
                "training_feature_table",
                "training_variant_features.csv",
            ),
            validation_feature_table=out_path(
                "validation_feature_table",
                "validation_variant_features.csv",
            ),
            metrics=out_path("metrics", "external_validation_metrics.csv"),
            predictions=out_path(
                "predictions",
                "external_validation_predictions.csv",
            ),
            comparison=out_path(
                "comparison",
                "validation_vs_internal_graph_ml.csv",
            ),
            feature_shift_summary=out_path(
                "feature_shift_summary",
                "feature_shift_summary.csv",
            ),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "external_validation_report.md"),
        )


@dataclass(frozen=True)
class ExternalValidationResult:
    run_dir: Path
    training_feature_table: Path
    validation_feature_table: Path
    metrics: Path
    predictions: Path
    comparison: Path
    feature_shift_summary: Path
    metadata: Path
    report: Path
    training_variant_count: int
    validation_variant_count: int
    feature_set_count: int


def _feature_table(
    benchmark_config_path: Path,
    *,
    family_column: str,
    embedding_dimensions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, nx.MultiDiGraph]:
    benchmark_config = PipelineConfig.from_file(benchmark_config_path)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    feature_table = _attach_protein_family(
        feature_table,
        benchmark_config,
        family_column,
    )
    embeddings, embedding_summary = _spectral_embeddings(
        graph,
        dimensions=embedding_dimensions,
    )
    feature_table = _variant_embedding_features(feature_table, embeddings)
    return feature_table, embedding_summary, graph


def _embedding_columns(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
) -> list[str]:
    columns = sorted(
        {
            column
            for column in list(training_features.columns) + list(validation_features.columns)
            if column.startswith("graph_ml_emb_")
        }
    )
    for column in columns:
        if column not in training_features.columns:
            training_features[column] = 0.0
        if column not in validation_features.columns:
            validation_features[column] = 0.0
    return columns


def _feature_sets(embedding_columns: list[str]) -> dict[str, list[str]]:
    return {
        "transparent_structure_plus_kg": TRANSPARENT_FEATURES,
        "spectral_graph_embedding": embedding_columns,
        "transparent_plus_spectral": TRANSPARENT_FEATURES + embedding_columns,
    }


def _external_evaluation(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    config: ExternalValidationConfig,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = training_features[
        training_features[config.split_column] == config.train_value
    ].copy()
    validation = validation_features.copy()
    train["eval_split"] = config.train_value
    validation["eval_split"] = config.test_value
    train["evaluation_snapshot"] = "training"
    validation["evaluation_snapshot"] = "validation"
    combined = pd.concat([train, validation], ignore_index=True, sort=False)
    evaluation = run_baseline_evaluation(
        combined,
        label_column=config.label_column,
        split_column="eval_split",
        train_value=config.train_value,
        test_value=config.test_value,
        feature_sets=feature_sets,
    )
    metrics = evaluation.metrics.copy()
    metrics.insert(0, "evaluation_context", "external_validation_snapshot")
    metrics.insert(1, "train_snapshot", "v1.4_evidence")
    metrics.insert(2, "validation_snapshot", "v2.1_validation")
    predictions = evaluation.predictions.copy()
    predictions.insert(0, "evaluation_context", "external_validation_snapshot")
    predictions.insert(1, "train_snapshot", "v1.4_evidence")
    predictions.insert(2, "validation_snapshot", "v2.1_validation")
    return metrics, predictions


def _internal_comparison(
    external_metrics: pd.DataFrame,
    internal_metrics_path: Path,
) -> pd.DataFrame:
    external = external_metrics[
        ["feature_set", "accuracy", "f1", "auroc", "n_train", "n_test"]
    ].rename(
        columns={
            "accuracy": "external_accuracy",
            "f1": "external_f1",
            "auroc": "external_auroc",
            "n_train": "external_n_train",
            "n_test": "external_n_test",
        }
    )
    if internal_metrics_path.exists():
        internal_metrics = pd.read_csv(internal_metrics_path)
        internal = (
            internal_metrics.groupby("feature_set", as_index=False)
            .agg(
                internal_run_count=("run_name", "nunique"),
                internal_mean_accuracy=("accuracy", "mean"),
                internal_mean_f1=("f1", "mean"),
                internal_mean_auroc=("auroc", "mean"),
            )
            .sort_values("feature_set")
        )
    else:
        internal = pd.DataFrame({"feature_set": external["feature_set"]})
        internal["internal_run_count"] = 0
        internal["internal_mean_accuracy"] = np.nan
        internal["internal_mean_f1"] = np.nan
        internal["internal_mean_auroc"] = np.nan
    comparison = external.merge(internal, on="feature_set", how="left")
    comparison["delta_accuracy_external_vs_internal_mean"] = (
        comparison["external_accuracy"] - comparison["internal_mean_accuracy"]
    )
    comparison["delta_f1_external_vs_internal_mean"] = (
        comparison["external_f1"] - comparison["internal_mean_f1"]
    )
    comparison["delta_auroc_external_vs_internal_mean"] = (
        comparison["external_auroc"] - comparison["internal_mean_auroc"]
    )
    return comparison.sort_values("feature_set")


def _feature_shift_summary(
    training_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        training_values = pd.to_numeric(training_features[column], errors="coerce")
        validation_values = pd.to_numeric(validation_features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "training_mean": float(training_values.mean()),
                "validation_mean": float(validation_values.mean()),
                "delta_mean_validation_minus_training": float(
                    validation_values.mean() - training_values.mean()
                ),
                "training_std": float(training_values.std(ddof=0)),
                "validation_std": float(validation_values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(
    config: ExternalValidationConfig,
    *,
    validation_summary: dict[str, Any],
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_shift: pd.DataFrame,
) -> Path:
    validation = validation_summary["validation"]
    transform_rows = pd.DataFrame(
        [
            {"source_type": source_type, **summary}
            for source_type, summary in validation_summary["transformations"].items()
        ]
    )
    lines = [
        f"# {config.name}",
        "",
        "## Summary",
        "",
        "- Training snapshot: `v1.4_evidence`",
        "- Validation snapshot: `v2.1_validation`",
        f"- Validation variants: `{validation['variant_count']}`",
        f"- Validation proteins: `{validation['protein_count']}`",
        f"- Validation residue mapping rate: `{validation['variant_residue_mapping_rate']:.3f}`",
        f"- Validation CA coverage rate: `{validation['variant_ca_coverage_rate']:.3f}`",
        "",
        "## External Validation Metrics",
        "",
        _markdown_table(
            metrics,
            ["feature_set", "n_train", "n_test", "accuracy", "precision", "recall", "f1", "auroc"],
        ),
        "",
        "## Internal Mean Versus Validation Snapshot",
        "",
        _markdown_table(
            comparison,
            [
                "feature_set",
                "external_f1",
                "internal_mean_f1",
                "delta_f1_external_vs_internal_mean",
                "external_auroc",
                "internal_mean_auroc",
                "delta_auroc_external_vs_internal_mean",
            ],
        ),
        "",
        "## Feature Shift Probe",
        "",
        _markdown_table(
            feature_shift,
            [
                "feature",
                "training_mean",
                "validation_mean",
                "delta_mean_validation_minus_training",
            ],
        ),
        "",
        "## Snapshot Transformations",
        "",
        _markdown_table(transform_rows),
        "",
        "## Interpretation Guardrail",
        "",
        "The v2.1 validation pack is a compact offline scaffold for exercising external-validation plumbing. It is not an independent clinical validation cohort. Spectral graph-embedding coordinates are deterministic within each graph, but this prototype does not yet learn a transferable embedding space across snapshots.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")
    return config.report


def run_external_validation(config_path: str | Path) -> ExternalValidationResult:
    config = ExternalValidationConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    training_ingestion = run_ingestion(config.training_ingestion_config)
    validation_ingestion = run_ingestion(config.validation_ingestion_config)
    diagnostics_result = run_snapshot_diagnostics(config.validation_diagnostics_config)

    training_features, training_embedding_summary, training_graph = _feature_table(
        config.training_benchmark_config,
        family_column=config.family_column,
        embedding_dimensions=config.embedding_dimensions,
    )
    validation_features, validation_embedding_summary, validation_graph = _feature_table(
        config.validation_benchmark_config,
        family_column=config.family_column,
        embedding_dimensions=config.embedding_dimensions,
    )
    embedding_columns = _embedding_columns(training_features, validation_features)
    feature_sets = _feature_sets(embedding_columns)
    metrics, predictions = _external_evaluation(
        training_features,
        validation_features,
        config,
        feature_sets,
    )
    comparison = _internal_comparison(metrics, config.internal_graph_ml_metrics)
    feature_shift = _feature_shift_summary(
        training_features,
        validation_features,
        TRANSPARENT_FEATURES + embedding_columns,
    )

    config.training_feature_table.parent.mkdir(parents=True, exist_ok=True)
    training_features.to_csv(config.training_feature_table, index=False)
    validation_features.to_csv(config.validation_feature_table, index=False)
    metrics.to_csv(config.metrics, index=False)
    predictions.to_csv(config.predictions, index=False)
    comparison.to_csv(config.comparison, index=False)
    feature_shift.to_csv(config.feature_shift_summary, index=False)

    validation_summary = _read_json(validation_ingestion.validation_summary)
    _write_report(
        config,
        validation_summary=validation_summary,
        metrics=metrics,
        comparison=comparison,
        feature_shift=feature_shift,
    )
    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "trustprotkg_version": _project_version(),
        "training": {
            "ingestion_config": _relative(config.training_ingestion_config),
            "benchmark_config": _relative(config.training_benchmark_config),
            "validation_summary": _relative(training_ingestion.validation_summary),
            "variants": int(training_features["variant_id"].nunique()),
            "graph_nodes": int(training_graph.number_of_nodes()),
            "graph_edges": int(training_graph.number_of_edges()),
            "embedding_dimensions": int(training_embedding_summary.shape[0]),
        },
        "validation": {
            "ingestion_config": _relative(config.validation_ingestion_config),
            "benchmark_config": _relative(config.validation_benchmark_config),
            "validation_summary": _relative(validation_ingestion.validation_summary),
            "diagnostics_metadata": _relative(diagnostics_result.metadata),
            "variants": int(validation_features["variant_id"].nunique()),
            "graph_nodes": int(validation_graph.number_of_nodes()),
            "graph_edges": int(validation_graph.number_of_edges()),
            "embedding_dimensions": int(validation_embedding_summary.shape[0]),
            "warnings": validation_summary["validation"]["warnings"],
            "residue_mapping_rate": validation_summary["validation"][
                "variant_residue_mapping_rate"
            ],
            "ca_coverage_rate": validation_summary["validation"][
                "variant_ca_coverage_rate"
            ],
        },
        "feature_sets": feature_sets,
        "outputs": {
            "training_feature_table": _relative(config.training_feature_table),
            "validation_feature_table": _relative(config.validation_feature_table),
            "metrics": _relative(config.metrics),
            "predictions": _relative(config.predictions),
            "comparison": _relative(config.comparison),
            "feature_shift_summary": _relative(config.feature_shift_summary),
            "report": _relative(config.report),
        },
        "dataset_summary": {
            "training_variants": int(training_features["variant_id"].nunique()),
            "validation_variants": int(validation_features["variant_id"].nunique()),
            "feature_sets": len(feature_sets),
            "metric_rows": int(metrics.shape[0]),
            "prediction_rows": int(predictions.shape[0]),
        },
    }
    config.metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ExternalValidationResult(
        run_dir=config.run_dir,
        training_feature_table=config.training_feature_table,
        validation_feature_table=config.validation_feature_table,
        metrics=config.metrics,
        predictions=config.predictions,
        comparison=config.comparison,
        feature_shift_summary=config.feature_shift_summary,
        metadata=config.metadata,
        report=config.report,
        training_variant_count=int(training_features["variant_id"].nunique()),
        validation_variant_count=int(validation_features["variant_id"].nunique()),
        feature_set_count=len(feature_sets),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TrustProtKG external validation snapshot scaffold."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v2.1_external_validation.yaml",
        help="Path to a TrustProtKG external validation config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_external_validation(args.config)
    print("TrustProtKG external validation scaffold complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Training variants: {result.training_variant_count}")
    print(f"Validation variants: {result.validation_variant_count}")
    print(f"Feature sets: {result.feature_set_count}")
    print(f"Metrics: {result.metrics}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
