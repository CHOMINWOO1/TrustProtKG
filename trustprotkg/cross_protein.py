"""Cross-protein generalization experiments for TrustProtKG v0.6."""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trustprotkg.config import PipelineConfig, load_simple_yaml
from trustprotkg.evaluation import (
    EvaluationResult,
    compute_grouped_classification_metrics,
    run_baseline_evaluation,
)
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
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _project_version() -> str:
    pyproject = Path.cwd() / "pyproject.toml"
    if not pyproject.exists():
        return "unknown"
    with pyproject.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


@dataclass(frozen=True)
class CrossProteinConfig:
    path: Path
    name: str
    benchmark_config: Path
    random_seed: int
    timestamp: str
    strategy: str
    holdout_protein_id: str | None
    holdout_family_id: str | None
    user_split_file: Path | None
    include_random_split: bool
    label_column: str
    split_column: str
    train_value: str
    test_value: str
    protein_column: str
    family_column: str
    low_confidence_threshold: float
    sparse_kg_threshold: int
    feature_sets: dict[str, list[str]]
    run_dir: Path
    feature_table: Path
    metrics: Path
    predictions: Path
    per_protein_metrics: Path
    per_family_metrics: Path
    failure_modes: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "CrossProteinConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        split = raw.get("split", {})
        label = raw.get("label", {})
        analysis = raw.get("analysis", {})
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
            timestamp=str(experiment.get("timestamp", "")),
            strategy=str(split.get("strategy", "existing_split")),
            holdout_protein_id=(
                str(split["holdout_protein_id"])
                if "holdout_protein_id" in split
                else None
            ),
            holdout_family_id=(
                str(split["holdout_family_id"])
                if "holdout_family_id" in split
                else None
            ),
            user_split_file=(
                _resolve_path(split["user_split_file"], config_path)
                if "user_split_file" in split
                else None
            ),
            include_random_split=bool(split.get("include_random_split", False)),
            label_column=str(label.get("column", "label_binary")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            low_confidence_threshold=float(
                analysis.get("low_confidence_threshold", 75.0)
            ),
            sparse_kg_threshold=int(analysis.get("sparse_kg_threshold", 1)),
            feature_sets={
                str(name): _split_csv(columns)
                for name, columns in feature_sets.items()
            },
            run_dir=run_dir,
            feature_table=out_path("feature_table", "variant_features.csv"),
            metrics=out_path("metrics", "metrics.csv"),
            predictions=out_path("predictions", "predictions.csv"),
            per_protein_metrics=out_path(
                "per_protein_metrics", "per_protein_metrics.csv"
            ),
            per_family_metrics=out_path(
                "per_family_metrics", "per_family_metrics.csv"
            ),
            failure_modes=out_path("failure_modes", "failure_modes.csv"),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "cross_protein_report.md"),
        )


@dataclass(frozen=True)
class CrossProteinResult:
    run_dir: Path
    metrics: Path
    predictions: Path
    per_protein_metrics: Path
    per_family_metrics: Path
    failure_modes: Path
    metadata: Path
    report: Path
    run_count: int


def apply_split_strategy(
    feature_table: pd.DataFrame,
    *,
    strategy: str,
    protein_column: str = "protein_id",
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    holdout_protein_id: str | None = None,
    holdout_family_id: str | None = None,
    user_split: pd.DataFrame | None = None,
    family_column: str = "protein_family",
) -> pd.DataFrame:
    """Return a feature table with `eval_split` according to a strategy."""

    table = feature_table.copy()
    if strategy == "existing_split":
        table["eval_split"] = table[split_column].astype(str)
    elif strategy == "leave_one_protein_out":
        if not holdout_protein_id:
            raise ValueError("leave_one_protein_out requires holdout_protein_id")
        table["eval_split"] = np.where(
            table[protein_column] == holdout_protein_id,
            test_value,
            train_value,
        )
    elif strategy == "leave_one_family_out":
        if not holdout_family_id:
            raise ValueError("leave_one_family_out requires holdout_family_id")
        if family_column not in table.columns:
            raise ValueError(f"Feature table missing family column: {family_column}")
        table["eval_split"] = np.where(
            table[family_column] == holdout_family_id,
            test_value,
            train_value,
        )
    elif strategy == "user_defined":
        if user_split is None:
            raise ValueError("user_defined strategy requires a user split table")
        required = {"variant_id", split_column}
        missing = required.difference(user_split.columns)
        if missing:
            raise ValueError(f"user split missing columns: {sorted(missing)}")
        table = table.drop(columns=["eval_split"], errors="ignore").merge(
            user_split[["variant_id", split_column]].rename(
                columns={split_column: "eval_split"}
            ),
            on="variant_id",
            how="left",
        )
        if table["eval_split"].isna().any():
            missing_variants = table[table["eval_split"].isna()]["variant_id"].tolist()
            raise ValueError(f"user split missing variants: {missing_variants}")
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if not {train_value, test_value}.issubset(set(table["eval_split"])):
        raise ValueError("Split strategy must produce both train and test rows.")
    return table


def _evaluate_one_run(
    feature_table: pd.DataFrame,
    config: CrossProteinConfig,
    *,
    run_name: str,
    strategy: str,
    holdout_protein_id: str | None,
    holdout_family_id: str | None,
    user_split: pd.DataFrame | None = None,
) -> EvaluationResult:
    split_table = apply_split_strategy(
        feature_table,
        strategy=strategy,
        protein_column=config.protein_column,
        split_column=config.split_column,
        train_value=config.train_value,
        test_value=config.test_value,
        holdout_protein_id=holdout_protein_id,
        holdout_family_id=holdout_family_id,
        user_split=user_split,
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
    evaluation.metrics.insert(0, "run_name", run_name)
    evaluation.metrics.insert(1, "split_strategy", strategy)
    evaluation.metrics.insert(2, "holdout_protein_id", holdout_protein_id or "")
    evaluation.metrics.insert(3, "holdout_family_id", holdout_family_id or "")
    evaluation.predictions.insert(0, "run_name", run_name)
    evaluation.predictions.insert(1, "split_strategy", strategy)
    evaluation.predictions.insert(2, "holdout_protein_id", holdout_protein_id or "")
    evaluation.predictions.insert(3, "holdout_family_id", holdout_family_id or "")
    predictions = evaluation.predictions
    if config.family_column in split_table.columns:
        family_lookup = split_table[["variant_id", config.family_column]].drop_duplicates()
        predictions = predictions.merge(
            family_lookup,
            on="variant_id",
            how="left",
        )
    return EvaluationResult(metrics=evaluation.metrics, predictions=predictions)


def _per_protein_metrics(
    predictions: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    return compute_grouped_classification_metrics(
        predictions,
        group_columns=[
            "run_name",
            "split_strategy",
            "holdout_protein_id",
            "feature_set",
            "protein_id",
        ],
        label_column=label_column,
        probability_column="prob_pathogenic",
    )


def _per_family_metrics(
    predictions: pd.DataFrame,
    *,
    label_column: str,
    family_column: str,
) -> pd.DataFrame:
    if family_column not in predictions.columns:
        return pd.DataFrame()
    return compute_grouped_classification_metrics(
        predictions,
        group_columns=[
            "run_name",
            "split_strategy",
            "holdout_family_id",
            "feature_set",
            family_column,
        ],
        label_column=label_column,
        probability_column="prob_pathogenic",
    )


def _failure_modes(
    predictions: pd.DataFrame,
    feature_table: pd.DataFrame,
    config: CrossProteinConfig,
) -> pd.DataFrame:
    merged = predictions.merge(
        feature_table[
            [
                "variant_id",
                "residue_confidence",
                "num_go_pathway_disease_edges",
                "is_in_domain",
            ]
        ],
        on="variant_id",
        how="left",
    )
    merged["is_false_positive"] = (
        (merged["predicted_label"] == 1)
        & (merged[config.label_column].astype(int) == 0)
    )
    merged["is_false_negative"] = (
        (merged["predicted_label"] == 0)
        & (merged[config.label_column].astype(int) == 1)
    )
    merged["is_low_confidence_structure"] = (
        pd.to_numeric(merged["residue_confidence"], errors="coerce")
        < config.low_confidence_threshold
    )
    merged["is_sparse_kg_context"] = (
        pd.to_numeric(merged["num_go_pathway_disease_edges"], errors="coerce")
        <= config.sparse_kg_threshold
    )
    merged["is_missing_domain_context"] = ~merged["is_in_domain"].astype(bool)

    rows: list[dict[str, Any]] = []
    group_columns = ["run_name", "split_strategy", "holdout_protein_id", "feature_set", "protein_id"]
    for keys, group in merged.groupby(group_columns, dropna=False):
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_test": int(group.shape[0]),
                "false_positives": int(group["is_false_positive"].sum()),
                "false_negatives": int(group["is_false_negative"].sum()),
                "low_confidence_structure_cases": int(
                    group["is_low_confidence_structure"].sum()
                ),
                "sparse_kg_context_cases": int(group["is_sparse_kg_context"].sum()),
                "missing_domain_context_cases": int(
                    group["is_missing_domain_context"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _robustness_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    leave_one = metrics[metrics["split_strategy"] == "leave_one_protein_out"]
    if leave_one.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for feature_set, group in leave_one.groupby("feature_set"):
        rows.append(
            {
                "feature_set": feature_set,
                "mean_accuracy": float(group["accuracy"].mean()),
                "std_accuracy": float(group["accuracy"].std(ddof=0)),
                "mean_f1": float(group["f1"].mean()),
                "std_f1": float(group["f1"].std(ddof=0)),
                "mean_auroc": float(group["auroc"].mean()),
                "n_leave_one_runs": int(group["run_name"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        frame = frame[columns]
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
    config: CrossProteinConfig,
    metadata: dict[str, Any],
    metrics: pd.DataFrame,
    per_protein: pd.DataFrame,
    per_family: pd.DataFrame,
    failure_modes: pd.DataFrame,
) -> None:
    random_metrics = metrics[metrics["split_strategy"] == "existing_split"]
    leave_one_metrics = metrics[metrics["split_strategy"] == "leave_one_protein_out"]
    leave_family_metrics = metrics[
        metrics["split_strategy"] == "leave_one_family_out"
    ]
    robustness = _robustness_summary(metrics)
    lines = [
        f"# {config.name}",
        "",
        "## Run Metadata",
        "",
        f"- Timestamp: `{metadata['timestamp']}`",
        f"- Random seed: `{metadata['random_seed']}`",
        f"- Benchmark config: `{metadata['benchmark_config']}`",
        f"- TrustProtKG version: `{metadata['trustprotkg_version']}`",
        f"- Split strategy: `{config.strategy}`",
        "",
        "## Split Strategy Summary",
        "",
        _markdown_table(pd.DataFrame(metadata["split_runs"])),
        "",
        "## Random Split Performance",
        "",
        _markdown_table(
            random_metrics,
            [
                "run_name",
                "feature_set",
                "n_train",
                "n_test",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
            ],
        ),
        "",
        "## Leave-One-Protein-Out Performance",
        "",
        _markdown_table(
            leave_one_metrics,
            [
                "run_name",
                "holdout_protein_id",
                "feature_set",
                "n_train",
                "n_test",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
            ],
        ),
        "",
        "## Leave-One-Family-Out Performance",
        "",
        (
            _markdown_table(
                leave_family_metrics,
                [
                    "run_name",
                    "holdout_family_id",
                    "feature_set",
                    "n_train",
                    "n_test",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auroc",
                ],
            )
            if not leave_family_metrics.empty
            else "_No leave-one-family-out runs._"
        ),
        "",
        "## Per-Protein Metrics",
        "",
        _markdown_table(
            per_protein,
            [
                "run_name",
                "feature_set",
                "protein_id",
                "n_test",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auroc",
            ],
        ),
        "",
        "## Per-Family Metrics",
        "",
        (
            _markdown_table(
                per_family,
                [
                    "run_name",
                    "feature_set",
                    config.family_column,
                    "n_test",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auroc",
                ],
            )
            if not per_family.empty
            else "_No protein-family metadata available._"
        ),
        "",
        "## Feature-Set Robustness Across Proteins",
        "",
        _markdown_table(robustness),
        "",
        "## Failure-Mode Summary",
        "",
        _markdown_table(
            failure_modes,
            [
                "run_name",
                "feature_set",
                "protein_id",
                "false_positives",
                "false_negatives",
                "low_confidence_structure_cases",
                "sparse_kg_context_cases",
                "missing_domain_context_cases",
            ],
        ),
        "",
        "## Limitations",
        "",
        "- The v0.5 benchmark is intentionally tiny, so leave-one-protein-out scores are diagnostic rather than definitive.",
        "- AUROC is reported only when a test group contains both classes.",
        "- Compact PDB-style fragments preserve mapped residue evidence but are not full-length structural models.",
        "- Larger real snapshots and protein-family-aware splits are needed before making strong performance claims.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def _split_runs(config: CrossProteinConfig, feature_table: pd.DataFrame) -> list[dict[str, str]]:
    runs: list[dict[str, str]] = []
    if config.strategy in {"all_leave_one_protein", "all_family_aware"}:
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
            if config.family_column not in feature_table.columns:
                raise ValueError(
                    f"all_family_aware requires family column: {config.family_column}"
                )
            for family_id in sorted(feature_table[config.family_column].unique()):
                runs.append(
                    {
                        "run_name": f"leave_family_{family_id}_out",
                        "strategy": "leave_one_family_out",
                        "holdout_protein_id": "",
                        "holdout_family_id": str(family_id),
                    }
                )
    elif config.strategy == "all_leave_one_family":
        if config.include_random_split:
            runs.append(
                {
                    "run_name": "random_split",
                    "strategy": "existing_split",
                    "holdout_protein_id": "",
                    "holdout_family_id": "",
                }
            )
        if config.family_column not in feature_table.columns:
            raise ValueError(
                f"all_leave_one_family requires family column: {config.family_column}"
            )
        for family_id in sorted(feature_table[config.family_column].unique()):
            runs.append(
                {
                    "run_name": f"leave_family_{family_id}_out",
                    "strategy": "leave_one_family_out",
                    "holdout_protein_id": "",
                    "holdout_family_id": str(family_id),
                }
            )
    elif config.strategy == "leave_one_protein_out":
        runs.append(
            {
                "run_name": f"leave_{config.holdout_protein_id}_out",
                "strategy": "leave_one_protein_out",
                "holdout_protein_id": config.holdout_protein_id or "",
                "holdout_family_id": "",
            }
        )
    elif config.strategy == "leave_one_family_out":
        runs.append(
            {
                "run_name": f"leave_family_{config.holdout_family_id}_out",
                "strategy": "leave_one_family_out",
                "holdout_protein_id": "",
                "holdout_family_id": config.holdout_family_id or "",
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
    elif config.strategy == "user_defined":
        runs.append(
            {
                "run_name": "user_defined_split",
                "strategy": "user_defined",
                "holdout_protein_id": "",
                "holdout_family_id": "",
            }
        )
    else:
        raise ValueError(f"Unsupported split strategy: {config.strategy}")
    return runs


def run_cross_protein_experiment(config_path: str | Path) -> CrossProteinResult:
    config = CrossProteinConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    proteins = pd.read_csv(benchmark_config.proteins)
    if config.family_column in proteins.columns:
        feature_table = feature_table.merge(
            proteins[[config.protein_column, config.family_column]],
            on=config.protein_column,
            how="left",
        )
    config.feature_table.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(config.feature_table, index=False)

    user_split = pd.read_csv(config.user_split_file) if config.user_split_file else None
    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    split_runs = _split_runs(config, feature_table)
    for run in split_runs:
        evaluation = _evaluate_one_run(
            feature_table,
            config,
            run_name=run["run_name"],
            strategy=run["strategy"],
            holdout_protein_id=run["holdout_protein_id"] or None,
            holdout_family_id=run["holdout_family_id"] or None,
            user_split=user_split,
        )
        metric_frames.append(evaluation.metrics)
        prediction_frames.append(evaluation.predictions)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    per_protein = _per_protein_metrics(predictions, label_column=config.label_column)
    per_family = _per_family_metrics(
        predictions,
        label_column=config.label_column,
        family_column=config.family_column,
    )
    failure_modes = _failure_modes(predictions, feature_table, config)

    metrics.to_csv(config.metrics, index=False)
    predictions.to_csv(config.predictions, index=False)
    per_protein.to_csv(config.per_protein_metrics, index=False)
    per_family.to_csv(config.per_family_metrics, index=False)
    failure_modes.to_csv(config.failure_modes, index=False)

    metadata = {
        "timestamp": config.timestamp,
        "config_path": str(config.path.relative_to(Path.cwd())),
        "benchmark_config": str(config.benchmark_config.relative_to(Path.cwd())),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "split_strategy": config.strategy,
        "split_runs": split_runs,
        "feature_sets": config.feature_sets,
        "graph_summary": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        },
        "dataset_summary": {
            "variants": int(feature_table.shape[0]),
            "proteins": int(feature_table[config.protein_column].nunique()),
            "families": (
                int(feature_table[config.family_column].nunique())
                if config.family_column in feature_table.columns
                else 0
            ),
        },
        "outputs": {
            "feature_table": str(config.feature_table.relative_to(Path.cwd())),
            "metrics": str(config.metrics.relative_to(Path.cwd())),
            "predictions": str(config.predictions.relative_to(Path.cwd())),
            "per_protein_metrics": str(
                config.per_protein_metrics.relative_to(Path.cwd())
            ),
            "per_family_metrics": str(
                config.per_family_metrics.relative_to(Path.cwd())
            ),
            "failure_modes": str(config.failure_modes.relative_to(Path.cwd())),
            "report": str(config.report.relative_to(Path.cwd())),
        },
    }
    config.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(config, metadata, metrics, per_protein, per_family, failure_modes)

    return CrossProteinResult(
        run_dir=config.run_dir,
        metrics=config.metrics,
        predictions=config.predictions,
        per_protein_metrics=config.per_protein_metrics,
        per_family_metrics=config.per_family_metrics,
        failure_modes=config.failure_modes,
        metadata=config.metadata,
        report=config.report,
        run_count=len(split_runs),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a TrustProtKG cross-protein generalization experiment."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v0.6_all_leave_one_protein_summary.yaml",
        help="Path to a TrustProtKG cross-protein config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_cross_protein_experiment(args.config)
    print("TrustProtKG cross-protein experiment complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Split runs: {result.run_count}")
    print(f"Metrics: {result.metrics}")
    print(f"Predictions: {result.predictions}")
    print(f"Per-protein metrics: {result.per_protein_metrics}")
    print(f"Per-family metrics: {result.per_family_metrics}")
    print(f"Failure modes: {result.failure_modes}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
