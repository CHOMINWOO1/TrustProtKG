"""Calibration and threshold reporting for TrustProtKG graph ML outputs."""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trustprotkg.config import load_simple_yaml


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
class ModelCalibrationConfig:
    path: Path
    name: str
    random_seed: int
    timestamp: str
    internal_predictions: Path
    validation_predictions: Path
    internal_metrics: Path
    validation_metrics: Path
    bins: int
    thresholds: list[float]
    run_dir: Path
    combined_predictions: Path
    calibration_bins: Path
    calibration_summary: Path
    threshold_sweep: Path
    decision_thresholds: Path
    class_balance: Path
    metadata: Path
    report: Path
    readme: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelCalibrationConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        inputs = raw.get("input", {})
        calibration = raw.get("calibration", {})
        outputs = raw.get("outputs", {})
        run_dir = _resolve_path(outputs["run_dir"], config_path)

        def out_path(key: str, default_name: str) -> Path:
            value = Path(str(outputs.get(key, default_name)))
            return value if value.is_absolute() else run_dir / value

        thresholds = [
            float(value.strip())
            for value in str(calibration.get("thresholds", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")).split(",")
            if value.strip()
        ]
        return cls(
            path=config_path,
            name=str(experiment["name"]),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            internal_predictions=_resolve_path(inputs["internal_predictions"], config_path),
            validation_predictions=_resolve_path(
                inputs["validation_predictions"],
                config_path,
            ),
            internal_metrics=_resolve_path(inputs["internal_metrics"], config_path),
            validation_metrics=_resolve_path(inputs["validation_metrics"], config_path),
            bins=int(calibration.get("bins", 5)),
            thresholds=thresholds,
            run_dir=run_dir,
            combined_predictions=out_path("combined_predictions", "combined_predictions.csv"),
            calibration_bins=out_path("calibration_bins", "calibration_bins.csv"),
            calibration_summary=out_path("calibration_summary", "calibration_summary.csv"),
            threshold_sweep=out_path("threshold_sweep", "threshold_sweep.csv"),
            decision_thresholds=out_path("decision_thresholds", "decision_thresholds.csv"),
            class_balance=out_path("class_balance", "class_balance.csv"),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "model_calibration_report.md"),
            readme=out_path("readme", "README.md"),
        )


@dataclass(frozen=True)
class ModelCalibrationResult:
    run_dir: Path
    combined_predictions: Path
    calibration_bins: Path
    calibration_summary: Path
    threshold_sweep: Path
    decision_thresholds: Path
    class_balance: Path
    metadata: Path
    report: Path
    readme: Path
    prediction_count: int
    threshold_row_count: int
    decision_row_count: int


def _load_predictions(config: ModelCalibrationConfig) -> pd.DataFrame:
    internal = pd.read_csv(config.internal_predictions).fillna("")
    internal["evaluation_context"] = "internal_family_aware"
    validation = pd.read_csv(config.validation_predictions).fillna("")
    if "run_name" not in validation.columns:
        validation["run_name"] = "external_validation"
    if "split_strategy" not in validation.columns:
        validation["split_strategy"] = "external_validation_snapshot"
    required = {
        "evaluation_context",
        "run_name",
        "variant_id",
        "protein_id",
        "clinical_label",
        "label_binary",
        "feature_set",
        "prob_pathogenic",
    }
    frames = []
    for label, frame in [("internal_predictions", internal), ("validation_predictions", validation)]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        active = frame[list(required)].copy()
        active["source_table"] = label
        active["label_binary"] = active["label_binary"].astype(int)
        active["prob_pathogenic"] = pd.to_numeric(active["prob_pathogenic"], errors="coerce")
        frames.append(active)
    combined = pd.concat(frames, ignore_index=True)
    if combined["prob_pathogenic"].isna().any():
        raise ValueError("Predictions contain non-numeric prob_pathogenic values.")
    return combined


def _classification_metrics(labels: pd.Series, probabilities: pd.Series, threshold: float) -> dict[str, float]:
    y_true = labels.astype(int).to_numpy()
    scores = probabilities.astype(float).to_numpy()
    y_pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    total = max(1, int(y_true.size))
    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "f1": f1,
        "predicted_positive_fraction": float(y_pred.mean()) if y_pred.size else 0.0,
    }


def _class_balance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (context, feature_set), group in predictions.groupby(["evaluation_context", "feature_set"], sort=True):
        labels = group["label_binary"].astype(int)
        positives = int(labels.sum())
        negatives = int((labels == 0).sum())
        total = int(group.shape[0])
        rows.append(
            {
                "evaluation_context": context,
                "feature_set": feature_set,
                "n_predictions": total,
                "positive_labels": positives,
                "negative_labels": negatives,
                "positive_fraction": positives / max(1, total),
                "minority_fraction": min(positives, negatives) / max(1, total),
            }
        )
    return pd.DataFrame(rows)


def _threshold_sweep(predictions: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (context, feature_set), group in predictions.groupby(["evaluation_context", "feature_set"], sort=True):
        for threshold in thresholds:
            rows.append(
                {
                    "evaluation_context": context,
                    "feature_set": feature_set,
                    "threshold": threshold,
                    "n_predictions": int(group.shape[0]),
                    **_classification_metrics(
                        group["label_binary"],
                        group["prob_pathogenic"],
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _decision_thresholds(threshold_sweep: pd.DataFrame, class_balance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    balance_lookup = {
        (row["evaluation_context"], row["feature_set"]): row
        for _, row in class_balance.iterrows()
    }
    for (context, feature_set), group in threshold_sweep.groupby(["evaluation_context", "feature_set"], sort=True):
        best_f1 = group.sort_values(["f1", "balanced_accuracy", "threshold"], ascending=[False, False, True]).iloc[0]
        best_balanced = group.sort_values(["balanced_accuracy", "f1", "threshold"], ascending=[False, False, True]).iloc[0]
        default_rows = group[np.isclose(group["threshold"], 0.5)]
        default = default_rows.iloc[0] if not default_rows.empty else group.iloc[(group["threshold"] - 0.5).abs().argmin()]
        balance = balance_lookup[(context, feature_set)]
        rows.append(
            {
                "evaluation_context": context,
                "feature_set": feature_set,
                "default_threshold": float(default["threshold"]),
                "default_f1": float(default["f1"]),
                "default_balanced_accuracy": float(default["balanced_accuracy"]),
                "best_f1_threshold": float(best_f1["threshold"]),
                "best_f1": float(best_f1["f1"]),
                "best_balanced_accuracy_threshold": float(best_balanced["threshold"]),
                "best_balanced_accuracy": float(best_balanced["balanced_accuracy"]),
                "positive_fraction": float(balance["positive_fraction"]),
                "minority_fraction": float(balance["minority_fraction"]),
            }
        )
    return pd.DataFrame(rows)


def _calibration_bins(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for (context, feature_set), group in predictions.groupby(["evaluation_context", "feature_set"], sort=True):
        assigned = group.copy()
        assigned["bin_index"] = pd.cut(
            assigned["prob_pathogenic"],
            bins=edges,
            labels=False,
            include_lowest=True,
        )
        for bin_index in range(bins):
            bin_group = assigned[assigned["bin_index"] == bin_index]
            if bin_group.empty:
                rows.append(
                    {
                        "evaluation_context": context,
                        "feature_set": feature_set,
                        "bin_index": bin_index,
                        "bin_lower": edges[bin_index],
                        "bin_upper": edges[bin_index + 1],
                        "n_predictions": 0,
                        "mean_predicted_probability": np.nan,
                        "observed_positive_fraction": np.nan,
                        "absolute_calibration_error": np.nan,
                    }
                )
                continue
            mean_prob = float(bin_group["prob_pathogenic"].mean())
            observed = float(bin_group["label_binary"].astype(int).mean())
            rows.append(
                {
                    "evaluation_context": context,
                    "feature_set": feature_set,
                    "bin_index": bin_index,
                    "bin_lower": edges[bin_index],
                    "bin_upper": edges[bin_index + 1],
                    "n_predictions": int(bin_group.shape[0]),
                    "mean_predicted_probability": mean_prob,
                    "observed_positive_fraction": observed,
                    "absolute_calibration_error": abs(mean_prob - observed),
                }
            )
    return pd.DataFrame(rows)


def _calibration_summary(calibration_bins: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (context, feature_set), group in calibration_bins.groupby(["evaluation_context", "feature_set"], sort=True):
        nonempty = group[group["n_predictions"] > 0].copy()
        total = max(1, int(nonempty["n_predictions"].sum()))
        weighted_error = (
            nonempty["n_predictions"] * nonempty["absolute_calibration_error"]
        ).sum() / total
        rows.append(
            {
                "evaluation_context": context,
                "feature_set": feature_set,
                "n_predictions": total,
                "nonempty_bins": int(nonempty.shape[0]),
                "expected_calibration_error": float(weighted_error),
                "max_bin_calibration_error": float(nonempty["absolute_calibration_error"].max()),
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


def _write_report(
    config: ModelCalibrationConfig,
    *,
    calibration_summary: pd.DataFrame,
    decision_thresholds: pd.DataFrame,
    class_balance: pd.DataFrame,
) -> None:
    lines = [
        f"# {config.name}",
        "",
        "## Calibration Summary",
        "",
        _markdown_table(
            calibration_summary,
            [
                "evaluation_context",
                "feature_set",
                "n_predictions",
                "nonempty_bins",
                "expected_calibration_error",
                "max_bin_calibration_error",
            ],
        ),
        "",
        "## Decision Threshold Summary",
        "",
        _markdown_table(
            decision_thresholds,
            [
                "evaluation_context",
                "feature_set",
                "default_f1",
                "default_balanced_accuracy",
                "best_f1_threshold",
                "best_f1",
                "best_balanced_accuracy_threshold",
                "best_balanced_accuracy",
            ],
        ),
        "",
        "## Class Balance",
        "",
        _markdown_table(
            class_balance,
            [
                "evaluation_context",
                "feature_set",
                "n_predictions",
                "positive_labels",
                "negative_labels",
                "positive_fraction",
                "minority_fraction",
            ],
        ),
        "",
        "## Interpretation Guardrail",
        "",
        "Calibration and threshold sweeps are descriptive diagnostics over compact prototype predictions. They should be read beside provenance, validation leakage warnings, and evidence-card context rather than treated as deployment-ready calibration evidence.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(config: ModelCalibrationConfig, result: ModelCalibrationResult | None = None) -> None:
    lines = [
        f"# {config.name}",
        "",
        "This folder contains model calibration and decision-threshold diagnostics for TrustProtKG graph ML outputs.",
        "",
        "## Rebuild",
        "",
        "```bash",
        f"python -m trustprotkg.model_calibration --config {_relative(config.path)}",
        "```",
        "",
        "## Main Files",
        "",
        f"- Combined predictions: `{_relative(config.combined_predictions)}`",
        f"- Calibration bins: `{_relative(config.calibration_bins)}`",
        f"- Calibration summary: `{_relative(config.calibration_summary)}`",
        f"- Threshold sweep: `{_relative(config.threshold_sweep)}`",
        f"- Decision thresholds: `{_relative(config.decision_thresholds)}`",
        f"- Class balance: `{_relative(config.class_balance)}`",
        f"- Report: `{_relative(config.report)}`",
        "",
    ]
    if result is not None:
        lines.extend(
            [
                "## Generated Counts",
                "",
                f"- Predictions: `{result.prediction_count}`",
                f"- Threshold rows: `{result.threshold_row_count}`",
                f"- Decision rows: `{result.decision_row_count}`",
                "",
            ]
        )
    config.readme.parent.mkdir(parents=True, exist_ok=True)
    config.readme.write_text("\n".join(lines), encoding="utf-8")


def run_model_calibration(config_path: str | Path) -> ModelCalibrationResult:
    config = ModelCalibrationConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(config)
    class_balance = _class_balance(predictions)
    threshold_sweep = _threshold_sweep(predictions, config.thresholds)
    decision_thresholds = _decision_thresholds(threshold_sweep, class_balance)
    calibration_bins = _calibration_bins(predictions, config.bins)
    calibration_summary = _calibration_summary(calibration_bins)

    config.combined_predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(config.combined_predictions, index=False)
    calibration_bins.to_csv(config.calibration_bins, index=False)
    calibration_summary.to_csv(config.calibration_summary, index=False)
    threshold_sweep.to_csv(config.threshold_sweep, index=False)
    decision_thresholds.to_csv(config.decision_thresholds, index=False)
    class_balance.to_csv(config.class_balance, index=False)
    _write_report(
        config,
        calibration_summary=calibration_summary,
        decision_thresholds=decision_thresholds,
        class_balance=class_balance,
    )
    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "trustprotkg_version": _project_version(),
        "inputs": {
            "internal_predictions": _relative(config.internal_predictions),
            "validation_predictions": _relative(config.validation_predictions),
            "internal_metrics": _relative(config.internal_metrics),
            "validation_metrics": _relative(config.validation_metrics),
        },
        "calibration": {
            "bins": config.bins,
            "thresholds": config.thresholds,
        },
        "counts": {
            "combined_predictions": int(predictions.shape[0]),
            "contexts": int(predictions["evaluation_context"].nunique()),
            "feature_sets": int(predictions["feature_set"].nunique()),
            "calibration_bin_rows": int(calibration_bins.shape[0]),
            "threshold_rows": int(threshold_sweep.shape[0]),
            "decision_threshold_rows": int(decision_thresholds.shape[0]),
        },
        "outputs": {
            "combined_predictions": _relative(config.combined_predictions),
            "calibration_bins": _relative(config.calibration_bins),
            "calibration_summary": _relative(config.calibration_summary),
            "threshold_sweep": _relative(config.threshold_sweep),
            "decision_thresholds": _relative(config.decision_thresholds),
            "class_balance": _relative(config.class_balance),
            "report": _relative(config.report),
            "readme": _relative(config.readme),
        },
    }
    config.metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = ModelCalibrationResult(
        run_dir=config.run_dir,
        combined_predictions=config.combined_predictions,
        calibration_bins=config.calibration_bins,
        calibration_summary=config.calibration_summary,
        threshold_sweep=config.threshold_sweep,
        decision_thresholds=config.decision_thresholds,
        class_balance=config.class_balance,
        metadata=config.metadata,
        report=config.report,
        readme=config.readme,
        prediction_count=int(predictions.shape[0]),
        threshold_row_count=int(threshold_sweep.shape[0]),
        decision_row_count=int(decision_thresholds.shape[0]),
    )
    _write_readme(config, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TrustProtKG model calibration and threshold report."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v2.5_model_calibration.yaml",
        help="Path to a TrustProtKG model calibration config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_model_calibration(args.config)
    print("TrustProtKG model calibration complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Predictions: {result.prediction_count}")
    print(f"Threshold rows: {result.threshold_row_count}")
    print(f"Decision rows: {result.decision_row_count}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
