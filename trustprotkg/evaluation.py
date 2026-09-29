"""Small reproducible baseline evaluation for TrustProtKG benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_SETS = {
    "structure_only": [
        "residue_confidence",
        "num_structural_contacts",
        "avg_contact_distance",
    ],
    "kg_only": [
        "is_in_domain",
        "num_go_pathway_disease_edges",
    ],
    "structure_plus_kg": [
        "residue_confidence",
        "num_structural_contacts",
        "avg_contact_distance",
        "is_in_domain",
        "num_go_pathway_disease_edges",
    ],
}


@dataclass(frozen=True)
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def _as_numeric_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    matrix = frame[list(columns)].copy()
    for column in matrix.columns:
        if matrix[column].dtype == bool:
            matrix[column] = matrix[column].astype(int)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    return matrix.to_numpy(dtype=float)


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.05,
    steps: int = 2000,
    l2: float = 0.01,
) -> np.ndarray:
    x_bias = np.column_stack([np.ones(x.shape[0]), x])
    weights = np.zeros(x_bias.shape[1], dtype=float)
    for _ in range(steps):
        logits = x_bias @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = x_bias.T @ (probabilities - y) / y.size
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def _predict_probabilities(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_bias = np.column_stack([np.ones(x.shape[0]), x])
    logits = x_bias @ weights
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    wins = 0.0
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / float(positives.size * negatives.size)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    tp = int(np.sum((predictions == 1) & (y_true == 1)))
    tn = int(np.sum((predictions == 0) & (y_true == 0)))
    fp = int(np.sum((predictions == 1) & (y_true == 0)))
    fn = int(np.sum((predictions == 0) & (y_true == 1)))
    accuracy = float((tp + tn) / y_true.size)
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _auroc(y_true, probabilities),
    }


def compute_classification_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> dict[str, float]:
    """Compute binary classification metrics from labels and probabilities."""

    return _metrics(
        np.asarray(list(y_true), dtype=int),
        np.asarray(list(probabilities), dtype=float),
    )


def compute_grouped_classification_metrics(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    label_column: str,
    probability_column: str = "prob_pathogenic",
) -> pd.DataFrame:
    """Compute classification metrics within each configured prediction group."""

    required = set(group_columns) | {label_column, probability_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for keys, group in predictions.groupby(list(group_columns), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = compute_classification_metrics(
            group[label_column].astype(int).tolist(),
            group[probability_column].astype(float).tolist(),
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_test": int(group.shape[0]),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def run_baseline_evaluation(
    feature_table: pd.DataFrame,
    *,
    label_column: str = "label_binary",
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    feature_sets: Mapping[str, Sequence[str]] | None = None,
) -> EvaluationResult:
    """Train simple logistic baselines on train rows and score test rows."""

    if label_column not in feature_table.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if split_column not in feature_table.columns:
        raise ValueError(f"Missing split column: {split_column}")

    train = feature_table[feature_table[split_column] == train_value].copy()
    test = feature_table[feature_table[split_column] == test_value].copy()
    if train.empty or test.empty:
        raise ValueError("Benchmark feature table must contain train and test rows.")

    y_train = train[label_column].astype(int).to_numpy()
    y_test = test[label_column].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2:
        raise ValueError("Training split must contain both binary classes.")

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    active_feature_sets = dict(feature_sets or FEATURE_SETS)
    for feature_set_name, columns in active_feature_sets.items():
        x_train = _as_numeric_matrix(train, columns)
        x_test = _as_numeric_matrix(test, columns)
        x_train, x_test = _standardize_train_test(x_train, x_test)
        weights = _fit_logistic_regression(x_train, y_train)
        probabilities = _predict_probabilities(x_test, weights)
        metric_row = {
            "model": "numpy_logistic_regression",
            "feature_set": feature_set_name,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            **_metrics(y_test, probabilities),
        }
        metric_rows.append(metric_row)

        predictions = test[
            ["variant_id", "protein_id", "clinical_label", label_column]
        ].copy()
        predictions["feature_set"] = feature_set_name
        predictions["prob_pathogenic"] = probabilities
        predictions["predicted_label"] = (probabilities >= 0.5).astype(int)
        prediction_frames.append(predictions)

    return EvaluationResult(
        metrics=pd.DataFrame(metric_rows),
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )
