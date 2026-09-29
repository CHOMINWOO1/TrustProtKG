"""Evidence perturbation experiments for TrustProtKG v0.9."""

from __future__ import annotations

import argparse
import copy
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

from trustprotkg.config import PipelineConfig, load_simple_yaml
from trustprotkg.cross_protein import apply_split_strategy
from trustprotkg.evaluation import run_baseline_evaluation
from trustprotkg.explanation_quality import (
    QUALITY_WEIGHTS,
    score_explanation_quality,
)
from trustprotkg.explanations import explain_variant
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.models import EdgeType, NodeType
from trustprotkg.pipeline import _build_graph


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


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    perturbation_type: str
    target: str
    source_db: str = ""
    confidence_scale: float = 1.0


@dataclass(frozen=True)
class PerturbationConfig:
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
    max_contacts: int
    source_diversity_cap: int
    close_contact_distance: float
    supported_contact_distance: float
    min_residue_confidence: float
    perturbation_names: list[str]
    weaken_confidence_scale: float
    case_study_variants: list[str]
    run_dir: Path
    feature_table: Path
    variant_quality_scores: Path
    metrics: Path
    predictions: Path
    prediction_deltas: Path
    quality_deltas: Path
    protein_robustness: Path
    family_robustness: Path
    before_after_cards_jsonl: Path
    before_after_cards_markdown: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "PerturbationConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        split = raw.get("split", {})
        label = raw.get("label", {})
        feature_sets = raw.get("feature_sets", {})
        quality = raw.get("quality", {})
        perturbations = raw.get("perturbations", {})
        case_studies = raw.get("case_studies", {})
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
            strategy=str(split.get("strategy", "all_family_aware")),
            include_random_split=bool(split.get("include_random_split", True)),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            split_column=str(split.get("column", "split")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            label_column=str(label.get("column", "label_binary")),
            feature_sets={
                str(name): _split_csv(columns)
                for name, columns in feature_sets.items()
            },
            max_contacts=int(quality.get("max_contacts", 5)),
            source_diversity_cap=int(quality.get("source_diversity_cap", 6)),
            close_contact_distance=float(quality.get("close_contact_distance", 5.0)),
            supported_contact_distance=float(
                quality.get("supported_contact_distance", 8.0)
            ),
            min_residue_confidence=float(quality.get("min_residue_confidence", 70.0)),
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
            case_study_variants=_split_csv(case_studies.get("variant_ids", "")),
            run_dir=run_dir,
            feature_table=out_path("feature_table", "variant_features.csv"),
            variant_quality_scores=out_path(
                "variant_quality_scores", "variant_quality_scores.csv"
            ),
            metrics=out_path("metrics", "metrics.csv"),
            predictions=out_path("predictions", "predictions.csv"),
            prediction_deltas=out_path("prediction_deltas", "prediction_deltas.csv"),
            quality_deltas=out_path("quality_deltas", "quality_deltas.csv"),
            protein_robustness=out_path("protein_robustness", "protein_robustness.csv"),
            family_robustness=out_path("family_robustness", "family_robustness.csv"),
            before_after_cards_jsonl=out_path(
                "before_after_cards_jsonl", "before_after_cards.jsonl"
            ),
            before_after_cards_markdown=out_path(
                "before_after_cards_markdown", "before_after_cards.md"
            ),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "perturbation_report.md"),
        )


@dataclass(frozen=True)
class PerturbationResult:
    run_dir: Path
    variant_quality_scores: Path
    metrics: Path
    predictions: Path
    prediction_deltas: Path
    quality_deltas: Path
    protein_robustness: Path
    family_robustness: Path
    before_after_cards_jsonl: Path
    before_after_cards_markdown: Path
    metadata: Path
    report: Path
    perturbation_count: int
    split_run_count: int
    variant_count: int
    card_count: int


def build_perturbation_specs(
    names: list[str],
    confidence_scale: float,
) -> list[PerturbationSpec]:
    catalog = {
        "baseline": PerturbationSpec("baseline", "none", "none"),
        "remove_structural_contacts": PerturbationSpec(
            "remove_structural_contacts",
            "remove_edge_type",
            EdgeType.STRUCTURAL_CONTACT.value,
        ),
        "remove_domain_edges": PerturbationSpec(
            "remove_domain_edges",
            "remove_domain_context",
            "Domain",
        ),
        "remove_go_edges": PerturbationSpec(
            "remove_go_edges",
            "remove_edge_type",
            EdgeType.HAS_FUNCTION.value,
        ),
        "remove_pathway_edges": PerturbationSpec(
            "remove_pathway_edges",
            "remove_edge_type",
            EdgeType.PARTICIPATES_IN.value,
        ),
        "remove_disease_edges": PerturbationSpec(
            "remove_disease_edges",
            "remove_edge_type",
            EdgeType.ASSOCIATED_WITH.value,
        ),
        "remove_source_db_interpro": PerturbationSpec(
            "remove_source_db_interpro",
            "remove_source_db",
            "InterPro real-derived snapshot",
            source_db="InterPro real-derived snapshot",
        ),
        "weaken_source_db_go": PerturbationSpec(
            "weaken_source_db_go",
            "weaken_source_db_confidence",
            "GO real-derived snapshot",
            source_db="GO real-derived snapshot",
            confidence_scale=confidence_scale,
        ),
    }
    missing = [name for name in names if name not in catalog]
    if missing:
        raise ValueError(f"Unsupported perturbation names: {missing}")
    return [catalog[name] for name in names]


def _source_db(attrs: dict[str, Any]) -> str:
    provenance = attrs.get("provenance", {})
    return str(attrs.get("source_db") or provenance.get("source_db") or "")


def _node_type(graph: nx.MultiDiGraph, node_id: str) -> str:
    return str(graph.nodes[node_id].get("node_type", ""))


def _remove_edges(
    graph: nx.MultiDiGraph,
    predicate: Callable[[str, str, dict[str, Any]], bool],
) -> int:
    to_remove: list[tuple[str, str, int]] = []
    for source, target, key, attrs in graph.edges(keys=True, data=True):
        if predicate(source, target, attrs):
            to_remove.append((source, target, key))
    for source, target, key in to_remove:
        graph.remove_edge(source, target, key)
    return len(to_remove)


def _weaken_source_confidence(
    graph: nx.MultiDiGraph,
    *,
    source_db: str,
    confidence_scale: float,
) -> int:
    changed = 0
    for _source, _target, _key, attrs in graph.edges(keys=True, data=True):
        if _source_db(attrs) != source_db:
            continue
        if "confidence" in attrs:
            attrs["confidence"] = float(attrs["confidence"]) * confidence_scale
        provenance = attrs.get("provenance")
        if isinstance(provenance, dict) and "confidence" in provenance:
            provenance["confidence"] = float(provenance["confidence"]) * confidence_scale
        changed += 1
    return changed


def apply_graph_perturbation(
    graph: nx.MultiDiGraph,
    spec: PerturbationSpec,
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    """Return a perturbed graph copy and a compact perturbation summary."""

    perturbed = copy.deepcopy(graph)
    summary = {
        "perturbation": spec.name,
        "perturbation_type": spec.perturbation_type,
        "target": spec.target,
        "source_db": spec.source_db,
        "confidence_scale": spec.confidence_scale,
        "edges_before": int(perturbed.number_of_edges()),
        "edges_removed": 0,
        "edges_weakened": 0,
    }

    if spec.perturbation_type == "none":
        pass
    elif spec.perturbation_type == "remove_edge_type":
        summary["edges_removed"] = _remove_edges(
            perturbed,
            lambda _source, _target, attrs: attrs.get("edge_type") == spec.target,
        )
    elif spec.perturbation_type == "remove_domain_context":
        summary["edges_removed"] = _remove_edges(
            perturbed,
            lambda source, target, attrs: (
                attrs.get("edge_type") == EdgeType.HAS_DOMAIN.value
                or (
                    attrs.get("edge_type") == EdgeType.LOCATED_AT.value
                    and (
                        _node_type(perturbed, source) == NodeType.DOMAIN.value
                        or _node_type(perturbed, target) == NodeType.DOMAIN.value
                    )
                )
            ),
        )
    elif spec.perturbation_type == "remove_source_db":
        summary["edges_removed"] = _remove_edges(
            perturbed,
            lambda _source, _target, attrs: _source_db(attrs) == spec.source_db,
        )
    elif spec.perturbation_type == "weaken_source_db_confidence":
        summary["edges_weakened"] = _weaken_source_confidence(
            perturbed,
            source_db=spec.source_db,
            confidence_scale=spec.confidence_scale,
        )
    else:
        raise ValueError(f"Unsupported perturbation type: {spec.perturbation_type}")

    summary["edges_after"] = int(perturbed.number_of_edges())
    return perturbed, summary


def _feature_table_for_graph(
    graph: nx.MultiDiGraph,
    benchmark_config: PipelineConfig,
    config: PerturbationConfig,
) -> pd.DataFrame:
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    proteins = pd.read_csv(benchmark_config.proteins)
    if config.family_column in proteins.columns:
        feature_table = feature_table.merge(
            proteins[[config.protein_column, config.family_column]],
            on=config.protein_column,
            how="left",
        )
    return feature_table


def _split_runs(config: PerturbationConfig, feature_table: pd.DataFrame) -> list[dict[str, str]]:
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


def _evaluate_feature_table(
    feature_table: pd.DataFrame,
    config: PerturbationConfig,
    split_runs: list[dict[str, str]],
    spec: PerturbationSpec,
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
        metrics.insert(0, "perturbation", spec.name)
        metrics.insert(1, "perturbation_type", spec.perturbation_type)
        metrics.insert(2, "perturbation_target", spec.target)
        metrics.insert(3, "run_name", run["run_name"])
        metrics.insert(4, "split_strategy", run["strategy"])
        metrics.insert(5, "holdout_protein_id", run["holdout_protein_id"])
        metrics.insert(6, "holdout_family_id", run["holdout_family_id"])
        metric_frames.append(metrics)

        predictions = evaluation.predictions.copy()
        predictions.insert(0, "perturbation", spec.name)
        predictions.insert(1, "perturbation_type", spec.perturbation_type)
        predictions.insert(2, "perturbation_target", spec.target)
        predictions.insert(3, "run_name", run["run_name"])
        predictions.insert(4, "split_strategy", run["strategy"])
        predictions.insert(5, "holdout_protein_id", run["holdout_protein_id"])
        predictions.insert(6, "holdout_family_id", run["holdout_family_id"])
        if config.family_column in split_table.columns:
            family_lookup = split_table[
                ["variant_id", config.family_column]
            ].drop_duplicates()
            predictions = predictions.merge(family_lookup, on="variant_id", how="left")
        prediction_frames.append(predictions)

    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def _quality_table_for_graph(
    graph: nx.MultiDiGraph,
    feature_table: pd.DataFrame,
    config: PerturbationConfig,
    spec: PerturbationSpec,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    explanations: dict[str, dict[str, Any]] = {}
    for _, row in feature_table.iterrows():
        row_dict = row.where(pd.notnull(row), None).to_dict()
        variant_id = str(row_dict["variant_id"])
        explanation = explain_variant(
            graph,
            variant_id,
            row_dict,
            max_contacts=config.max_contacts,
        )
        explanations[variant_id] = explanation
        quality = score_explanation_quality(
            explanation,
            source_diversity_cap=config.source_diversity_cap,
            close_contact_distance=config.close_contact_distance,
            supported_contact_distance=config.supported_contact_distance,
            min_residue_confidence=config.min_residue_confidence,
        )
        quality["confidence_weighted_explanation_quality_score"] = (
            float(quality["explanation_quality_score"])
            * _clip01(float(quality["evidence_confidence_mean"]))
        )
        quality["perturbation"] = spec.name
        quality["perturbation_type"] = spec.perturbation_type
        quality["perturbation_target"] = spec.target
        records.append(quality)
    return pd.DataFrame(records), explanations


def _delta_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for key in keys:
        if key in output.columns:
            output[key] = output[key].fillna("").astype(str)
    return output


def _prediction_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
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
    ].rename(columns={name: f"baseline_{name}" for name in numeric_columns})
    merged = metrics_for_merge.merge(baseline, on=keys, how="left")
    for column in numeric_columns:
        merged[f"delta_{column}"] = merged[column] - merged[f"baseline_{column}"]
    return merged


def _quality_deltas(variant_quality: pd.DataFrame) -> pd.DataFrame:
    keys = ["variant_id", "protein_id", "protein_family"]
    numeric_columns = [
        "residue_confidence",
        "contact_count",
        "domain_evidence_count",
        "go_evidence_count",
        "pathway_evidence_count",
        "disease_evidence_count",
        "evidence_source_count",
        "evidence_confidence_mean",
        "provenance_path_coverage",
        "evidence_source_diversity_score",
        "structural_contact_plausibility_score",
        "domain_support_score",
        "go_pathway_disease_support_score",
        "explanation_quality_score",
        "confidence_weighted_explanation_quality_score",
    ]
    quality_for_merge = _delta_keys(variant_quality, keys)
    baseline = quality_for_merge[quality_for_merge["perturbation"] == "baseline"][
        keys + numeric_columns
    ].rename(columns={name: f"baseline_{name}" for name in numeric_columns})
    merged = quality_for_merge.merge(baseline, on=keys, how="left")
    for column in numeric_columns:
        merged[f"delta_{column}"] = merged[column] - merged[f"baseline_{column}"]
    return merged


def _robustness_summary(
    quality_deltas: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    if group_column not in quality_deltas.columns:
        return pd.DataFrame()
    return (
        quality_deltas.groupby(["perturbation", group_column], dropna=False)
        .agg(
            n_variants=("variant_id", "nunique"),
            mean_delta_explanation_quality_score=(
                "delta_explanation_quality_score",
                "mean",
            ),
            mean_delta_confidence_weighted_quality_score=(
                "delta_confidence_weighted_explanation_quality_score",
                "mean",
            ),
            mean_delta_provenance_path_coverage=(
                "delta_provenance_path_coverage",
                "mean",
            ),
            mean_delta_structural_plausibility_score=(
                "delta_structural_contact_plausibility_score",
                "mean",
            ),
            mean_delta_domain_support_score=("delta_domain_support_score", "mean"),
            mean_delta_go_pathway_disease_support_score=(
                "delta_go_pathway_disease_support_score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(["perturbation", group_column])
    )


def _evidence_counts(row: pd.Series) -> dict[str, Any]:
    return {
        "contact_count": int(row.get("contact_count", 0)),
        "domain_evidence_count": int(row.get("domain_evidence_count", 0)),
        "go_evidence_count": int(row.get("go_evidence_count", 0)),
        "pathway_evidence_count": int(row.get("pathway_evidence_count", 0)),
        "disease_evidence_count": int(row.get("disease_evidence_count", 0)),
        "evidence_source_count": int(row.get("evidence_source_count", 0)),
        "evidence_confidence_mean": float(row.get("evidence_confidence_mean", 0.0)),
    }


def _build_before_after_cards(
    config: PerturbationConfig,
    variant_quality: pd.DataFrame,
    quality_deltas: pd.DataFrame,
    perturbation_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    quality_by_variant = {
        str(row["variant_id"]): row
        for _, row in variant_quality[
            variant_quality["perturbation"] == "baseline"
        ].iterrows()
    }
    deltas = quality_deltas[quality_deltas["perturbation"] != "baseline"].copy()
    summary_by_name = {
        str(item["perturbation"]): item for item in perturbation_summaries
    }
    cards: list[dict[str, Any]] = []
    for variant_id in config.case_study_variants:
        if variant_id not in quality_by_variant:
            raise ValueError(f"Case-study variant not found: {variant_id}")
        baseline = quality_by_variant[variant_id]
        comparisons: list[dict[str, Any]] = []
        variant_rows = deltas[deltas["variant_id"] == variant_id].sort_values(
            "perturbation"
        )
        for _, row in variant_rows.iterrows():
            comparisons.append(
                {
                    "perturbation": row["perturbation"],
                    "perturbation_type": row["perturbation_type"],
                    "perturbation_target": row["perturbation_target"],
                    "edges_removed": summary_by_name[str(row["perturbation"])].get(
                        "edges_removed", 0
                    ),
                    "edges_weakened": summary_by_name[str(row["perturbation"])].get(
                        "edges_weakened", 0
                    ),
                    "after": {
                        "explanation_quality_score": float(
                            row["explanation_quality_score"]
                        ),
                        "confidence_weighted_explanation_quality_score": float(
                            row["confidence_weighted_explanation_quality_score"]
                        ),
                        **_evidence_counts(row),
                    },
                    "delta": {
                        "explanation_quality_score": float(
                            row["delta_explanation_quality_score"]
                        ),
                        "confidence_weighted_explanation_quality_score": float(
                            row[
                                "delta_confidence_weighted_explanation_quality_score"
                            ]
                        ),
                        "provenance_path_coverage": float(
                            row["delta_provenance_path_coverage"]
                        ),
                        "structural_contact_plausibility_score": float(
                            row["delta_structural_contact_plausibility_score"]
                        ),
                        "domain_support_score": float(row["delta_domain_support_score"]),
                        "go_pathway_disease_support_score": float(
                            row["delta_go_pathway_disease_support_score"]
                        ),
                    },
                }
            )
        cards.append(
            _json_safe(
                {
                    "variant_id": variant_id,
                    "protein_id": baseline["protein_id"],
                    "protein_family": baseline.get("protein_family", ""),
                    "baseline": {
                        "explanation_quality_score": float(
                            baseline["explanation_quality_score"]
                        ),
                        "confidence_weighted_explanation_quality_score": float(
                            baseline["confidence_weighted_explanation_quality_score"]
                        ),
                        **_evidence_counts(baseline),
                    },
                    "comparisons": comparisons,
                }
            )
        )
    return cards


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


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


def _write_cards_markdown(cards: list[dict[str, Any]], path: Path) -> None:
    lines = ["# TrustProtKG v0.9 Before/After Perturbation Cards", ""]
    for card in cards:
        baseline = card["baseline"]
        lines.extend(
            [
                f"## {card['variant_id']} ({card['protein_id']})",
                "",
                f"- Protein family: `{card.get('protein_family', '')}`",
                f"- Baseline quality: `{baseline['explanation_quality_score']:.3f}`",
                "- Baseline confidence-weighted quality: "
                f"`{baseline['confidence_weighted_explanation_quality_score']:.3f}`",
                f"- Baseline contacts/domains/context: "
                f"`{baseline['contact_count']}` contacts, "
                f"`{baseline['domain_evidence_count']}` domains, "
                f"`{baseline['go_evidence_count']}` GO, "
                f"`{baseline['pathway_evidence_count']}` pathways, "
                f"`{baseline['disease_evidence_count']}` diseases",
                "",
                "| perturbation | quality | delta_quality | conf_weighted | "
                "delta_conf_weighted | contacts | domains | GO | pathways | diseases |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for comparison in card["comparisons"]:
            after = comparison["after"]
            delta = comparison["delta"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(comparison["perturbation"]),
                        f"{after['explanation_quality_score']:.3f}",
                        f"{delta['explanation_quality_score']:.3f}",
                        f"{after['confidence_weighted_explanation_quality_score']:.3f}",
                        f"{delta['confidence_weighted_explanation_quality_score']:.3f}",
                        str(after["contact_count"]),
                        str(after["domain_evidence_count"]),
                        str(after["go_evidence_count"]),
                        str(after["pathway_evidence_count"]),
                        str(after["disease_evidence_count"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    config: PerturbationConfig,
    metadata: dict[str, Any],
    perturbation_summaries: pd.DataFrame,
    prediction_deltas: pd.DataFrame,
    quality_deltas: pd.DataFrame,
    protein_robustness: pd.DataFrame,
    family_robustness: pd.DataFrame,
    cards: list[dict[str, Any]],
) -> None:
    non_baseline_metrics = prediction_deltas[
        prediction_deltas["perturbation"] != "baseline"
    ]
    metric_summary = (
        non_baseline_metrics.groupby(["perturbation", "feature_set"], dropna=False)
        .agg(
            mean_delta_accuracy=("delta_accuracy", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            mean_delta_auroc=("delta_auroc", "mean"),
        )
        .reset_index()
        .sort_values(["perturbation", "feature_set"])
    )
    quality_summary = (
        quality_deltas[quality_deltas["perturbation"] != "baseline"]
        .groupby("perturbation", dropna=False)
        .agg(
            n_variants=("variant_id", "nunique"),
            mean_delta_explanation_quality_score=(
                "delta_explanation_quality_score",
                "mean",
            ),
            mean_delta_confidence_weighted_quality_score=(
                "delta_confidence_weighted_explanation_quality_score",
                "mean",
            ),
            mean_delta_provenance_path_coverage=(
                "delta_provenance_path_coverage",
                "mean",
            ),
            mean_delta_evidence_confidence_mean=(
                "delta_evidence_confidence_mean",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("perturbation")
    )
    card_summary = pd.DataFrame(
        [
            {
                "variant_id": card["variant_id"],
                "protein_id": card["protein_id"],
                "protein_family": card.get("protein_family", ""),
                "comparisons": len(card["comparisons"]),
            }
            for card in cards
        ]
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
        f"- Split strategy: `{metadata['split_strategy']}`",
        f"- Perturbations: `{metadata['dataset_summary']['perturbations']}`",
        f"- Variants: `{metadata['dataset_summary']['variants']}`",
        "",
        "## Perturbation Plan",
        "",
        _markdown_table(
            perturbation_summaries,
            [
                "perturbation",
                "perturbation_type",
                "target",
                "edges_before",
                "edges_removed",
                "edges_weakened",
                "edges_after",
            ],
        ),
        "",
        "## Prediction Metric Deltas",
        "",
        _markdown_table(
            metric_summary,
            [
                "perturbation",
                "feature_set",
                "mean_delta_accuracy",
                "mean_delta_f1",
                "mean_delta_auroc",
            ],
        ),
        "",
        "## Explanation-Quality Deltas",
        "",
        _markdown_table(
            quality_summary,
            [
                "perturbation",
                "n_variants",
                "mean_delta_explanation_quality_score",
                "mean_delta_confidence_weighted_quality_score",
                "mean_delta_provenance_path_coverage",
                "mean_delta_evidence_confidence_mean",
            ],
        ),
        "",
        "## Protein Robustness Summary",
        "",
        _markdown_table(
            protein_robustness,
            [
                "perturbation",
                "protein_id",
                "n_variants",
                "mean_delta_explanation_quality_score",
                "mean_delta_confidence_weighted_quality_score",
            ],
        ),
        "",
        "## Family Robustness Summary",
        "",
        _markdown_table(
            family_robustness,
            [
                "perturbation",
                "protein_family",
                "n_variants",
                "mean_delta_explanation_quality_score",
                "mean_delta_confidence_weighted_quality_score",
            ],
        ),
        "",
        "## Before/After Case Cards",
        "",
        _markdown_table(card_summary),
        "",
        "## Limitations",
        "",
        "- Perturbations are controlled graph edits on compact local snapshots, not claims about missing biology.",
        "- The logistic baseline is intentionally simple and should be treated as a diagnostic probe.",
        "- Confidence weakening only affects edge confidence fields; node-level residue confidence is unchanged.",
        "- The benchmark remains small, so deltas are best read as reproducibility and explanation-stability signals.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def run_perturbation_experiment(config_path: str | Path) -> PerturbationResult:
    config = PerturbationConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    base_graph = _build_graph(benchmark_config)
    perturbations = build_perturbation_specs(
        config.perturbation_names,
        config.weaken_confidence_scale,
    )

    metric_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    perturbation_summaries: list[dict[str, Any]] = []
    baseline_feature_table: pd.DataFrame | None = None
    split_runs: list[dict[str, str]] | None = None

    for spec in perturbations:
        graph, summary = apply_graph_perturbation(base_graph, spec)
        feature_table = _feature_table_for_graph(graph, benchmark_config, config)
        feature_table.insert(0, "perturbation", spec.name)
        if spec.name == "baseline":
            baseline_feature_table = feature_table.copy()
            split_runs = _split_runs(config, feature_table)
        if split_runs is None:
            raise ValueError("Baseline perturbation must run before other perturbations.")

        eval_feature_table = feature_table.drop(columns=["perturbation"])
        metrics, predictions = _evaluate_feature_table(
            eval_feature_table,
            config,
            split_runs,
            spec,
        )
        quality, _explanations = _quality_table_for_graph(
            graph,
            eval_feature_table,
            config,
            spec,
        )
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
        quality_frames.append(quality)
        feature_frames.append(feature_table)
        perturbation_summaries.append(summary)

    if baseline_feature_table is None or split_runs is None:
        raise ValueError("Perturbation config must include the baseline perturbation.")

    all_features = pd.concat(feature_frames, ignore_index=True)

    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    variant_quality = pd.concat(quality_frames, ignore_index=True)
    prediction_deltas = _prediction_deltas(metrics)
    quality_deltas = _quality_deltas(variant_quality)
    protein_robustness = _robustness_summary(
        quality_deltas,
        group_column=config.protein_column,
    )
    family_robustness = _robustness_summary(
        quality_deltas,
        group_column=config.family_column,
    )
    cards = _build_before_after_cards(
        config,
        variant_quality,
        quality_deltas,
        perturbation_summaries,
    )

    all_features.to_csv(config.feature_table, index=False)
    variant_quality.to_csv(config.variant_quality_scores, index=False)
    metrics.to_csv(config.metrics, index=False)
    predictions.to_csv(config.predictions, index=False)
    prediction_deltas.to_csv(config.prediction_deltas, index=False)
    quality_deltas.to_csv(config.quality_deltas, index=False)
    protein_robustness.to_csv(config.protein_robustness, index=False)
    family_robustness.to_csv(config.family_robustness, index=False)
    _write_jsonl(cards, config.before_after_cards_jsonl)
    _write_cards_markdown(cards, config.before_after_cards_markdown)

    perturbation_summary_frame = pd.DataFrame(perturbation_summaries)
    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "benchmark_config": _relative(config.benchmark_config),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "split_strategy": config.strategy,
        "split_runs": split_runs,
        "feature_sets": config.feature_sets,
        "quality_weights": QUALITY_WEIGHTS,
        "quality_thresholds": {
            "source_diversity_cap": config.source_diversity_cap,
            "close_contact_distance": config.close_contact_distance,
            "supported_contact_distance": config.supported_contact_distance,
            "min_residue_confidence": config.min_residue_confidence,
        },
        "perturbations": perturbation_summaries,
        "graph_summary": {
            "nodes": int(base_graph.number_of_nodes()),
            "edges": int(base_graph.number_of_edges()),
        },
        "dataset_summary": {
            "perturbations": len(perturbations),
            "split_runs": len(split_runs),
            "variants": int(baseline_feature_table["variant_id"].nunique()),
            "proteins": int(baseline_feature_table[config.protein_column].nunique()),
            "families": (
                int(baseline_feature_table[config.family_column].nunique())
                if config.family_column in baseline_feature_table.columns
                else 0
            ),
            "prediction_rows": int(predictions.shape[0]),
            "case_study_cards": len(cards),
        },
        "outputs": {
            "feature_table": _relative(config.feature_table),
            "variant_quality_scores": _relative(config.variant_quality_scores),
            "metrics": _relative(config.metrics),
            "predictions": _relative(config.predictions),
            "prediction_deltas": _relative(config.prediction_deltas),
            "quality_deltas": _relative(config.quality_deltas),
            "protein_robustness": _relative(config.protein_robustness),
            "family_robustness": _relative(config.family_robustness),
            "before_after_cards_jsonl": _relative(config.before_after_cards_jsonl),
            "before_after_cards_markdown": _relative(
                config.before_after_cards_markdown
            ),
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
        perturbation_summary_frame,
        prediction_deltas,
        quality_deltas,
        protein_robustness,
        family_robustness,
        cards,
    )

    return PerturbationResult(
        run_dir=config.run_dir,
        variant_quality_scores=config.variant_quality_scores,
        metrics=config.metrics,
        predictions=config.predictions,
        prediction_deltas=config.prediction_deltas,
        quality_deltas=config.quality_deltas,
        protein_robustness=config.protein_robustness,
        family_robustness=config.family_robustness,
        before_after_cards_jsonl=config.before_after_cards_jsonl,
        before_after_cards_markdown=config.before_after_cards_markdown,
        metadata=config.metadata,
        report=config.report,
        perturbation_count=len(perturbations),
        split_run_count=len(split_runs),
        variant_count=int(baseline_feature_table["variant_id"].nunique()),
        card_count=len(cards),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a TrustProtKG explanation-robustness perturbation experiment."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v0.9_perturbation.yaml",
        help="Path to a TrustProtKG perturbation config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_perturbation_experiment(args.config)
    print("TrustProtKG perturbation experiment complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Perturbations: {result.perturbation_count}")
    print(f"Split runs: {result.split_run_count}")
    print(f"Variants scored: {result.variant_count}")
    print(f"Cards written: {result.card_count}")
    print(f"Metrics: {result.metrics}")
    print(f"Prediction deltas: {result.prediction_deltas}")
    print(f"Quality deltas: {result.quality_deltas}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
