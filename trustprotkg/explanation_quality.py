"""Explanation-quality evaluation for TrustProtKG v0.7."""

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

from trustprotkg.config import PipelineConfig, load_simple_yaml
from trustprotkg.cross_protein import (
    CrossProteinConfig,
    run_cross_protein_experiment,
)
from trustprotkg.explanations import explain_variant
from trustprotkg.features import extract_variant_feature_table
from trustprotkg.pipeline import _build_graph


QUALITY_WEIGHTS = {
    "provenance_path_coverage": 0.25,
    "evidence_source_diversity_score": 0.20,
    "structural_contact_plausibility_score": 0.20,
    "domain_support_score": 0.15,
    "go_pathway_disease_support_score": 0.20,
}


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


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric):
        return default
    return numeric


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


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
class ExplanationQualityConfig:
    path: Path
    name: str
    benchmark_config: Path
    cross_protein_config: Path
    run_cross_protein: bool
    random_seed: int
    timestamp: str
    max_contacts: int
    source_diversity_cap: int
    close_contact_distance: float
    supported_contact_distance: float
    min_residue_confidence: float
    case_study_variants: list[str]
    run_dir: Path
    variant_quality_scores: Path
    quality_by_context: Path
    quality_by_family: Path
    explanation_cards_jsonl: Path
    explanation_cards_markdown: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "ExplanationQualityConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        quality = raw.get("quality", {})
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
            cross_protein_config=_resolve_path(
                experiment["cross_protein_config"], config_path
            ),
            run_cross_protein=bool(experiment.get("run_cross_protein", True)),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            max_contacts=int(quality.get("max_contacts", 5)),
            source_diversity_cap=int(quality.get("source_diversity_cap", 5)),
            close_contact_distance=float(quality.get("close_contact_distance", 5.0)),
            supported_contact_distance=float(
                quality.get("supported_contact_distance", 8.0)
            ),
            min_residue_confidence=float(quality.get("min_residue_confidence", 70.0)),
            case_study_variants=_split_csv(case_studies.get("variant_ids", "")),
            run_dir=run_dir,
            variant_quality_scores=out_path(
                "variant_quality_scores", "variant_quality_scores.csv"
            ),
            quality_by_context=out_path(
                "quality_by_context", "explanation_quality_by_context.csv"
            ),
            quality_by_family=out_path(
                "quality_by_family", "explanation_quality_by_family.csv"
            ),
            explanation_cards_jsonl=out_path(
                "explanation_cards_jsonl", "explanation_cards.jsonl"
            ),
            explanation_cards_markdown=out_path(
                "explanation_cards_markdown", "explanation_cards.md"
            ),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "explanation_quality_report.md"),
        )


@dataclass(frozen=True)
class ExplanationQualityResult:
    run_dir: Path
    variant_quality_scores: Path
    quality_by_context: Path
    quality_by_family: Path
    explanation_cards_jsonl: Path
    explanation_cards_markdown: Path
    metadata: Path
    report: Path
    variant_count: int
    card_count: int


def _provenance_records(explanation: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for items in explanation["evidence"].values():
        for item in items:
            provenance = item.get("provenance", {})
            if provenance:
                records.append(provenance)
    return records


def _protein_context_counts(explanation: dict[str, Any]) -> dict[str, int]:
    counts = {"go_terms": 0, "pathways": 0, "diseases": 0}
    for item in explanation["evidence"]["protein_context"]:
        edge_type = item.get("edge_type")
        if edge_type == "HAS_FUNCTION":
            counts["go_terms"] += 1
        elif edge_type == "PARTICIPATES_IN":
            counts["pathways"] += 1
        elif edge_type == "ASSOCIATED_WITH":
            counts["diseases"] += 1
    return counts


def score_explanation_quality(
    explanation: dict[str, Any],
    *,
    source_diversity_cap: int = 5,
    close_contact_distance: float = 5.0,
    supported_contact_distance: float = 8.0,
    min_residue_confidence: float = 70.0,
) -> dict[str, Any]:
    """Score one provenance-backed variant explanation."""

    evidence = explanation["evidence"]
    contacts = evidence["structural_contacts"]
    domains = evidence["domain_membership"]
    located_at = evidence["located_at"]
    context_counts = _protein_context_counts(explanation)
    provenance = _provenance_records(explanation)

    path_flags = {
        "variant_residue_path": bool(located_at),
        "structural_contact_path": bool(contacts),
        "domain_path": bool(domains),
        "go_path": context_counts["go_terms"] > 0,
        "pathway_path": context_counts["pathways"] > 0,
        "disease_path": context_counts["diseases"] > 0,
    }
    provenance_path_coverage = _mean([1.0 if value else 0.0 for value in path_flags.values()])

    source_dbs = {
        str(item.get("source_db"))
        for item in provenance
        if item.get("source_db") not in {None, ""}
    }
    evidence_types = {
        str(item.get("evidence_type"))
        for item in provenance
        if item.get("evidence_type") not in {None, ""}
    }
    evidence_confidence_mean = _mean(
        [_as_float(item.get("confidence")) for item in provenance],
        default=0.0,
    )
    evidence_source_diversity_score = _clip01(
        len(source_dbs) / max(1, source_diversity_cap)
    )

    distance_scores: list[float] = []
    contact_confidences: list[float] = []
    for item in contacts:
        distance = _as_float(item.get("distance"), default=supported_contact_distance)
        if distance <= close_contact_distance:
            distance_scores.append(1.0)
        elif distance <= supported_contact_distance:
            span = max(0.001, supported_contact_distance - close_contact_distance)
            distance_scores.append(_clip01(1.0 - (distance - close_contact_distance) / span))
        else:
            distance_scores.append(0.0)
        contact_confidences.append(_as_float(item.get("confidence"), default=0.0))

    residue_confidence = _as_float(explanation["residue"].get("plddt"), default=0.0)
    residue_confidence_score = _clip01(residue_confidence / 100.0)
    confidence_floor_score = 1.0 if residue_confidence >= min_residue_confidence else 0.5
    structural_contact_plausibility_score = (
        _mean(
            [
                _mean(distance_scores),
                _mean(contact_confidences),
                residue_confidence_score,
                confidence_floor_score,
            ]
        )
        if contacts
        else 0.0
    )

    domain_confidences = [
        _as_float(item.get("provenance", {}).get("confidence"), default=0.0)
        for item in domains
    ]
    domain_support_score = _mean(domain_confidences) if domains else 0.0
    go_pathway_disease_support_score = _mean(
        [
            1.0 if context_counts["go_terms"] else 0.0,
            1.0 if context_counts["pathways"] else 0.0,
            1.0 if context_counts["diseases"] else 0.0,
        ]
    )
    score_components = {
        "provenance_path_coverage": provenance_path_coverage,
        "evidence_source_diversity_score": evidence_source_diversity_score,
        "structural_contact_plausibility_score": structural_contact_plausibility_score,
        "domain_support_score": domain_support_score,
        "go_pathway_disease_support_score": go_pathway_disease_support_score,
    }
    explanation_quality_score = sum(
        QUALITY_WEIGHTS[name] * score_components[name] for name in QUALITY_WEIGHTS
    )

    return {
        "variant_id": explanation["variant"].get("variant_id")
        or str(explanation["variant"]["id"]).split(":", 1)[-1],
        "protein_id": explanation["residue"].get("protein_id"),
        "protein_family": explanation.get("features", {}).get("protein_family"),
        "chain_id": explanation["residue"].get("chain_id"),
        "residue_index": int(explanation["residue"].get("residue_index")),
        "residue_confidence": residue_confidence,
        "contact_count": len(contacts),
        "domain_evidence_count": len(domains),
        "go_evidence_count": context_counts["go_terms"],
        "pathway_evidence_count": context_counts["pathways"],
        "disease_evidence_count": context_counts["diseases"],
        "evidence_source_count": len(source_dbs),
        "evidence_type_count": len(evidence_types),
        "evidence_confidence_mean": evidence_confidence_mean,
        "provenance_path_coverage": provenance_path_coverage,
        "evidence_source_diversity_score": evidence_source_diversity_score,
        "structural_contact_plausibility_score": structural_contact_plausibility_score,
        "domain_support_score": domain_support_score,
        "go_pathway_disease_support_score": go_pathway_disease_support_score,
        "explanation_quality_score": explanation_quality_score,
        **path_flags,
    }


def _feature_alignment_score(row: pd.Series) -> float:
    feature_set = str(row["feature_set"])
    components: list[float] = []
    if feature_set in {"structure_only", "structure_plus_kg", "no_domain"}:
        components.append(float(row["structural_contact_plausibility_score"]))
    if feature_set in {"kg_only", "structure_plus_kg", "no_structural_contacts"}:
        components.append(float(row["domain_support_score"]))
        components.append(float(row["go_pathway_disease_support_score"]))
    if feature_set == "no_domain":
        components.append(float(row["go_pathway_disease_support_score"]))
    if feature_set == "no_structural_contacts":
        components.append(float(row["residue_confidence"]) / 100.0)
    if not components:
        components.append(float(row["explanation_quality_score"]))
    return _clip01(_mean(components))


def compare_quality_by_prediction_context(
    predictions: pd.DataFrame,
    variant_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Compare explanation quality across split strategies and feature sets."""

    merge_keys = ["variant_id", "protein_id"]
    if (
        "protein_family" in predictions.columns
        and "protein_family" in variant_quality.columns
        and variant_quality["protein_family"].notna().any()
    ):
        merge_keys.append("protein_family")
    merged = predictions.merge(variant_quality, on=merge_keys, how="left")
    merged["prediction_correct"] = (
        merged["predicted_label"].astype(int) == merged["label_binary"].astype(int)
    )
    merged["feature_evidence_alignment_score"] = merged.apply(
        _feature_alignment_score,
        axis=1,
    )
    group_columns = ["run_name", "split_strategy", "holdout_protein_id", "feature_set"]
    if "holdout_family_id" in merged.columns:
        group_columns.insert(3, "holdout_family_id")
    rows: list[dict[str, Any]] = []
    for keys, group in merged.groupby(group_columns, dropna=False):
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_predictions": int(group.shape[0]),
                "n_variants": int(group["variant_id"].nunique()),
                "mean_prediction_accuracy": float(group["prediction_correct"].mean()),
                "mean_explanation_quality_score": float(
                    group["explanation_quality_score"].mean()
                ),
                "mean_feature_evidence_alignment_score": float(
                    group["feature_evidence_alignment_score"].mean()
                ),
                "mean_provenance_path_coverage": float(
                    group["provenance_path_coverage"].mean()
                ),
                "mean_source_diversity_score": float(
                    group["evidence_source_diversity_score"].mean()
                ),
                "mean_structural_plausibility_score": float(
                    group["structural_contact_plausibility_score"].mean()
                ),
                "mean_domain_support_score": float(group["domain_support_score"].mean()),
                "mean_go_pathway_disease_support_score": float(
                    group["go_pathway_disease_support_score"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _prediction_contexts_for_variant(
    predictions: pd.DataFrame,
    variant_id: str,
) -> list[dict[str, Any]]:
    contexts = predictions[predictions["variant_id"] == variant_id].copy()
    sort_columns = ["split_strategy", "run_name", "feature_set"]
    contexts = contexts.sort_values(sort_columns)
    columns = [
        "run_name",
        "split_strategy",
        "holdout_protein_id",
        "feature_set",
        "clinical_label",
        "label_binary",
        "prob_pathogenic",
        "predicted_label",
    ]
    records: list[dict[str, Any]] = []
    for _, row in contexts[columns].iterrows():
        record = row.to_dict()
        record["prediction_correct"] = int(record["predicted_label"]) == int(
            record["label_binary"]
        )
        records.append(_json_safe(record))
    return records


def _evidence_summary(explanation: dict[str, Any]) -> dict[str, Any]:
    contacts = explanation["evidence"]["structural_contacts"][:3]
    domains = explanation["evidence"]["domain_membership"]
    context = explanation["evidence"]["protein_context"]
    return {
        "nearest_contacts": [
            {
                "residue": item["contact_residue"].get("id"),
                "residue_name": item["contact_residue"].get("residue_name"),
                "distance": item.get("distance"),
                "source_db": item.get("provenance", {}).get("source_db"),
            }
            for item in contacts
        ],
        "domains": [
            {
                "domain_id": item["domain"].get("domain_id"),
                "name": item["domain"].get("name"),
                "source_db": item.get("provenance", {}).get("source_db"),
            }
            for item in domains
        ],
        "go_terms": [
            item["neighbor"]
            for item in context
            if item.get("edge_type") == "HAS_FUNCTION"
        ],
        "pathways": [
            item["neighbor"]
            for item in context
            if item.get("edge_type") == "PARTICIPATES_IN"
        ],
        "diseases": [
            item["neighbor"]
            for item in context
            if item.get("edge_type") == "ASSOCIATED_WITH"
        ],
        "source_dbs": sorted(
            {
                str(item.get("source_db"))
                for item in _provenance_records(explanation)
                if item.get("source_db") not in {None, ""}
            }
        ),
    }


def build_explanation_card(
    explanation: dict[str, Any],
    quality: dict[str, Any],
    prediction_contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a reviewer-facing explanation card data object."""

    feature_snapshot = explanation.get("features", {})
    return _json_safe(
        {
            "variant_id": quality["variant_id"],
            "protein_id": quality["protein_id"],
            "residue": explanation["residue"],
            "clinical_label": feature_snapshot.get("clinical_label"),
            "amino_acid_change": (
                f"{feature_snapshot.get('reference_aa', '')}"
                f"{quality['residue_index']}"
                f"{feature_snapshot.get('alternate_aa', '')}"
            ),
            "quality_metrics": quality,
            "evidence_summary": _evidence_summary(explanation),
            "prediction_contexts": prediction_contexts,
        }
    )


def render_explanation_card_markdown(card: dict[str, Any]) -> str:
    """Render one explanation card as Markdown."""

    quality = card["quality_metrics"]
    evidence = card["evidence_summary"]
    lines = [
        f"## {card['variant_id']} ({card['protein_id']})",
        "",
        f"- Amino-acid change: `{card['amino_acid_change']}`",
        f"- Clinical label: `{card.get('clinical_label')}`",
        f"- Residue confidence: `{quality['residue_confidence']:.1f}`",
        f"- Explanation quality score: `{quality['explanation_quality_score']:.3f}`",
        f"- Provenance path coverage: `{quality['provenance_path_coverage']:.3f}`",
        f"- Evidence source count: `{quality['evidence_source_count']}`",
        "",
        "### Evidence Snapshot",
        "",
        f"- Structural contacts summarized: `{len(evidence['nearest_contacts'])}`",
        f"- Domain evidence items: `{len(evidence['domains'])}`",
        f"- GO terms: `{len(evidence['go_terms'])}`",
        f"- Pathways: `{len(evidence['pathways'])}`",
        f"- Diseases: `{len(evidence['diseases'])}`",
        f"- Source DBs: `{', '.join(evidence['source_dbs'])}`",
        "",
        "### Prediction Contexts",
        "",
        "| run_name | split_strategy | feature_set | prob_pathogenic | predicted | correct |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for context in card["prediction_contexts"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(context["run_name"]),
                    str(context["split_strategy"]),
                    str(context["feature_set"]),
                    f"{float(context['prob_pathogenic']):.3f}",
                    str(context["predicted_label"]),
                    str(context["prediction_correct"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


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


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")


def _write_cards_markdown(cards: list[dict[str, Any]], path: Path) -> None:
    lines = ["# TrustProtKG Explanation Cards", ""]
    for card in cards:
        lines.append(render_explanation_card_markdown(card))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    config: ExplanationQualityConfig,
    metadata: dict[str, Any],
    variant_quality: pd.DataFrame,
    quality_by_context: pd.DataFrame,
    quality_by_family: pd.DataFrame,
    cards: list[dict[str, Any]],
) -> None:
    top_variants = variant_quality.sort_values(
        ["explanation_quality_score", "variant_id"],
        ascending=[False, True],
    )
    lines = [
        f"# {config.name}",
        "",
        "## Run Metadata",
        "",
        f"- Timestamp: `{metadata['timestamp']}`",
        f"- Random seed: `{metadata['random_seed']}`",
        f"- Benchmark config: `{metadata['benchmark_config']}`",
        f"- Cross-protein config: `{metadata['cross_protein_config']}`",
        f"- TrustProtKG version: `{metadata['trustprotkg_version']}`",
        "",
        "## Explanation-Quality Metric Definitions",
        "",
        "- `provenance_path_coverage`: fraction of expected evidence paths present: variant-residue, structural contact, domain, GO, pathway, and disease.",
        "- `evidence_source_diversity_score`: unique provenance source databases divided by the configured cap.",
        "- `structural_contact_plausibility_score`: contact distance support, contact confidence, and residue confidence.",
        "- `domain_support_score`: confidence-backed domain evidence at the residue.",
        "- `go_pathway_disease_support_score`: coverage across protein function, pathway, and disease context.",
        "- `explanation_quality_score`: weighted average of the metrics above.",
        "",
        "## Per-Variant Explanation Quality",
        "",
        _markdown_table(
            top_variants,
            [
                "variant_id",
                "protein_id",
                "explanation_quality_score",
                "provenance_path_coverage",
                "evidence_source_count",
                "structural_contact_plausibility_score",
                "domain_support_score",
                "go_pathway_disease_support_score",
            ],
        ),
        "",
        "## Quality By Split And Feature Context",
        "",
        _markdown_table(
            quality_by_context,
            [
                "run_name",
                "split_strategy",
                "feature_set",
                "n_variants",
                "mean_prediction_accuracy",
                "mean_explanation_quality_score",
                "mean_feature_evidence_alignment_score",
                "mean_provenance_path_coverage",
            ],
        ),
        "",
        "## Quality By Protein Family",
        "",
        (
            _markdown_table(quality_by_family)
            if not quality_by_family.empty
            else "_No protein-family metadata available._"
        ),
        "",
        "## Case Study Cards",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "variant_id": card["variant_id"],
                        "protein_id": card["protein_id"],
                        "quality_score": card["quality_metrics"][
                            "explanation_quality_score"
                        ],
                        "prediction_contexts": len(card["prediction_contexts"]),
                    }
                    for card in cards
                ]
            )
        ),
        "",
        "## Limitations",
        "",
        "- Quality scores are heuristic and intended for prototype comparison, not clinical grading.",
        "- The configured compact snapshot is intentionally small, so aggregate values are diagnostic only.",
        "- Explanation evidence is measured from local graph provenance and compact structure fragments.",
        "- Feature-set alignment scores summarize whether a feature family has matching evidence, not causal explanation fidelity.",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def run_explanation_quality_evaluation(
    config_path: str | Path,
) -> ExplanationQualityResult:
    config = ExplanationQualityConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    cross_config = CrossProteinConfig.from_file(config.cross_protein_config)
    if config.run_cross_protein:
        cross_result = run_cross_protein_experiment(config.cross_protein_config)
        predictions_path = cross_result.predictions
    else:
        predictions_path = cross_config.predictions

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    graph = _build_graph(benchmark_config)
    variants = pd.read_csv(benchmark_config.variants)
    feature_table = extract_variant_feature_table(graph, variants)
    proteins = pd.read_csv(benchmark_config.proteins)
    if "protein_family" in proteins.columns:
        feature_table = feature_table.merge(
            proteins[["protein_id", "protein_family"]],
            on="protein_id",
            how="left",
        )
    predictions = pd.read_csv(predictions_path)

    quality_records: list[dict[str, Any]] = []
    explanations_by_variant: dict[str, dict[str, Any]] = {}
    for _, row in feature_table.iterrows():
        variant_id = str(row["variant_id"])
        explanation = explain_variant(
            graph,
            variant_id,
            row.where(pd.notnull(row), None).to_dict(),
            max_contacts=config.max_contacts,
        )
        explanations_by_variant[variant_id] = explanation
        quality_records.append(
            score_explanation_quality(
                explanation,
                source_diversity_cap=config.source_diversity_cap,
                close_contact_distance=config.close_contact_distance,
                supported_contact_distance=config.supported_contact_distance,
                min_residue_confidence=config.min_residue_confidence,
            )
        )
    variant_quality = pd.DataFrame(quality_records)
    quality_by_context = compare_quality_by_prediction_context(
        predictions,
        variant_quality,
    )
    if (
        "protein_family" in variant_quality.columns
        and variant_quality["protein_family"].notna().any()
    ):
        quality_by_family = (
            variant_quality.groupby("protein_family", dropna=False)
            .agg(
                n_variants=("variant_id", "nunique"),
                mean_explanation_quality_score=(
                    "explanation_quality_score",
                    "mean",
                ),
                mean_provenance_path_coverage=("provenance_path_coverage", "mean"),
                mean_source_diversity_score=(
                    "evidence_source_diversity_score",
                    "mean",
                ),
                mean_structural_plausibility_score=(
                    "structural_contact_plausibility_score",
                    "mean",
                ),
                mean_domain_support_score=("domain_support_score", "mean"),
                mean_go_pathway_disease_support_score=(
                    "go_pathway_disease_support_score",
                    "mean",
                ),
            )
            .reset_index()
        )
    else:
        quality_by_family = pd.DataFrame()

    cards: list[dict[str, Any]] = []
    quality_by_variant = variant_quality.set_index("variant_id")
    for variant_id in config.case_study_variants:
        if variant_id not in explanations_by_variant:
            raise ValueError(f"Case-study variant not found: {variant_id}")
        quality_record = quality_by_variant.loc[variant_id].to_dict()
        quality_record["variant_id"] = variant_id
        cards.append(
            build_explanation_card(
                explanations_by_variant[variant_id],
                quality_record,
                _prediction_contexts_for_variant(predictions, variant_id),
            )
        )

    variant_quality.to_csv(config.variant_quality_scores, index=False)
    quality_by_context.to_csv(config.quality_by_context, index=False)
    quality_by_family.to_csv(config.quality_by_family, index=False)
    _write_jsonl(cards, config.explanation_cards_jsonl)
    _write_cards_markdown(cards, config.explanation_cards_markdown)

    metadata = {
        "timestamp": config.timestamp,
        "config_path": str(config.path.relative_to(Path.cwd())),
        "benchmark_config": str(config.benchmark_config.relative_to(Path.cwd())),
        "cross_protein_config": str(
            config.cross_protein_config.relative_to(Path.cwd())
        ),
        "cross_protein_predictions": str(predictions_path.relative_to(Path.cwd())),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "quality_weights": QUALITY_WEIGHTS,
        "quality_thresholds": {
            "source_diversity_cap": config.source_diversity_cap,
            "close_contact_distance": config.close_contact_distance,
            "supported_contact_distance": config.supported_contact_distance,
            "min_residue_confidence": config.min_residue_confidence,
        },
        "dataset_summary": {
            "variants": int(variant_quality.shape[0]),
            "proteins": int(variant_quality["protein_id"].nunique()),
            "families": (
                int(variant_quality["protein_family"].nunique())
                if (
                    "protein_family" in variant_quality.columns
                    and variant_quality["protein_family"].notna().any()
                )
                else 0
            ),
            "prediction_contexts": int(predictions.shape[0]),
            "case_study_cards": len(cards),
        },
        "case_study_variants": config.case_study_variants,
        "outputs": {
            "variant_quality_scores": str(
                config.variant_quality_scores.relative_to(Path.cwd())
            ),
            "quality_by_context": str(config.quality_by_context.relative_to(Path.cwd())),
            "quality_by_family": str(config.quality_by_family.relative_to(Path.cwd())),
            "explanation_cards_jsonl": str(
                config.explanation_cards_jsonl.relative_to(Path.cwd())
            ),
            "explanation_cards_markdown": str(
                config.explanation_cards_markdown.relative_to(Path.cwd())
            ),
            "report": str(config.report.relative_to(Path.cwd())),
        },
    }
    config.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(
        config,
        metadata,
        variant_quality,
        quality_by_context,
        quality_by_family,
        cards,
    )

    return ExplanationQualityResult(
        run_dir=config.run_dir,
        variant_quality_scores=config.variant_quality_scores,
        quality_by_context=config.quality_by_context,
        quality_by_family=config.quality_by_family,
        explanation_cards_jsonl=config.explanation_cards_jsonl,
        explanation_cards_markdown=config.explanation_cards_markdown,
        metadata=config.metadata,
        report=config.report,
        variant_count=int(variant_quality.shape[0]),
        card_count=len(cards),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a TrustProtKG explanation-quality evaluation."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v0.7_explanation_quality.yaml",
        help="Path to a TrustProtKG explanation-quality config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_explanation_quality_evaluation(args.config)
    print("TrustProtKG explanation-quality evaluation complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Variants scored: {result.variant_count}")
    print(f"Cards written: {result.card_count}")
    print(f"Variant quality: {result.variant_quality_scores}")
    print(f"Quality by context: {result.quality_by_context}")
    print(f"Quality by family: {result.quality_by_family}")
    print(f"Cards JSONL: {result.explanation_cards_jsonl}")
    print(f"Cards Markdown: {result.explanation_cards_markdown}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
