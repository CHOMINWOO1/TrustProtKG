"""Snapshot diagnostics for TrustProtKG evidence-aware benchmark packs."""

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


PROVENANCE_TABLES = [
    "variants",
    "go_annotations",
    "domains",
    "pathways",
    "disease_associations",
]

EVIDENCE_DIAGNOSTIC_COLUMNS = {
    "review_status",
    "assertion_conflict",
    "annotation_date",
    "evidence_strength",
    "calibrated_confidence",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class SnapshotDiagnosticsConfig:
    path: Path
    name: str
    benchmark_config: Path
    ingestion_validation_summary: Path
    random_seed: int
    timestamp: str
    protein_column: str
    family_column: str
    split_column: str
    label_column: str
    train_value: str
    test_value: str
    run_dir: Path
    label_diagnostics: Path
    leakage_diagnostics: Path
    provenance_diagnostics: Path
    evidence_strength_diagnostics: Path
    curation_summary: Path
    metadata: Path
    report: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "SnapshotDiagnosticsConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        experiment = raw.get("experiment", {})
        inputs = raw.get("input", {})
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
            benchmark_config=_resolve_path(inputs["benchmark_config"], config_path),
            ingestion_validation_summary=_resolve_path(
                inputs["ingestion_validation_summary"],
                config_path,
            ),
            random_seed=int(experiment.get("random_seed", 0)),
            timestamp=str(experiment.get("timestamp", "")),
            protein_column=str(split.get("protein_column", "protein_id")),
            family_column=str(split.get("family_column", "protein_family")),
            split_column=str(split.get("column", "split")),
            label_column=str(label.get("column", "label_binary")),
            train_value=str(split.get("train_value", "train")),
            test_value=str(split.get("test_value", "test")),
            run_dir=run_dir,
            label_diagnostics=out_path("label_diagnostics", "label_diagnostics.csv"),
            leakage_diagnostics=out_path(
                "leakage_diagnostics",
                "split_leakage_diagnostics.csv",
            ),
            provenance_diagnostics=out_path(
                "provenance_diagnostics",
                "provenance_diagnostics.csv",
            ),
            evidence_strength_diagnostics=out_path(
                "evidence_strength_diagnostics",
                "evidence_strength_diagnostics.csv",
            ),
            curation_summary=out_path("curation_summary", "curation_summary.csv"),
            metadata=out_path("metadata", "metadata.json"),
            report=out_path("report", "snapshot_diagnostics_report.md"),
        )


@dataclass(frozen=True)
class SnapshotDiagnosticsResult:
    run_dir: Path
    label_diagnostics: Path
    leakage_diagnostics: Path
    provenance_diagnostics: Path
    evidence_strength_diagnostics: Path
    curation_summary: Path
    metadata: Path
    report: Path
    variant_count: int
    protein_count: int
    family_count: int


def _load_tables(benchmark_config: PipelineConfig) -> dict[str, pd.DataFrame]:
    return {
        "proteins": pd.read_csv(benchmark_config.proteins),
        "variants": pd.read_csv(benchmark_config.variants),
        "go_annotations": pd.read_csv(benchmark_config.go_annotations),
        "domains": pd.read_csv(benchmark_config.domains),
        "pathways": pd.read_csv(benchmark_config.pathways),
        "disease_associations": pd.read_csv(benchmark_config.disease_associations),
        "structures": pd.read_csv(benchmark_config.structures)
        if benchmark_config.structures is not None
        else pd.DataFrame(),
    }


def _variant_context(config: SnapshotDiagnosticsConfig, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    variants = tables["variants"].copy()
    proteins = tables["proteins"][[config.protein_column, config.family_column]]
    return variants.merge(proteins, on=config.protein_column, how="left")


def _label_row(frame: pd.DataFrame, *, context_type: str, context_id: str) -> dict[str, Any]:
    labels = frame["label_binary"].astype(int)
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    total = int(frame.shape[0])
    return {
        "context_type": context_type,
        "context_id": context_id,
        "n_variants": total,
        "positive_labels": positives,
        "negative_labels": negatives,
        "positive_fraction": positives / max(1, total),
        "minority_fraction": min(positives, negatives) / max(1, total),
        "has_both_classes": positives > 0 and negatives > 0,
    }


def _label_diagnostics(config: SnapshotDiagnosticsConfig, variants: pd.DataFrame) -> pd.DataFrame:
    rows = [_label_row(variants, context_type="overall", context_id="all")]
    for split_value, group in variants.groupby(config.split_column, dropna=False):
        rows.append(_label_row(group, context_type="split", context_id=str(split_value)))
    for protein_id, group in variants.groupby(config.protein_column, dropna=False):
        rows.append(_label_row(group, context_type="protein", context_id=str(protein_id)))
    for family_id, group in variants.groupby(config.family_column, dropna=False):
        rows.append(_label_row(group, context_type="family", context_id=str(family_id)))
    for (split_value, family_id), group in variants.groupby(
        [config.split_column, config.family_column],
        dropna=False,
    ):
        rows.append(
            _label_row(
                group,
                context_type="split_family",
                context_id=f"{split_value}:{family_id}",
            )
        )
    return pd.DataFrame(rows)


def _status(ok: bool, warning: bool = False) -> str:
    if ok:
        return "pass"
    return "warning" if warning else "fail"


def _split_leakage_diagnostics(
    config: SnapshotDiagnosticsConfig,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    train = variants[variants[config.split_column] == config.train_value]
    test = variants[variants[config.split_column] == config.test_value]
    rows: list[dict[str, Any]] = []

    for column, label in [
        ("variant_id", "variant_id_overlap_between_train_test"),
        (config.protein_column, "protein_overlap_between_train_test"),
        (config.family_column, "family_overlap_between_train_test"),
    ]:
        overlap = sorted(set(train[column].astype(str)) & set(test[column].astype(str)))
        expected_for_random_split = column in {config.protein_column, config.family_column}
        rows.append(
            {
                "check_name": label,
                "context_type": "random_split",
                "context_id": "configured_train_test",
                "status": _status(
                    len(overlap) == 0,
                    warning=expected_for_random_split and len(overlap) > 0,
                ),
                "n_items": len(overlap),
                "detail": ",".join(overlap),
            }
        )

    for split_value, group in variants.groupby(config.split_column, dropna=False):
        labels = group[config.label_column].astype(int)
        positives = int(labels.sum())
        negatives = int((labels == 0).sum())
        rows.append(
            {
                "check_name": "split_has_both_label_classes",
                "context_type": "split",
                "context_id": str(split_value),
                "status": _status(positives > 0 and negatives > 0),
                "n_items": int(group.shape[0]),
                "detail": f"positive={positives};negative={negatives}",
            }
        )

    for protein_id, holdout in variants.groupby(config.protein_column, dropna=False):
        train_part = variants[variants[config.protein_column] != protein_id]
        labels = holdout[config.label_column].astype(int)
        rows.append(
            {
                "check_name": "leave_one_protein_out_holdout",
                "context_type": "protein",
                "context_id": str(protein_id),
                "status": _status(labels.nunique() == 2),
                "n_items": int(holdout.shape[0]),
                "detail": (
                    f"holdout_positive={int(labels.sum())};"
                    f"holdout_negative={int((labels == 0).sum())};"
                    f"train_proteins={train_part[config.protein_column].nunique()}"
                ),
            }
        )

    for family_id, holdout in variants.groupby(config.family_column, dropna=False):
        train_part = variants[variants[config.family_column] != family_id]
        labels = holdout[config.label_column].astype(int)
        rows.append(
            {
                "check_name": "leave_one_family_out_holdout",
                "context_type": "family",
                "context_id": str(family_id),
                "status": _status(labels.nunique() == 2),
                "n_items": int(holdout.shape[0]),
                "detail": (
                    f"holdout_positive={int(labels.sum())};"
                    f"holdout_negative={int((labels == 0).sum())};"
                    f"train_families={train_part[config.family_column].nunique()}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _provenance_diagnostics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_name in PROVENANCE_TABLES:
        frame = tables[table_name]
        for column in ["source_db", "evidence_type", "confidence"]:
            present = column in frame.columns
            completeness = (
                float(frame[column].astype(str).str.len().gt(0).mean())
                if present
                else 0.0
            )
        for column in sorted(EVIDENCE_DIAGNOSTIC_COLUMNS):
            if column not in frame.columns:
                continue
            completeness = float(frame[column].astype(str).str.len().gt(0).mean())
            rows.append(
                {
                    "table_name": table_name,
                    "check_name": f"{column}_completeness",
                    "n_rows": int(frame.shape[0]),
                    "value": completeness,
                    "status": "pass" if math.isclose(completeness, 1.0) else "fail",
                }
            )
            rows.append(
                {
                    "table_name": table_name,
                    "check_name": f"{column}_completeness",
                    "n_rows": int(frame.shape[0]),
                    "value": completeness,
                    "status": "pass" if math.isclose(completeness, 1.0) else "fail",
                }
            )
        if "source_db" in frame.columns:
            rows.append(
                {
                    "table_name": table_name,
                    "check_name": "source_db_count",
                    "n_rows": int(frame.shape[0]),
                    "value": int(frame["source_db"].nunique()),
                    "status": "pass",
                }
            )
        if "confidence" in frame.columns:
            confidence = pd.to_numeric(frame["confidence"], errors="coerce")
            rows.extend(
                [
                    {
                        "table_name": table_name,
                        "check_name": "confidence_min",
                        "n_rows": int(frame.shape[0]),
                        "value": float(confidence.min()),
                        "status": "pass" if confidence.min() >= 0.0 else "fail",
                    },
                    {
                        "table_name": table_name,
                        "check_name": "confidence_mean",
                        "n_rows": int(frame.shape[0]),
                        "value": float(confidence.mean()),
                        "status": "pass",
                    },
                ]
            )
    return pd.DataFrame(rows)


def _evidence_strength_diagnostics(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output_columns = [
        "table_name",
        "source_db",
        "n_rows",
        "strong_rows",
        "moderate_rows",
        "weak_rows",
        "conflicting_rows",
        "assertion_conflict_rows",
        "confidence_mean",
        "calibrated_confidence_mean",
        "mean_calibration_delta",
        "old_annotation_rows",
        "weak_or_conflicting_fraction",
        "status",
    ]
    rows: list[dict[str, Any]] = []
    for table_name in PROVENANCE_TABLES:
        frame = tables[table_name]
        missing = EVIDENCE_DIAGNOSTIC_COLUMNS.difference(frame.columns)
        if missing:
            continue

        evidence = frame.copy()
        evidence["confidence"] = pd.to_numeric(evidence["confidence"], errors="coerce")
        evidence["calibrated_confidence"] = pd.to_numeric(
            evidence["calibrated_confidence"],
            errors="coerce",
        )
        evidence["evidence_strength"] = (
            evidence["evidence_strength"].astype(str).str.lower()
        )
        evidence["assertion_conflict"] = (
            evidence["assertion_conflict"].astype(str).str.lower()
        )
        evidence["annotation_date"] = pd.to_datetime(
            evidence["annotation_date"],
            errors="coerce",
        )
        source_column = "source_db" if "source_db" in evidence.columns else "table_name"
        for source_db, group in evidence.groupby(source_column, dropna=False):
            strength = group["evidence_strength"]
            conflict = group["assertion_conflict"]
            conflict_rows = int(
                (
                    conflict.str.contains("conflict", na=False)
                    & (conflict != "no_conflict")
                ).sum()
            )
            weak_rows = int((strength == "weak").sum())
            old_annotation_rows = int(
                (group["annotation_date"] < pd.Timestamp("2023-01-01")).sum()
            )
            row_count = int(group.shape[0])
            weak_or_conflicting_fraction = (
                (weak_rows + conflict_rows) / max(1, row_count)
            )
            rows.append(
                {
                    "table_name": table_name,
                    "source_db": str(source_db),
                    "n_rows": row_count,
                    "strong_rows": int((strength == "strong").sum()),
                    "moderate_rows": int((strength == "moderate").sum()),
                    "weak_rows": weak_rows,
                    "conflicting_rows": int((strength == "conflicting").sum()),
                    "assertion_conflict_rows": conflict_rows,
                    "confidence_mean": float(group["confidence"].mean()),
                    "calibrated_confidence_mean": float(
                        group["calibrated_confidence"].mean()
                    ),
                    "mean_calibration_delta": float(
                        (
                            group["calibrated_confidence"]
                            - group["confidence"]
                        ).mean()
                    ),
                    "old_annotation_rows": old_annotation_rows,
                    "weak_or_conflicting_fraction": weak_or_conflicting_fraction,
                    "status": (
                        "warning"
                        if conflict_rows > 0 or weak_or_conflicting_fraction > 0.25
                        else "pass"
                    ),
                }
            )
    return pd.DataFrame(rows, columns=output_columns)


def _curation_summary(
    validation_summary: dict[str, Any],
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    validation = validation_summary["validation"]
    rows = [
        ("dataset", "variant_count", validation["variant_count"]),
        ("dataset", "protein_count", validation["protein_count"]),
        ("dataset", "protein_family_count", validation["protein_family_count"]),
        ("dataset", "positive_labels", validation["positive_labels"]),
        ("dataset", "negative_labels", validation["negative_labels"]),
        ("structure", "variant_residue_mapping_rate", validation["variant_residue_mapping_rate"]),
        ("structure", "variant_ca_coverage_rate", validation["variant_ca_coverage_rate"]),
        ("manifest", "records", len(validation_summary["manifest"])),
        (
            "manifest",
            "checksum_failures",
            sum(not row["checksum_ok"] for row in validation_summary["manifest"]),
        ),
        (
            "manifest",
            "row_count_failures",
            sum(not row["row_count_ok"] for row in validation_summary["manifest"]),
        ),
        ("annotations", "go_rows", int(tables["go_annotations"].shape[0])),
        ("annotations", "domain_rows", int(tables["domains"].shape[0])),
        ("annotations", "pathway_rows", int(tables["pathways"].shape[0])),
        ("annotations", "disease_rows", int(tables["disease_associations"].shape[0])),
    ]
    for source_type, transform in validation_summary["transformations"].items():
        rows.append((f"transform:{source_type}", "input_rows", transform["input_rows"]))
        rows.append((f"transform:{source_type}", "output_rows", transform["output_rows"]))
        rows.append((f"transform:{source_type}", "dropped_rows", transform["dropped_rows"]))
    return pd.DataFrame(rows, columns=["section", "measure", "value"])


def _round_table(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].round(digits)
    return output


def _markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        existing = [column for column in columns if column in frame.columns]
        frame = frame[existing]
    frame = _round_table(frame)
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def _write_report(
    config: SnapshotDiagnosticsConfig,
    *,
    metadata: dict[str, Any],
    label_diagnostics: pd.DataFrame,
    leakage_diagnostics: pd.DataFrame,
    provenance_diagnostics: pd.DataFrame,
    evidence_strength_diagnostics: pd.DataFrame,
    curation_summary: pd.DataFrame,
) -> None:
    leakage_warnings = leakage_diagnostics[
        leakage_diagnostics["status"].isin(["warning", "fail"])
    ]
    provenance_failures = provenance_diagnostics[
        provenance_diagnostics["status"] == "fail"
    ]
    evidence_warnings = (
        evidence_strength_diagnostics[
            evidence_strength_diagnostics["status"] == "warning"
        ]
        if not evidence_strength_diagnostics.empty
        else pd.DataFrame()
    )
    lines = [
        f"# {config.name}",
        "",
        "## Dataset Summary",
        "",
        f"- Variants: `{metadata['dataset_summary']['variants']}`",
        f"- Proteins: `{metadata['dataset_summary']['proteins']}`",
        f"- Families: `{metadata['dataset_summary']['families']}`",
        f"- TrustProtKG version: `{metadata['trustprotkg_version']}`",
        "",
        "## Label Balance Diagnostics",
        "",
        _markdown_table(
            label_diagnostics,
            [
                "context_type",
                "context_id",
                "n_variants",
                "positive_labels",
                "negative_labels",
                "minority_fraction",
                "has_both_classes",
            ],
        ),
        "",
        "## Leakage-Aware Split Diagnostics",
        "",
        _markdown_table(
            leakage_diagnostics,
            ["check_name", "context_type", "context_id", "status", "n_items", "detail"],
        ),
        "",
        "## Provenance Completeness Diagnostics",
        "",
        _markdown_table(
            provenance_diagnostics,
            ["table_name", "check_name", "n_rows", "value", "status"],
        ),
        "",
        "## Evidence Strength And Conflict Diagnostics",
        "",
        _markdown_table(
            evidence_strength_diagnostics,
            [
                "table_name",
                "source_db",
                "n_rows",
                "strong_rows",
                "moderate_rows",
                "weak_rows",
                "conflicting_rows",
                "assertion_conflict_rows",
                "mean_calibration_delta",
                "status",
            ],
        ),
        "",
        "## Curation And Dropped-Row Summary",
        "",
        _markdown_table(curation_summary),
        "",
        "## Warnings To Consider In A Paper",
        "",
        *[
            f"- `{row['check_name']}` / `{row['context_id']}`: {row['detail']}"
            for _, row in leakage_warnings.iterrows()
        ],
        *[
            f"- Provenance failure in `{row['table_name']}`: `{row['check_name']}`"
            for _, row in provenance_failures.iterrows()
        ],
        *[
            "- Evidence warning in "
            f"`{row['table_name']}` / `{row['source_db']}`: "
            f"weak/conflicting fraction `{row['weak_or_conflicting_fraction']:.3f}`"
            for _, row in evidence_warnings.iterrows()
        ],
        *(
            ["- No provenance failures were detected."]
            if provenance_failures.empty
            else []
        ),
        "",
        "## Outputs",
        "",
        f"- Label diagnostics: `{_relative(config.label_diagnostics)}`",
        f"- Split leakage diagnostics: `{_relative(config.leakage_diagnostics)}`",
        f"- Provenance diagnostics: `{_relative(config.provenance_diagnostics)}`",
        f"- Evidence strength diagnostics: `{_relative(config.evidence_strength_diagnostics)}`",
        f"- Curation summary: `{_relative(config.curation_summary)}`",
        "",
    ]
    config.report.parent.mkdir(parents=True, exist_ok=True)
    config.report.write_text("\n".join(lines), encoding="utf-8")


def run_snapshot_diagnostics(config_path: str | Path) -> SnapshotDiagnosticsResult:
    config = SnapshotDiagnosticsConfig.from_file(config_path)
    np.random.seed(config.random_seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_config = PipelineConfig.from_file(config.benchmark_config)
    validation_summary = _read_json(config.ingestion_validation_summary)
    tables = _load_tables(benchmark_config)
    variants = _variant_context(config, tables)

    label_diagnostics = _label_diagnostics(config, variants)
    leakage_diagnostics = _split_leakage_diagnostics(config, variants)
    provenance_diagnostics = _provenance_diagnostics(tables)
    evidence_strength_diagnostics = _evidence_strength_diagnostics(tables)
    curation_summary = _curation_summary(validation_summary, tables)

    label_diagnostics.to_csv(config.label_diagnostics, index=False)
    leakage_diagnostics.to_csv(config.leakage_diagnostics, index=False)
    provenance_diagnostics.to_csv(config.provenance_diagnostics, index=False)
    evidence_strength_diagnostics.to_csv(
        config.evidence_strength_diagnostics,
        index=False,
    )
    curation_summary.to_csv(config.curation_summary, index=False)

    evidence_summary = {
        "evidence_strength_tables": int(
            evidence_strength_diagnostics["table_name"].nunique()
        )
        if not evidence_strength_diagnostics.empty
        else 0,
        "strong_evidence_rows": int(
            evidence_strength_diagnostics["strong_rows"].sum()
        )
        if not evidence_strength_diagnostics.empty
        else 0,
        "weak_evidence_rows": int(evidence_strength_diagnostics["weak_rows"].sum())
        if not evidence_strength_diagnostics.empty
        else 0,
        "conflicting_evidence_rows": int(
            evidence_strength_diagnostics["conflicting_rows"].sum()
        )
        if not evidence_strength_diagnostics.empty
        else 0,
        "assertion_conflict_rows": int(
            evidence_strength_diagnostics["assertion_conflict_rows"].sum()
        )
        if not evidence_strength_diagnostics.empty
        else 0,
        "calibrated_evidence_rows": int(evidence_strength_diagnostics["n_rows"].sum())
        if not evidence_strength_diagnostics.empty
        else 0,
        "evidence_strength_warnings": int(
            (evidence_strength_diagnostics["status"] == "warning").sum()
        )
        if not evidence_strength_diagnostics.empty
        else 0,
    }

    metadata = {
        "timestamp": config.timestamp,
        "config_path": _relative(config.path),
        "benchmark_config": _relative(config.benchmark_config),
        "ingestion_validation_summary": _relative(config.ingestion_validation_summary),
        "trustprotkg_version": _project_version(),
        "random_seed": config.random_seed,
        "dataset_summary": {
            "variants": int(variants["variant_id"].nunique()),
            "proteins": int(variants[config.protein_column].nunique()),
            "families": int(variants[config.family_column].nunique()),
            "positive_labels": int(variants[config.label_column].astype(int).sum()),
            "negative_labels": int((variants[config.label_column].astype(int) == 0).sum()),
            "manifest_records": len(validation_summary["manifest"]),
            "provenance_failures": int(
                (provenance_diagnostics["status"] == "fail").sum()
            ),
            "leakage_warnings": int(
                leakage_diagnostics["status"].isin(["warning", "fail"]).sum()
            ),
            **evidence_summary,
        },
        "outputs": {
            "label_diagnostics": _relative(config.label_diagnostics),
            "leakage_diagnostics": _relative(config.leakage_diagnostics),
            "provenance_diagnostics": _relative(config.provenance_diagnostics),
            "evidence_strength_diagnostics": _relative(
                config.evidence_strength_diagnostics
            ),
            "curation_summary": _relative(config.curation_summary),
            "report": _relative(config.report),
        },
    }
    config.metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(
        config,
        metadata=metadata,
        label_diagnostics=label_diagnostics,
        leakage_diagnostics=leakage_diagnostics,
        provenance_diagnostics=provenance_diagnostics,
        evidence_strength_diagnostics=evidence_strength_diagnostics,
        curation_summary=curation_summary,
    )
    return SnapshotDiagnosticsResult(
        run_dir=config.run_dir,
        label_diagnostics=config.label_diagnostics,
        leakage_diagnostics=config.leakage_diagnostics,
        provenance_diagnostics=config.provenance_diagnostics,
        evidence_strength_diagnostics=config.evidence_strength_diagnostics,
        curation_summary=config.curation_summary,
        metadata=config.metadata,
        report=config.report,
        variant_count=int(variants["variant_id"].nunique()),
        protein_count=int(variants[config.protein_column].nunique()),
        family_count=int(variants[config.family_column].nunique()),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TrustProtKG expanded snapshot diagnostics."
    )
    parser.add_argument(
        "--config",
        default="experiments/configs/v1.4_snapshot_diagnostics.yaml",
        help="Path to a TrustProtKG snapshot diagnostics config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_snapshot_diagnostics(args.config)
    print("TrustProtKG snapshot diagnostics complete")
    print(f"Run dir: {result.run_dir}")
    print(f"Variants: {result.variant_count}")
    print(f"Proteins: {result.protein_count}")
    print(f"Families: {result.family_count}")
    print(f"Label diagnostics: {result.label_diagnostics}")
    print(f"Leakage diagnostics: {result.leakage_diagnostics}")
    print(f"Provenance diagnostics: {result.provenance_diagnostics}")
    print(f"Evidence strength diagnostics: {result.evidence_strength_diagnostics}")
    print(f"Report: {result.report}")


if __name__ == "__main__":
    main()
