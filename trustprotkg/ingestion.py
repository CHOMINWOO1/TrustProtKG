"""Snapshot ingestion and validation for TrustProtKG snapshot packs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trustprotkg.config import load_simple_yaml
from trustprotkg.structure import parse_residues_from_pdb


MANIFEST_COLUMNS = {
    "source_name",
    "source_type",
    "version",
    "snapshot_date",
    "license_note",
    "file_path",
    "checksum",
    "row_count",
}

REQUIRED_SOURCE_COLUMNS = {
    "protein_metadata": {
        "accession",
        "entry_name",
        "gene_symbol",
        "protein_name",
        "organism",
        "source_db",
        "evidence_type",
        "confidence",
    },
    "protein_family_metadata": {
        "protein_family",
        "display_name",
        "description",
        "source_db",
        "evidence_type",
        "confidence",
    },
    "structure_index": {
        "structure_id",
        "accession",
        "entry_name",
        "structure_file",
        "source_db",
        "confidence",
    },
    "variant_assertions": {
        "variation_id",
        "entry_name",
        "chain_id",
        "protein_position",
        "reference_aa",
        "alternate_aa",
        "clinical_significance",
        "split",
        "source_db",
        "evidence_type",
        "confidence",
    },
    "go_annotations": {
        "entry_name",
        "go_id",
        "term_name",
        "namespace",
        "evidence_code",
        "source_db",
        "confidence",
    },
    "domains": {
        "entry_name",
        "interpro_id",
        "domain_name",
        "chain_id",
        "start",
        "end",
        "evidence_code",
        "source_db",
        "confidence",
    },
    "pathways": {
        "entry_name",
        "pathway_id",
        "pathway_name",
        "evidence_code",
        "source_db",
        "confidence",
    },
    "disease_associations": {
        "entry_name",
        "disease_id",
        "disease_name",
        "evidence_code",
        "source_db",
        "confidence",
    },
}

EVIDENCE_ENRICHMENT_COLUMNS = [
    "review_status",
    "assertion_conflict",
    "annotation_date",
    "evidence_strength",
    "calibrated_confidence",
    "source_record_url",
    "citation",
]


@dataclass(frozen=True)
class SnapshotIngestionConfig:
    path: Path
    name: str
    manifest: Path
    output_dir: Path
    benchmark_config: Path
    provenance_report: Path
    validation_summary: Path
    previous_manifest: Path | None
    audit_report: Path | None
    created_at: str
    derived_from: str
    contact_threshold_angstrom: float
    structure_source_db: str
    graph_jsonl: Path
    graphml: Path
    variant_features: Path
    baseline_metrics: Path
    baseline_predictions: Path
    explanations: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "SnapshotIngestionConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        snapshot = raw["snapshot"]
        structure = raw["structure"]
        output = raw["output"]

        def resolve(value: str | Path) -> Path:
            path_value = Path(value)
            if path_value.is_absolute():
                return path_value
            return Path.cwd() / path_value

        return cls(
            path=config_path,
            name=str(snapshot["name"]),
            manifest=resolve(snapshot["manifest"]),
            output_dir=resolve(snapshot["output_dir"]),
            benchmark_config=resolve(snapshot["benchmark_config"]),
            provenance_report=resolve(snapshot["provenance_report"]),
            validation_summary=resolve(snapshot["validation_summary"]),
            previous_manifest=(
                resolve(snapshot["previous_manifest"])
                if "previous_manifest" in snapshot
                else None
            ),
            audit_report=(
                resolve(snapshot["audit_report"]) if "audit_report" in snapshot else None
            ),
            created_at=str(snapshot["created_at"]),
            derived_from=str(snapshot["derived_from"]),
            contact_threshold_angstrom=float(structure["contact_threshold_angstrom"]),
            structure_source_db=str(structure["structure_source_db"]),
            graph_jsonl=resolve(output["graph_jsonl"]),
            graphml=resolve(output["graphml"]),
            variant_features=resolve(output["variant_features"]),
            baseline_metrics=resolve(output["baseline_metrics"]),
            baseline_predictions=resolve(output["baseline_predictions"]),
            explanations=resolve(output["explanations"]),
        )


@dataclass(frozen=True)
class IngestionResult:
    output_dir: Path
    benchmark_config: Path
    provenance_report: Path
    validation_summary: Path
    audit_report: Path | None
    normalized_files: dict[str, Path]
    warnings: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_count(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        return int(pd.read_csv(path).shape[0])
    if path.suffix.lower() == ".pdb":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.startswith(("ATOM", "HETATM")))
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path).fillna("")
    missing = MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return manifest


def verify_manifest(path: str | Path) -> pd.DataFrame:
    """Verify source manifest checksums and record counts."""

    manifest_path = Path(path)
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        file_path = Path(str(row["file_path"]))
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path
        exists = file_path.exists()
        actual_checksum = _sha256(file_path) if exists else ""
        actual_row_count = _record_count(file_path) if exists else -1
        expected_checksum = str(row["checksum"]).lower()
        expected_row_count = int(row["row_count"])
        rows.append(
            {
                **row.to_dict(),
                "resolved_path": str(file_path),
                "exists": exists,
                "actual_checksum": actual_checksum,
                "checksum_ok": exists and actual_checksum == expected_checksum,
                "actual_row_count": actual_row_count,
                "row_count_ok": exists and actual_row_count == expected_row_count,
            }
        )
    return pd.DataFrame(rows)


def _source_file(manifest: pd.DataFrame, source_type: str) -> Path:
    matches = manifest[manifest["source_type"] == source_type]
    if matches.empty:
        raise ValueError(f"Manifest missing source_type: {source_type}")
    file_path = Path(str(matches.iloc[0]["file_path"]))
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path
    return file_path


def _has_source_type(manifest: pd.DataFrame, source_type: str) -> bool:
    return not manifest[manifest["source_type"] == source_type].empty


def _read_source(manifest: pd.DataFrame, source_type: str) -> pd.DataFrame:
    path = _source_file(manifest, source_type)
    frame = pd.read_csv(path).fillna("")
    required = REQUIRED_SOURCE_COLUMNS[source_type]
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source_type} missing columns: {sorted(missing)}")
    return frame


def _label_binary(clinical_significance: str) -> int:
    value = clinical_significance.lower()
    if "pathogenic" in value and "benign" not in value:
        return 1
    if "benign" in value:
        return 0
    raise ValueError(f"Unsupported clinical significance: {clinical_significance}")


def _variant_id(row: pd.Series) -> str:
    ref = str(row["reference_aa"])
    alt = str(row["alternate_aa"])
    position = int(row["protein_position"])
    symbol = str(row["entry_name"]).split("_")[0]
    return f"V_{symbol}_{ref}{position}{alt}"


def _append_optional_evidence_columns(
    normalized: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve source-specific evidence fields when a snapshot provides them."""

    output = normalized.copy()
    for column in EVIDENCE_ENRICHMENT_COLUMNS:
        if column not in raw.columns:
            continue
        if column == "calibrated_confidence":
            output[column] = pd.to_numeric(raw[column], errors="coerce")
        else:
            output[column] = raw[column]
    return output


def normalize_sources(
    config: SnapshotIngestionConfig,
    manifest: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Normalize source-like fixtures into the existing benchmark schema."""

    proteins_raw = _read_source(manifest, "protein_metadata")
    structures_raw = _read_source(manifest, "structure_index")
    variants_raw = _read_source(manifest, "variant_assertions")
    go_raw = _read_source(manifest, "go_annotations")
    domains_raw = _read_source(manifest, "domains")
    pathways_raw = _read_source(manifest, "pathways")
    diseases_raw = _read_source(manifest, "disease_associations")

    protein_columns = {
        "protein_id": proteins_raw["entry_name"],
        "symbol": proteins_raw["gene_symbol"],
        "name": proteins_raw["protein_name"],
        "taxon": proteins_raw["organism"],
    }
    if "protein_family" in proteins_raw.columns:
        protein_columns["protein_family"] = proteins_raw["protein_family"]
    if "family_source" in proteins_raw.columns:
        protein_columns["family_source"] = proteins_raw["family_source"]
    proteins = pd.DataFrame(protein_columns)
    structures = pd.DataFrame(
        {
            "structure_id": structures_raw["structure_id"],
            "protein_id": structures_raw["entry_name"],
            "structure_file": structures_raw["structure_file"],
            "source_db": structures_raw["source_db"],
        }
    )
    variants = pd.DataFrame(
        {
            "variant_id": variants_raw.apply(_variant_id, axis=1),
            "protein_id": variants_raw["entry_name"],
            "chain_id": variants_raw["chain_id"],
            "residue_index": variants_raw["protein_position"].astype(int),
            "reference_aa": variants_raw["reference_aa"],
            "alternate_aa": variants_raw["alternate_aa"],
            "clinical_label": variants_raw["clinical_significance"],
            "label_binary": variants_raw["clinical_significance"].map(_label_binary),
            "split": variants_raw["split"],
            "source_db": variants_raw["source_db"],
            "evidence_type": variants_raw["evidence_type"],
            "confidence": variants_raw["confidence"].astype(float),
        }
    )
    variants = _append_optional_evidence_columns(variants, variants_raw)
    go_annotations = pd.DataFrame(
        {
            "go_id": go_raw["go_id"],
            "name": go_raw["term_name"],
            "namespace": go_raw["namespace"],
            "protein_id": go_raw["entry_name"],
            "source_db": go_raw["source_db"],
            "evidence_type": "GO_" + go_raw["evidence_code"].astype(str),
            "confidence": go_raw["confidence"].astype(float),
        }
    )
    go_annotations = _append_optional_evidence_columns(go_annotations, go_raw)
    domains = pd.DataFrame(
        {
            "domain_id": domains_raw["interpro_id"],
            "protein_id": domains_raw["entry_name"],
            "chain_id": domains_raw["chain_id"],
            "start": domains_raw["start"].astype(int),
            "end": domains_raw["end"].astype(int),
            "name": domains_raw["domain_name"],
            "source_db": domains_raw["source_db"],
            "evidence_type": domains_raw["evidence_code"],
            "confidence": domains_raw["confidence"].astype(float),
        }
    )
    domains = _append_optional_evidence_columns(domains, domains_raw)
    pathways = pd.DataFrame(
        {
            "pathway_id": pathways_raw["pathway_id"],
            "name": pathways_raw["pathway_name"],
            "protein_id": pathways_raw["entry_name"],
            "source_db": pathways_raw["source_db"],
            "evidence_type": "Reactome_" + pathways_raw["evidence_code"].astype(str),
            "confidence": pathways_raw["confidence"].astype(float),
        }
    )
    pathways = _append_optional_evidence_columns(pathways, pathways_raw)
    disease_associations = pd.DataFrame(
        {
            "disease_id": diseases_raw["disease_id"],
            "name": diseases_raw["disease_name"],
            "protein_id": diseases_raw["entry_name"],
            "source_db": diseases_raw["source_db"],
            "evidence_type": diseases_raw["evidence_code"],
            "confidence": diseases_raw["confidence"].astype(float),
        }
    )
    disease_associations = _append_optional_evidence_columns(
        disease_associations,
        diseases_raw,
    )
    frames = {
        "proteins": proteins,
        "structures": structures,
        "variants": variants,
        "go_annotations": go_annotations,
        "domains": domains,
        "pathways": pathways,
        "disease_associations": disease_associations,
    }
    transformations = {
        "protein_metadata": {
            "input_rows": int(proteins_raw.shape[0]),
            "output_rows": int(proteins.shape[0]),
            "dropped_rows": 0,
        },
        "structure_index": {
            "input_rows": int(structures_raw.shape[0]),
            "output_rows": int(structures.shape[0]),
            "dropped_rows": 0,
        },
        "variant_assertions": {
            "input_rows": int(variants_raw.shape[0]),
            "output_rows": int(variants.shape[0]),
            "dropped_rows": 0,
        },
        "go_annotations": {
            "input_rows": int(go_raw.shape[0]),
            "output_rows": int(go_annotations.shape[0]),
            "dropped_rows": 0,
        },
        "domains": {
            "input_rows": int(domains_raw.shape[0]),
            "output_rows": int(domains.shape[0]),
            "dropped_rows": 0,
        },
        "pathways": {
            "input_rows": int(pathways_raw.shape[0]),
            "output_rows": int(pathways.shape[0]),
            "dropped_rows": 0,
        },
        "disease_associations": {
            "input_rows": int(diseases_raw.shape[0]),
            "output_rows": int(disease_associations.shape[0]),
            "dropped_rows": 0,
        },
    }
    if _has_source_type(manifest, "protein_family_metadata"):
        families_raw = _read_source(manifest, "protein_family_metadata")
        frames["protein_families"] = pd.DataFrame(
            {
                "protein_family": families_raw["protein_family"],
                "display_name": families_raw["display_name"],
                "description": families_raw["description"],
                "source_db": families_raw["source_db"],
                "evidence_type": families_raw["evidence_type"],
                "confidence": families_raw["confidence"].astype(float),
            }
        )
        transformations["protein_family_metadata"] = {
            "input_rows": int(families_raw.shape[0]),
            "output_rows": int(frames["protein_families"].shape[0]),
            "dropped_rows": 0,
        }
    return frames, transformations


def _check_provenance_fields(frames: dict[str, pd.DataFrame]) -> list[str]:
    warnings: list[str] = []
    for name in [
        "variants",
        "go_annotations",
        "domains",
        "pathways",
        "disease_associations",
    ]:
        frame = frames[name]
        for column in ["source_db", "evidence_type", "confidence"]:
            if column not in frame.columns:
                warnings.append(f"{name} missing provenance column {column}")
            elif frame[column].astype(str).str.len().eq(0).any():
                warnings.append(f"{name} has empty provenance values in {column}")
        for column in EVIDENCE_ENRICHMENT_COLUMNS:
            if column in frame.columns and frame[column].astype(str).str.len().eq(0).any():
                warnings.append(f"{name} has empty evidence enrichment values in {column}")
        if "calibrated_confidence" in frame.columns:
            calibrated = pd.to_numeric(frame["calibrated_confidence"], errors="coerce")
            if calibrated.isna().any() or not calibrated.between(0.0, 1.0).all():
                warnings.append(f"{name} has invalid calibrated_confidence values")
    return warnings


def _evidence_enrichment_summary(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    evidence_tables = [
        "variants",
        "go_annotations",
        "domains",
        "pathways",
        "disease_associations",
    ]
    enriched_tables = 0
    weak_rows = 0
    strong_rows = 0
    conflicting_rows = 0
    calibrated_rows = 0
    required = {
        "review_status",
        "assertion_conflict",
        "annotation_date",
        "evidence_strength",
        "calibrated_confidence",
    }
    for table_name in evidence_tables:
        frame = frames[table_name]
        if required.issubset(frame.columns):
            enriched_tables += 1
        if "evidence_strength" in frame.columns:
            strength = frame["evidence_strength"].astype(str).str.lower()
            weak_rows += int((strength == "weak").sum())
            strong_rows += int((strength == "strong").sum())
            conflicting_rows += int((strength == "conflicting").sum())
        elif "assertion_conflict" in frame.columns:
            conflict = frame["assertion_conflict"].astype(str).str.lower()
            conflict_flags = (
                conflict.str.contains("conflict", na=False)
                & (conflict != "no_conflict")
            )
            conflicting_rows += int(conflict_flags.sum())
        if "calibrated_confidence" in frame.columns:
            calibrated_rows += int(
                pd.to_numeric(
                    frame["calibrated_confidence"],
                    errors="coerce",
                )
                .notna()
                .sum()
            )
    return {
        "evidence_enriched_tables": enriched_tables,
        "strong_evidence_rows": strong_rows,
        "weak_evidence_rows": weak_rows,
        "conflicting_evidence_rows": conflicting_rows,
        "calibrated_confidence_rows": calibrated_rows,
    }


def validate_normalized_benchmark(
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    warnings: list[str] = []
    variants = frames["variants"]
    structures = frames["structures"]

    residue_index: set[tuple[str, str, int]] = set()
    residues_with_ca: set[tuple[str, str, int]] = set()
    for _, structure in structures.iterrows():
        path = Path(str(structure["structure_file"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        protein_id = str(structure["protein_id"])
        residues = parse_residues_from_pdb(path, protein_id)
        for residue in residues:
            key = (protein_id, residue.chain_id, residue.residue_index)
            residue_index.add(key)
            if residue.ca_coord is not None:
                residues_with_ca.add(key)

    mapped = 0
    ca_covered = 0
    missing_variants: list[str] = []
    for _, variant in variants.iterrows():
        key = (
            str(variant["protein_id"]),
            str(variant["chain_id"]),
            int(variant["residue_index"]),
        )
        if key in residue_index:
            mapped += 1
        else:
            missing_variants.append(str(variant["variant_id"]))
        if key in residues_with_ca:
            ca_covered += 1

    if missing_variants:
        warnings.append(
            "Variants without residue mapping: " + ", ".join(missing_variants)
        )
    labels = variants["label_binary"].astype(int)
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        warnings.append("Variant labels are not balanced across both classes.")
    if "protein_family" in frames["proteins"].columns:
        family_values = frames["proteins"]["protein_family"].astype(str)
        if family_values.str.len().eq(0).any():
            warnings.append("Some proteins are missing protein_family metadata.")
        family_count = int(family_values.nunique())
    else:
        family_count = 0
    warnings.extend(_check_provenance_fields(frames))
    evidence_summary = _evidence_enrichment_summary(frames)

    return {
        "variant_count": int(variants.shape[0]),
        "protein_count": int(frames["proteins"].shape[0]),
        "structure_count": int(structures.shape[0]),
        "protein_family_count": family_count,
        "variant_residue_mapped": mapped,
        "variant_residue_mapping_rate": mapped / max(1, int(variants.shape[0])),
        "variant_ca_covered": ca_covered,
        "variant_ca_coverage_rate": ca_covered / max(1, int(variants.shape[0])),
        "positive_labels": positives,
        "negative_labels": negatives,
        **evidence_summary,
        "warnings": warnings,
    }


def _write_normalized_outputs(
    config: SnapshotIngestionConfig,
    frames: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "proteins": config.output_dir / "proteins.csv",
        "structures": config.output_dir / "structures.csv",
        "variants": config.output_dir / "variants.csv",
        "go_annotations": config.output_dir / "go_annotations.csv",
        "domains": config.output_dir / "domains.csv",
        "pathways": config.output_dir / "pathways.csv",
        "disease_associations": config.output_dir / "disease_associations.csv",
    }
    if "protein_families" in frames:
        paths["protein_families"] = config.output_dir / "protein_families.csv"
    for name, path in paths.items():
        frames[name].to_csv(path, index=False)
    return paths


def _relative(path: Path) -> str:
    return path.resolve().relative_to(Path.cwd().resolve()).as_posix()


def _write_benchmark_config(
    config: SnapshotIngestionConfig,
    normalized_paths: dict[str, Path],
) -> None:
    text = f"""project:
  name: {config.name} normalized benchmark

input:
  proteins: {_relative(normalized_paths["proteins"])}
  variants: {_relative(normalized_paths["variants"])}
  go_annotations: {_relative(normalized_paths["go_annotations"])}
  disease_associations: {_relative(normalized_paths["disease_associations"])}
  domains: {_relative(normalized_paths["domains"])}
  pathways: {_relative(normalized_paths["pathways"])}
  structures: {_relative(normalized_paths["structures"])}

structure:
  contact_threshold_angstrom: {config.contact_threshold_angstrom}
  structure_source_db: {config.structure_source_db}

provenance:
  created_at: {config.created_at}
  derived_from: {config.derived_from}

output:
  graph_jsonl: {_relative(config.graph_jsonl)}
  graphml: {_relative(config.graphml)}
  variant_features: {_relative(config.variant_features)}
  baseline_metrics: {_relative(config.baseline_metrics)}
  baseline_predictions: {_relative(config.baseline_predictions)}
  explanations: {_relative(config.explanations)}
"""
    config.benchmark_config.parent.mkdir(parents=True, exist_ok=True)
    config.benchmark_config.write_text(text, encoding="utf-8")


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
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(lines)


def _write_provenance_report(
    config: SnapshotIngestionConfig,
    manifest_verification: pd.DataFrame,
    transformations: dict[str, Any],
    validation: dict[str, Any],
    normalized_paths: dict[str, Path],
) -> None:
    transform_frame = pd.DataFrame(
        [{"source_type": key, **value} for key, value in transformations.items()]
    )
    normalized_frame = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": _relative(path),
                "rows": int(pd.read_csv(path).shape[0]),
            }
            for name, path in normalized_paths.items()
        ]
    )
    warning_lines = validation["warnings"] or ["No validation warnings."]
    lines = [
        f"# {config.name} Data Provenance Report",
        "",
        "## Source Snapshot Manifest",
        "",
        _markdown_table(
            manifest_verification,
            [
                column
                for column in [
                    "source_name",
                    "source_type",
                    "version",
                    "snapshot_date",
                    "retrieval_date",
                    "source_url",
                    "citation",
                    "license_note",
                    "file_path",
                    "checksum_ok",
                    "row_count_ok",
                    "actual_row_count",
                ]
                if column in manifest_verification.columns
            ],
        ),
        "",
        "## Transformations",
        "",
        _markdown_table(transform_frame),
        "",
        "## Normalized Benchmark Artifacts",
        "",
        _markdown_table(normalized_frame),
        "",
        "## Validation Summary",
        "",
        _markdown_table(pd.DataFrame([{k: v for k, v in validation.items() if k != "warnings"}])),
        "",
        "## Validation Warnings",
        "",
        *[f"- {warning}" for warning in warning_lines],
        "",
    ]
    config.provenance_report.parent.mkdir(parents=True, exist_ok=True)
    config.provenance_report.write_text("\n".join(lines), encoding="utf-8")


def _write_snapshot_audit_report(
    *,
    current_manifest_path: Path,
    previous_manifest_path: Path,
    output_path: Path,
    normalized_paths: dict[str, Path],
    validation: dict[str, Any],
) -> None:
    current = verify_manifest(current_manifest_path)
    previous = verify_manifest(previous_manifest_path)
    current_summary = (
        current.groupby("source_type", as_index=False)
        .agg(current_rows=("actual_row_count", "sum"), current_files=("file_path", "count"))
        .sort_values("source_type")
    )
    previous_summary = (
        previous.groupby("source_type", as_index=False)
        .agg(previous_rows=("actual_row_count", "sum"), previous_files=("file_path", "count"))
        .sort_values("source_type")
    )
    comparison = previous_summary.merge(current_summary, on="source_type", how="outer")
    comparison = comparison.fillna(0)
    comparison["row_delta"] = (
        comparison["current_rows"].astype(int)
        - comparison["previous_rows"].astype(int)
    )
    comparison["file_delta"] = (
        comparison["current_files"].astype(int)
        - comparison["previous_files"].astype(int)
    )
    changed_sources = current.merge(
        previous[["source_type", "checksum"]].rename(
            columns={"checksum": "previous_checksum"}
        ),
        on="source_type",
        how="left",
    )
    changed_sources["checksum_changed"] = (
        changed_sources["checksum"] != changed_sources["previous_checksum"]
    )
    normalized_frame = pd.DataFrame(
        [
            {
                "artifact": name,
                "path": _relative(path),
                "rows": int(pd.read_csv(path).shape[0]),
            }
            for name, path in normalized_paths.items()
        ]
    )
    lines = [
        "# TrustProtKG Snapshot Audit Report",
        "",
        "## Compared Snapshot Packs",
        "",
        f"- Previous manifest: `{_relative(previous_manifest_path)}`",
        f"- Current manifest: `{_relative(current_manifest_path)}`",
        "",
        "## Source-Type Row/File Comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Current Source Checksum Changes",
        "",
        _markdown_table(
            changed_sources,
            [
                column
                for column in [
                    "source_name",
                    "source_type",
                    "version",
                    "file_path",
                    "checksum_changed",
                    "actual_row_count",
                ]
                if column in changed_sources.columns
            ],
        ),
        "",
        "## Current Normalized Benchmark",
        "",
        _markdown_table(normalized_frame),
        "",
        "## Current Validation",
        "",
        _markdown_table(pd.DataFrame([{k: v for k, v in validation.items() if k != "warnings"}])),
        "",
        "## Validation Warnings",
        "",
        *[f"- {warning}" for warning in (validation["warnings"] or ["No validation warnings."])],
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_ingestion(config_path: str | Path) -> IngestionResult:
    config = SnapshotIngestionConfig.from_file(config_path)
    manifest_verification = verify_manifest(config.manifest)
    failed = manifest_verification[
        ~(manifest_verification["checksum_ok"] & manifest_verification["row_count_ok"])
    ]
    if not failed.empty:
        raise ValueError(
            "Manifest verification failed for: "
            + ", ".join(failed["file_path"].astype(str).tolist())
        )

    manifest = load_manifest(config.manifest)
    frames, transformations = normalize_sources(config, manifest)
    validation = validate_normalized_benchmark(frames)
    normalized_paths = _write_normalized_outputs(config, frames)
    _write_benchmark_config(config, normalized_paths)
    _write_provenance_report(
        config,
        manifest_verification,
        transformations,
        validation,
        normalized_paths,
    )
    if config.previous_manifest is not None and config.audit_report is not None:
        _write_snapshot_audit_report(
            current_manifest_path=config.manifest,
            previous_manifest_path=config.previous_manifest,
            output_path=config.audit_report,
            normalized_paths=normalized_paths,
            validation=validation,
        )
    config.validation_summary.parent.mkdir(parents=True, exist_ok=True)
    config.validation_summary.write_text(
        json.dumps(
            {
                "manifest": manifest_verification.to_dict(orient="records"),
                "transformations": transformations,
                "validation": validation,
                "normalized_files": {
                    name: _relative(path) for name, path in normalized_paths.items()
                },
                "benchmark_config": _relative(config.benchmark_config),
                "provenance_report": _relative(config.provenance_report),
                "audit_report": (
                    _relative(config.audit_report) if config.audit_report else None
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return IngestionResult(
        output_dir=config.output_dir,
        benchmark_config=config.benchmark_config,
        provenance_report=config.provenance_report,
        validation_summary=config.validation_summary,
        audit_report=config.audit_report,
        normalized_files=normalized_paths,
        warnings=validation["warnings"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TrustProtKG snapshot ingestion.")
    parser.add_argument(
        "--config",
        default="configs/snapshot_ingestion.yaml",
        help="Path to a TrustProtKG snapshot ingestion config.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    result = run_ingestion(args.config)
    print("TrustProtKG snapshot ingestion complete")
    print(f"Normalized benchmark: {result.output_dir}")
    print(f"Benchmark config: {result.benchmark_config}")
    print(f"Provenance report: {result.provenance_report}")
    print(f"Validation summary: {result.validation_summary}")
    if result.audit_report is not None:
        print(f"Snapshot audit report: {result.audit_report}")
    if result.warnings:
        print("Validation warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
