"""Configuration loading for TrustProtKG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the small nested mapping style used by configs/demo.yaml.

    This intentionally avoids a PyYAML dependency. It supports comments, blank
    lines, scalar values, and indentation-based dictionaries.
    """

    config_path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if raw_value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def _resolve_path(path_value: str | Path, config_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or not (config_path.parent / path).exists():
        return cwd_path
    return config_path.parent / path


@dataclass(frozen=True)
class PipelineConfig:
    """Typed configuration used by demo and benchmark pipelines."""

    config_path: Path
    protein_id: str
    structure_id: str | None
    proteins: Path
    variants: Path
    go_annotations: Path
    disease_associations: Path
    domains: Path
    pathways: Path
    structure_file: Path | None
    structures: Path | None
    contact_threshold_angstrom: float
    structure_source_db: str
    created_at: str
    derived_from: str
    graph_jsonl: Path
    graphml: Path
    variant_features: Path
    baseline_metrics: Path | None = None
    baseline_predictions: Path | None = None
    explanations: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).resolve()
        raw = load_simple_yaml(config_path)
        project = raw.get("project", {})
        inputs = raw.get("input", {})
        structure = raw.get("structure", {})
        provenance = raw.get("provenance", {})
        output = raw.get("output", {})

        required_sections = {
            "project": project,
            "input": inputs,
            "structure": structure,
            "provenance": provenance,
            "output": output,
        }
        missing_sections = [name for name, value in required_sections.items() if not value]
        if missing_sections:
            raise ValueError(f"Missing config sections: {', '.join(missing_sections)}")

        return cls(
            config_path=config_path,
            protein_id=str(project.get("protein_id", "")),
            structure_id=(
                str(project["structure_id"]) if "structure_id" in project else None
            ),
            proteins=_resolve_path(inputs["proteins"], config_path),
            variants=_resolve_path(inputs["variants"], config_path),
            go_annotations=_resolve_path(inputs["go_annotations"], config_path),
            disease_associations=_resolve_path(
                inputs["disease_associations"], config_path
            ),
            domains=_resolve_path(inputs["domains"], config_path),
            pathways=_resolve_path(inputs["pathways"], config_path),
            structure_file=(
                _resolve_path(inputs["structure_file"], config_path)
                if "structure_file" in inputs
                else None
            ),
            structures=(
                _resolve_path(inputs["structures"], config_path)
                if "structures" in inputs
                else None
            ),
            contact_threshold_angstrom=float(structure["contact_threshold_angstrom"]),
            structure_source_db=str(structure["structure_source_db"]),
            created_at=str(provenance["created_at"]),
            derived_from=str(provenance["derived_from"]),
            graph_jsonl=_resolve_path(output["graph_jsonl"], config_path),
            graphml=_resolve_path(output["graphml"], config_path),
            variant_features=_resolve_path(output["variant_features"], config_path),
            baseline_metrics=(
                _resolve_path(output["baseline_metrics"], config_path)
                if "baseline_metrics" in output
                else None
            ),
            baseline_predictions=(
                _resolve_path(output["baseline_predictions"], config_path)
                if "baseline_predictions" in output
                else None
            ),
            explanations=(
                _resolve_path(output["explanations"], config_path)
                if "explanations" in output
                else None
            ),
        )
