"""Schema models for the heterogeneous TrustProtKG graph."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    PROTEIN = "Protein"
    STRUCTURE_MODEL = "StructureModel"
    RESIDUE = "Residue"
    VARIANT = "Variant"
    DOMAIN = "Domain"
    GO_TERM = "GO_Term"
    DISEASE = "Disease"
    PATHWAY = "Pathway"


class EdgeType(StrEnum):
    HAS_STRUCTURE = "HAS_STRUCTURE"
    HAS_RESIDUE = "HAS_RESIDUE"
    HAS_DOMAIN = "HAS_DOMAIN"
    HAS_FUNCTION = "HAS_FUNCTION"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    LOCATED_AT = "LOCATED_AT"
    SEQUENCE_NEIGHBOR = "SEQUENCE_NEIGHBOR"
    STRUCTURAL_CONTACT = "STRUCTURAL_CONTACT"


class EdgeProvenance(BaseModel):
    source_db: str
    evidence_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: str
    derived_from: str

    def as_edge_attributes(self) -> dict[str, Any]:
        data = self.model_dump()
        data["provenance"] = data.copy()
        return data


class ResidueRecord(BaseModel):
    protein_id: str
    chain_id: str
    residue_index: int
    residue_name: str
    residue_type: str
    ca_coord: list[float] | None = None
    plddt: float | None = None


def protein_node_id(protein_id: str) -> str:
    return f"Protein:{protein_id}"


def structure_node_id(structure_id: str) -> str:
    return f"StructureModel:{structure_id}"


def residue_node_id(protein_id: str, chain_id: str, residue_index: int) -> str:
    return f"Residue:{protein_id}:{chain_id}:{residue_index}"


def variant_node_id(variant_id: str) -> str:
    return f"Variant:{variant_id}"


def domain_node_id(domain_id: str) -> str:
    return f"Domain:{domain_id}"


def go_node_id(go_id: str) -> str:
    return f"GO_Term:{go_id}"


def disease_node_id(disease_id: str) -> str:
    return f"Disease:{disease_id}"


def pathway_node_id(pathway_id: str) -> str:
    return f"Pathway:{pathway_id}"
