"""Provenance-backed explanation exports for variant feature rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from trustprotkg.features import _incident_edges, _located_residue_for_variant, _other_node
from trustprotkg.models import EdgeType, NodeType, protein_node_id, variant_node_id


def _provenance(attrs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_db": attrs.get("source_db"),
        "evidence_type": attrs.get("evidence_type"),
        "confidence": attrs.get("confidence"),
        "created_at": attrs.get("created_at"),
        "derived_from": attrs.get("derived_from"),
    }


def _node_summary(graph: nx.MultiDiGraph, node_id: str) -> dict[str, Any]:
    attrs = graph.nodes[node_id]
    keys = [
        "node_type",
        "protein_id",
        "chain_id",
        "residue_index",
        "residue_name",
        "residue_type",
        "plddt",
        "name",
        "go_id",
        "disease_id",
        "pathway_id",
        "domain_id",
    ]
    return {"id": node_id, **{key: attrs[key] for key in keys if key in attrs}}


def explain_variant(
    graph: nx.MultiDiGraph,
    variant_id: str,
    feature_row: dict[str, Any] | None = None,
    *,
    max_contacts: int = 5,
) -> dict[str, Any]:
    """Create a compact explanation object grounded in edge provenance."""

    variant_node = variant_node_id(variant_id)
    residue_node = _located_residue_for_variant(graph, variant_id)
    residue = graph.nodes[residue_node]
    protein_node = protein_node_id(str(residue["protein_id"]))
    located_at: list[dict[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    domains: list[dict[str, Any]] = []

    for source, target, _key, attrs in _incident_edges(graph, residue_node):
        edge_type = attrs.get("edge_type")
        neighbor = _other_node(source, target, residue_node)
        if edge_type == EdgeType.LOCATED_AT.value and neighbor == variant_node:
            located_at.append(
                {
                    "edge_type": edge_type,
                    "source": source,
                    "target": target,
                    "provenance": _provenance(attrs),
                }
            )
        elif edge_type == EdgeType.STRUCTURAL_CONTACT.value:
            contacts.append(
                {
                    "edge_type": edge_type,
                    "contact_residue": _node_summary(graph, neighbor),
                    "distance": attrs.get("distance"),
                    "confidence": attrs.get("confidence"),
                    "provenance": _provenance(attrs),
                }
            )
        elif edge_type == EdgeType.LOCATED_AT.value:
            if graph.nodes[neighbor].get("node_type") == NodeType.DOMAIN.value:
                domains.append(
                    {
                        "edge_type": edge_type,
                        "domain": _node_summary(graph, neighbor),
                        "provenance": _provenance(attrs),
                    }
                )

    protein_context: list[dict[str, Any]] = []
    for source, target, _key, attrs in _incident_edges(graph, protein_node):
        if attrs.get("edge_type") not in {
            EdgeType.HAS_FUNCTION.value,
            EdgeType.PARTICIPATES_IN.value,
            EdgeType.ASSOCIATED_WITH.value,
        }:
            continue
        neighbor = _other_node(source, target, protein_node)
        if graph.nodes[neighbor].get("node_type") in {
            NodeType.GO_TERM.value,
            NodeType.PATHWAY.value,
            NodeType.DISEASE.value,
        }:
            protein_context.append(
                {
                    "edge_type": attrs.get("edge_type"),
                    "neighbor": _node_summary(graph, neighbor),
                    "provenance": _provenance(attrs),
                }
            )

    contacts.sort(key=lambda item: float(item["distance"] or 0.0))
    return {
        "variant": _node_summary(graph, variant_node),
        "residue": _node_summary(graph, residue_node),
        "features": feature_row or {},
        "evidence": {
            "located_at": located_at,
            "structural_contacts": contacts[:max_contacts],
            "domain_membership": domains,
            "protein_context": protein_context,
        },
    }


def export_variant_explanations(
    graph: nx.MultiDiGraph,
    feature_table: pd.DataFrame,
    path: str | Path,
) -> None:
    """Write one JSON explanation per variant feature row."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, row in feature_table.iterrows():
            explanation = explain_variant(
                graph,
                str(row["variant_id"]),
                row.where(pd.notnull(row), None).to_dict(),
            )
            handle.write(json.dumps(explanation, sort_keys=True) + "\n")
