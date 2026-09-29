"""Baseline missense-variant feature extraction."""

from __future__ import annotations

from statistics import mean
from typing import Any

import networkx as nx
import pandas as pd

from trustprotkg.models import EdgeType, NodeType, protein_node_id, variant_node_id


def _incident_edges(graph: nx.MultiDiGraph, node_id: str):
    yield from graph.out_edges(node_id, keys=True, data=True)
    yield from graph.in_edges(node_id, keys=True, data=True)


def _other_node(source: str, target: str, node_id: str) -> str:
    return target if source == node_id else source


def _located_residue_for_variant(graph: nx.MultiDiGraph, variant_id: str) -> str:
    node_id = variant_node_id(variant_id)
    if node_id not in graph:
        raise KeyError(f"Variant not found: {variant_id}")
    for _source, target, _key, attrs in graph.out_edges(node_id, keys=True, data=True):
        if attrs.get("edge_type") == EdgeType.LOCATED_AT.value:
            if graph.nodes[target].get("node_type") == NodeType.RESIDUE.value:
                return target
    raise ValueError(f"Variant has no LOCATED_AT residue edge: {variant_id}")


def _residue_in_domain(graph: nx.MultiDiGraph, residue_id: str) -> bool:
    for source, target, _key, attrs in _incident_edges(graph, residue_id):
        if attrs.get("edge_type") != EdgeType.LOCATED_AT.value:
            continue
        neighbor = _other_node(source, target, residue_id)
        if graph.nodes[neighbor].get("node_type") == NodeType.DOMAIN.value:
            return True
    return False


def _protein_context_edge_count(graph: nx.MultiDiGraph, protein_id: str) -> int:
    protein_node = protein_node_id(protein_id)
    target_types = {NodeType.GO_TERM.value, NodeType.PATHWAY.value, NodeType.DISEASE.value}
    edge_types = {
        EdgeType.HAS_FUNCTION.value,
        EdgeType.PARTICIPATES_IN.value,
        EdgeType.ASSOCIATED_WITH.value,
    }
    count = 0
    for source, target, _key, attrs in _incident_edges(graph, protein_node):
        neighbor = _other_node(source, target, protein_node)
        if attrs.get("edge_type") in edge_types:
            if graph.nodes[neighbor].get("node_type") in target_types:
                count += 1
    return count


def extract_variant_features(graph: nx.MultiDiGraph, variant_id: str) -> dict[str, Any]:
    """Extract simple interpretable features for one missense variant."""

    residue_id = _located_residue_for_variant(graph, variant_id)
    residue_attrs = graph.nodes[residue_id]
    contact_distances: list[float] = []

    for _source, _target, _key, attrs in _incident_edges(graph, residue_id):
        if attrs.get("edge_type") == EdgeType.STRUCTURAL_CONTACT.value:
            if "distance" in attrs:
                contact_distances.append(float(attrs["distance"]))

    protein_id = str(residue_attrs["protein_id"])
    return {
        "variant_id": variant_id,
        "protein_id": protein_id,
        "chain_id": residue_attrs["chain_id"],
        "residue_index": int(residue_attrs["residue_index"]),
        "residue_confidence": residue_attrs.get("plddt"),
        "num_structural_contacts": len(contact_distances),
        "avg_contact_distance": mean(contact_distances) if contact_distances else None,
        "is_in_domain": _residue_in_domain(graph, residue_id),
        "num_go_pathway_disease_edges": _protein_context_edge_count(graph, protein_id),
    }


def extract_variant_feature_table(
    graph: nx.MultiDiGraph,
    variants: pd.DataFrame,
) -> pd.DataFrame:
    """Extract interpretable features for every variant row in a table."""

    rows: list[dict[str, Any]] = []
    for _, variant in variants.iterrows():
        row = extract_variant_features(graph, str(variant["variant_id"]))
        for column in [
            "reference_aa",
            "alternate_aa",
            "clinical_label",
            "label_binary",
            "split",
        ]:
            if column in variants.columns:
                row[column] = variant[column]
        rows.append(row)
    return pd.DataFrame(rows)
