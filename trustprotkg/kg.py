"""Heterogeneous biomedical-style knowledge graph construction."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

from trustprotkg.models import (
    EdgeProvenance,
    EdgeType,
    NodeType,
    disease_node_id,
    domain_node_id,
    go_node_id,
    pathway_node_id,
    protein_node_id,
    residue_node_id,
    structure_node_id,
    variant_node_id,
)
from trustprotkg.structure import build_residue_graph


def _display_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path_obj)


def _provenance_from_row(
    row: pd.Series,
    created_at: str,
    default_source_db: str,
    default_evidence_type: str,
    derived_from: str,
) -> EdgeProvenance:
    return EdgeProvenance(
        source_db=str(row.get("source_db", default_source_db)),
        evidence_type=str(row.get("evidence_type", default_evidence_type)),
        confidence=float(row.get("confidence", 1.0)),
        created_at=created_at,
        derived_from=str(row.get("derived_from", derived_from)),
    )


def _edge_attrs(edge_type: EdgeType, provenance: EdgeProvenance) -> dict[str, object]:
    attrs = provenance.as_edge_attributes()
    attrs["edge_type"] = edge_type.value
    attrs["source"] = "trustprotkg.kg"
    return attrs


def _add_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    edge_type: EdgeType,
    provenance: EdgeProvenance,
) -> None:
    graph.add_edge(source, target, **_edge_attrs(edge_type, provenance))


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def _merge_residue_graph(graph: nx.MultiDiGraph, residue_graph: nx.MultiDiGraph) -> None:
    for node_id, attrs in residue_graph.nodes(data=True):
        graph.add_node(node_id, **attrs)
    for source, target, _key, attrs in residue_graph.edges(keys=True, data=True):
        graph.add_edge(source, target, **attrs)


def _add_proteins(graph: nx.MultiDiGraph, proteins: pd.DataFrame) -> None:
    for _, row in proteins.iterrows():
        pid = str(row["protein_id"])
        node_id = protein_node_id(pid)
        graph.add_node(
            node_id,
            node_type=NodeType.PROTEIN.value,
            protein_id=pid,
            symbol=str(row.get("symbol", "")),
            name=str(row.get("name", "")),
            taxon=str(row.get("taxon", "")),
        )


def _add_structure_and_residues(
    graph: nx.MultiDiGraph,
    *,
    structure_file: str | Path,
    protein_id: str,
    structure_id: str,
    contact_threshold_angstrom: float,
    structure_source_db: str,
    created_at: str,
    derived_from: str | None = None,
) -> None:
    structure_node = structure_node_id(structure_id)
    structure_derived_from = derived_from or _display_path(structure_file)
    graph.add_node(
        structure_node,
        node_type=NodeType.STRUCTURE_MODEL.value,
        structure_id=structure_id,
        protein_id=protein_id,
        structure_file=_display_path(structure_file),
        method=structure_source_db,
    )
    structure_provenance = EdgeProvenance(
        source_db=structure_source_db,
        evidence_type="toy_structure_model",
        confidence=1.0,
        created_at=created_at,
        derived_from=structure_derived_from,
    )
    _add_edge(
        graph,
        protein_node_id(protein_id),
        structure_node,
        EdgeType.HAS_STRUCTURE,
        structure_provenance,
    )

    residue_graph = build_residue_graph(
        pdb_path=structure_file,
        protein_id=protein_id,
        contact_threshold_angstrom=contact_threshold_angstrom,
        source_db=structure_source_db,
        created_at=created_at,
        derived_from=structure_derived_from,
    )
    _merge_residue_graph(graph, residue_graph)

    for residue_id, residue_attrs in residue_graph.nodes(data=True):
        residue_provenance = EdgeProvenance(
            source_db=structure_source_db,
            evidence_type="residue_observed_in_structure_file",
            confidence=1.0,
            created_at=created_at,
            derived_from=structure_derived_from,
        )
        _add_edge(
            graph,
            protein_node_id(str(residue_attrs["protein_id"])),
            residue_id,
            EdgeType.HAS_RESIDUE,
            residue_provenance,
        )
        _add_edge(
            graph,
            structure_node,
            residue_id,
            EdgeType.HAS_RESIDUE,
            residue_provenance,
        )


def _add_annotation_layers(
    graph: nx.MultiDiGraph,
    *,
    variants: pd.DataFrame,
    go_annotations: pd.DataFrame,
    disease_associations: pd.DataFrame,
    domains: pd.DataFrame,
    pathways: pd.DataFrame,
    created_at: str,
    derived_from: str,
) -> None:
    for _, row in variants.iterrows():
        variant_id = str(row["variant_id"])
        node_id = variant_node_id(variant_id)
        graph.add_node(
            node_id,
            node_type=NodeType.VARIANT.value,
            variant_id=variant_id,
            protein_id=str(row["protein_id"]),
            chain_id=str(row["chain_id"]),
            residue_index=int(row["residue_index"]),
            reference_aa=str(row["reference_aa"]),
            alternate_aa=str(row["alternate_aa"]),
            clinical_label=str(row.get("clinical_label", "")),
            label_binary=(
                int(row["label_binary"]) if str(row.get("label_binary", "")) != "" else ""
            ),
            split=str(row.get("split", "")),
        )
        provenance = _provenance_from_row(
            row, created_at, "toy_variants", "toy_variant_annotation", derived_from
        )
        _add_edge(
            graph,
            node_id,
            residue_node_id(
                str(row["protein_id"]), str(row["chain_id"]), int(row["residue_index"])
            ),
            EdgeType.LOCATED_AT,
            provenance,
        )

    for _, row in domains.iterrows():
        domain_id = str(row["domain_id"])
        node_id = domain_node_id(domain_id)
        start = int(row["start"])
        end = int(row["end"])
        chain_id = str(row["chain_id"])
        graph.add_node(
            node_id,
            node_type=NodeType.DOMAIN.value,
            domain_id=domain_id,
            protein_id=str(row["protein_id"]),
            chain_id=chain_id,
            start=start,
            end=end,
            name=str(row["name"]),
        )
        provenance = _provenance_from_row(
            row, created_at, "toy_domains", "toy_domain_annotation", derived_from
        )
        _add_edge(
            graph,
            protein_node_id(str(row["protein_id"])),
            node_id,
            EdgeType.HAS_DOMAIN,
            provenance,
        )
        for residue_index in range(start, end + 1):
            residue_id = residue_node_id(str(row["protein_id"]), chain_id, residue_index)
            if residue_id in graph:
                _add_edge(graph, residue_id, node_id, EdgeType.LOCATED_AT, provenance)

    for _, row in go_annotations.iterrows():
        go_id = str(row["go_id"])
        node_id = go_node_id(go_id)
        graph.add_node(
            node_id,
            node_type=NodeType.GO_TERM.value,
            go_id=go_id,
            name=str(row["name"]),
            namespace=str(row.get("namespace", "")),
        )
        provenance = _provenance_from_row(
            row, created_at, "toy_go", "toy_go_annotation", derived_from
        )
        _add_edge(
            graph,
            protein_node_id(str(row["protein_id"])),
            node_id,
            EdgeType.HAS_FUNCTION,
            provenance,
        )

    for _, row in disease_associations.iterrows():
        disease_id = str(row["disease_id"])
        node_id = disease_node_id(disease_id)
        graph.add_node(
            node_id,
            node_type=NodeType.DISEASE.value,
            disease_id=disease_id,
            name=str(row["name"]),
        )
        provenance = _provenance_from_row(
            row, created_at, "toy_diseases", "toy_disease_association", derived_from
        )
        _add_edge(
            graph,
            protein_node_id(str(row["protein_id"])),
            node_id,
            EdgeType.ASSOCIATED_WITH,
            provenance,
        )

    for _, row in pathways.iterrows():
        pathway_id = str(row["pathway_id"])
        node_id = pathway_node_id(pathway_id)
        graph.add_node(
            node_id,
            node_type=NodeType.PATHWAY.value,
            pathway_id=pathway_id,
            name=str(row["name"]),
        )
        provenance = _provenance_from_row(
            row, created_at, "toy_pathways", "toy_pathway_annotation", derived_from
        )
        _add_edge(
            graph,
            protein_node_id(str(row["protein_id"])),
            node_id,
            EdgeType.PARTICIPATES_IN,
            provenance,
        )


def build_knowledge_graph(
    *,
    proteins_path: str | Path,
    variants_path: str | Path,
    go_annotations_path: str | Path,
    disease_associations_path: str | Path,
    domains_path: str | Path,
    pathways_path: str | Path,
    structure_file: str | Path,
    protein_id: str,
    structure_id: str,
    contact_threshold_angstrom: float,
    structure_source_db: str,
    created_at: str,
    derived_from: str,
) -> nx.MultiDiGraph:
    """Build the single-structure TrustProtKG demo graph from local toy data."""

    graph = nx.MultiDiGraph(name="TrustProtKG")
    proteins = _read_csv(proteins_path)
    variants = _read_csv(variants_path)
    go_annotations = _read_csv(go_annotations_path)
    disease_associations = _read_csv(disease_associations_path)
    domains = _read_csv(domains_path)
    pathways = _read_csv(pathways_path)

    _add_proteins(graph, proteins)
    _add_structure_and_residues(
        graph,
        structure_file=structure_file,
        protein_id=protein_id,
        structure_id=structure_id,
        contact_threshold_angstrom=contact_threshold_angstrom,
        structure_source_db=structure_source_db,
        created_at=created_at,
        derived_from=_display_path(structure_file),
    )
    _add_annotation_layers(
        graph,
        variants=variants,
        go_annotations=go_annotations,
        disease_associations=disease_associations,
        domains=domains,
        pathways=pathways,
        created_at=created_at,
        derived_from=derived_from,
    )

    return graph


def build_benchmark_knowledge_graph(
    *,
    proteins_path: str | Path,
    variants_path: str | Path,
    go_annotations_path: str | Path,
    disease_associations_path: str | Path,
    domains_path: str | Path,
    pathways_path: str | Path,
    structures_path: str | Path,
    contact_threshold_angstrom: float,
    default_structure_source_db: str,
    created_at: str,
    derived_from: str,
) -> nx.MultiDiGraph:
    """Build a reproducible multi-protein benchmark graph."""

    graph = nx.MultiDiGraph(name="TrustProtKG benchmark")
    proteins = _read_csv(proteins_path)
    variants = _read_csv(variants_path)
    go_annotations = _read_csv(go_annotations_path)
    disease_associations = _read_csv(disease_associations_path)
    domains = _read_csv(domains_path)
    pathways = _read_csv(pathways_path)
    structures = _read_csv(structures_path)

    _add_proteins(graph, proteins)
    for _, row in structures.iterrows():
        structure_file = Path(str(row["structure_file"]))
        if not structure_file.is_absolute():
            structure_file = Path.cwd() / structure_file
        source_db = str(row.get("source_db", default_structure_source_db))
        _add_structure_and_residues(
            graph,
            structure_file=structure_file,
            protein_id=str(row["protein_id"]),
            structure_id=str(row["structure_id"]),
            contact_threshold_angstrom=contact_threshold_angstrom,
            structure_source_db=source_db,
            created_at=created_at,
            derived_from=_display_path(structure_file),
        )

    _add_annotation_layers(
        graph,
        variants=variants,
        go_annotations=go_annotations,
        disease_associations=disease_associations,
        domains=domains,
        pathways=pathways,
        created_at=created_at,
        derived_from=derived_from,
    )

    return graph

    return graph
