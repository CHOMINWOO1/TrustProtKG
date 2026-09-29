"""Residue-level protein structure graph construction."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

from trustprotkg.models import (
    EdgeProvenance,
    EdgeType,
    NodeType,
    ResidueRecord,
    residue_node_id,
)


def _one_letter_code(residue_name: str) -> str:
    try:
        return seq1(residue_name)
    except Exception:
        return "X"


def _normalized_plddt(value: float | None) -> float:
    if value is None:
        return 0.5
    if value > 1.0:
        return max(0.0, min(1.0, value / 100.0))
    return max(0.0, min(1.0, value))


def _distance(coord_a: Iterable[float], coord_b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(coord_a) - np.asarray(coord_b)))


def parse_residues_from_pdb(
    pdb_path: str | Path,
    protein_id: str,
) -> list[ResidueRecord]:
    """Parse standard PDB or AlphaFold-style PDB residues.

    AlphaFold DB stores pLDDT in the B-factor column. We copy the CA atom
    B-factor into the residue-level `plddt` field when CA is available.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_id, str(pdb_path))
    model = next(structure.get_models())
    records: list[ResidueRecord] = []

    for chain in model:
        chain_id = chain.id
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            residue_index = int(residue.id[1])
            residue_name = residue.get_resname().strip()
            ca_coord = None
            plddt = None
            if "CA" in residue:
                ca_atom = residue["CA"]
                ca_coord = [float(x) for x in ca_atom.coord.tolist()]
                plddt = float(ca_atom.bfactor)

            records.append(
                ResidueRecord(
                    protein_id=protein_id,
                    chain_id=chain_id,
                    residue_index=residue_index,
                    residue_name=residue_name,
                    residue_type=_one_letter_code(residue_name),
                    ca_coord=ca_coord,
                    plddt=plddt,
                )
            )

    records.sort(key=lambda item: (item.chain_id, item.residue_index))
    return records


def _add_residue_node(graph: nx.MultiDiGraph, residue: ResidueRecord) -> str:
    node_id = residue_node_id(
        residue.protein_id, residue.chain_id, residue.residue_index
    )
    graph.add_node(
        node_id,
        node_type=NodeType.RESIDUE.value,
        **residue.model_dump(),
    )
    return node_id


def _edge_attrs(
    edge_type: EdgeType,
    provenance: EdgeProvenance,
    source: str,
    distance: float | None = None,
    confidence: float | None = None,
) -> dict[str, object]:
    attrs = provenance.as_edge_attributes()
    attrs.update(
        {
            "edge_type": edge_type.value,
            "source": source,
            "confidence": provenance.confidence if confidence is None else confidence,
        }
    )
    attrs["provenance"]["confidence"] = attrs["confidence"]
    if distance is not None:
        attrs["distance"] = distance
    return attrs


def build_residue_graph(
    pdb_path: str | Path,
    protein_id: str,
    contact_threshold_angstrom: float = 8.0,
    source_db: str = "local PDB",
    created_at: str = "2026-06-12T00:00:00Z",
    derived_from: str | None = None,
) -> nx.MultiDiGraph:
    """Build residue nodes plus sequence and structural-contact edges."""

    residues = parse_residues_from_pdb(pdb_path, protein_id)
    graph = nx.MultiDiGraph(name=f"{protein_id}_residue_graph")
    residue_ids: dict[tuple[str, int], str] = {}

    for residue in residues:
        residue_ids[(residue.chain_id, residue.residue_index)] = _add_residue_node(
            graph, residue
        )

    base_derived_from = derived_from or str(pdb_path)
    sequence_provenance = EdgeProvenance(
        source_db=source_db,
        evidence_type="sequence_order_from_structure_file",
        confidence=1.0,
        created_at=created_at,
        derived_from=base_derived_from,
    )
    contact_provenance = EdgeProvenance(
        source_db=source_db,
        evidence_type=f"CA_distance_le_{contact_threshold_angstrom:g}_angstrom",
        confidence=1.0,
        created_at=created_at,
        derived_from=base_derived_from,
    )

    residues_by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in residues:
        residues_by_chain.setdefault(residue.chain_id, []).append(residue)

    for chain_residues in residues_by_chain.values():
        for left, right in zip(chain_residues, chain_residues[1:]):
            if right.residue_index != left.residue_index + 1:
                continue
            left_id = residue_ids[(left.chain_id, left.residue_index)]
            right_id = residue_ids[(right.chain_id, right.residue_index)]
            distance = None
            if left.ca_coord is not None and right.ca_coord is not None:
                distance = _distance(left.ca_coord, right.ca_coord)
            graph.add_edge(
                left_id,
                right_id,
                **_edge_attrs(
                    EdgeType.SEQUENCE_NEIGHBOR,
                    sequence_provenance,
                    source="trustprotkg.structure",
                    distance=distance,
                    confidence=1.0,
                ),
            )

    residues_with_ca = [residue for residue in residues if residue.ca_coord is not None]
    for left, right in combinations(residues_with_ca, 2):
        distance = _distance(left.ca_coord or [], right.ca_coord or [])
        if distance > contact_threshold_angstrom:
            continue
        confidence = min(_normalized_plddt(left.plddt), _normalized_plddt(right.plddt))
        graph.add_edge(
            residue_ids[(left.chain_id, left.residue_index)],
            residue_ids[(right.chain_id, right.residue_index)],
            **_edge_attrs(
                EdgeType.STRUCTURAL_CONTACT,
                contact_provenance,
                source="trustprotkg.structure",
                distance=distance,
                confidence=confidence,
            ),
        )

    return graph
