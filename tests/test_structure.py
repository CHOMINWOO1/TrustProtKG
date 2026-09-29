from pathlib import Path

from trustprotkg.models import EdgeType
from trustprotkg.structure import build_residue_graph, parse_residues_from_pdb


ROOT = Path(__file__).resolve().parents[1]
PDB_PATH = ROOT / "data" / "structures" / "tp53_demo_af_style.pdb"


def test_parse_residues_from_alpha_fold_style_pdb():
    residues = parse_residues_from_pdb(PDB_PATH, protein_id="TP53_HUMAN")

    assert len(residues) == 8
    assert residues[0].residue_name == "MET"
    assert residues[0].residue_type == "M"
    assert residues[0].ca_coord == [0.0, 0.0, 0.0]
    assert residues[0].plddt == 92.0


def test_residue_graph_has_sequence_and_contact_edges_with_provenance():
    graph = build_residue_graph(
        PDB_PATH,
        protein_id="TP53_HUMAN",
        contact_threshold_angstrom=8.0,
        source_db="AlphaFold-style toy PDB",
    )

    sequence_edges = [
        attrs
        for _, _, attrs in graph.edges(data=True)
        if attrs["edge_type"] == EdgeType.SEQUENCE_NEIGHBOR.value
    ]
    contact_edges = [
        attrs
        for _, _, attrs in graph.edges(data=True)
        if attrs["edge_type"] == EdgeType.STRUCTURAL_CONTACT.value
    ]

    assert graph.number_of_nodes() == 8
    assert len(sequence_edges) == 7
    assert contact_edges
    assert all(edge["distance"] <= 8.0 for edge in contact_edges)
    for edge in graph.edges(data=True):
        attrs = edge[2]
        for key in ["source_db", "evidence_type", "confidence", "created_at", "derived_from"]:
            assert key in attrs
















