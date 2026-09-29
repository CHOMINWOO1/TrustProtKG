import json
from pathlib import Path

from trustprotkg.features import extract_variant_features
from trustprotkg.kg import build_knowledge_graph
from trustprotkg.models import EdgeType, NodeType
from trustprotkg.pipeline import run_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "demo.yaml"


def _build_demo_graph():
    return build_knowledge_graph(
        proteins_path=ROOT / "data" / "toy" / "proteins.csv",
        variants_path=ROOT / "data" / "toy" / "variants.csv",
        go_annotations_path=ROOT / "data" / "toy" / "go_annotations.csv",
        disease_associations_path=ROOT / "data" / "toy" / "disease_associations.csv",
        domains_path=ROOT / "data" / "toy" / "domains.csv",
        pathways_path=ROOT / "data" / "toy" / "pathways.csv",
        structure_file=ROOT / "data" / "structures" / "tp53_demo_af_style.pdb",
        protein_id="TP53_HUMAN",
        structure_id="AF-TP53-DEMO",
        contact_threshold_angstrom=8.0,
        structure_source_db="AlphaFold-style toy PDB",
        created_at="2026-06-12T00:00:00Z",
        derived_from="local toy data",
    )


def test_heterogeneous_schema_and_edge_provenance():
    graph = _build_demo_graph()
    node_types = {attrs["node_type"] for _, attrs in graph.nodes(data=True)}
    edge_types = {attrs["edge_type"] for _, _, attrs in graph.edges(data=True)}

    assert {
        NodeType.PROTEIN.value,
        NodeType.STRUCTURE_MODEL.value,
        NodeType.RESIDUE.value,
        NodeType.VARIANT.value,
        NodeType.DOMAIN.value,
        NodeType.GO_TERM.value,
        NodeType.DISEASE.value,
        NodeType.PATHWAY.value,
    }.issubset(node_types)
    assert {
        EdgeType.HAS_STRUCTURE.value,
        EdgeType.HAS_RESIDUE.value,
        EdgeType.HAS_DOMAIN.value,
        EdgeType.HAS_FUNCTION.value,
        EdgeType.PARTICIPATES_IN.value,
        EdgeType.ASSOCIATED_WITH.value,
        EdgeType.LOCATED_AT.value,
        EdgeType.SEQUENCE_NEIGHBOR.value,
        EdgeType.STRUCTURAL_CONTACT.value,
    }.issubset(edge_types)

    for _, _, attrs in graph.edges(data=True):
        for key in ["source_db", "evidence_type", "confidence", "created_at", "derived_from"]:
            assert key in attrs
        assert 0.0 <= float(attrs["confidence"]) <= 1.0


def test_variant_feature_extractor():
    graph = _build_demo_graph()
    features = extract_variant_features(graph, "V_TP53_Y5H")

    assert features["residue_confidence"] == 76.0
    assert features["num_structural_contacts"] > 0
    assert features["avg_contact_distance"] <= 8.0
    assert features["is_in_domain"] is True
    assert features["num_go_pathway_disease_edges"] == 4


def test_pipeline_writes_jsonl_graphml_and_features():
    result = run_pipeline(CONFIG)

    assert result.node_count > 0
    assert result.edge_count > 0
    assert result.graph_jsonl.exists()
    assert result.graphml.exists()
    assert result.variant_features.exists()

    first_record = json.loads(result.graph_jsonl.read_text().splitlines()[0])
    assert first_record["record_type"] == "node"
















