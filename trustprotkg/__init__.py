"""TrustProtKG: a provenance-aware protein structure KG research MVP."""

from trustprotkg.features import extract_variant_features
from trustprotkg.kg import build_benchmark_knowledge_graph, build_knowledge_graph
from trustprotkg.safe_io import install_safe_path_write_text
from trustprotkg.structure import build_residue_graph

install_safe_path_write_text()

__all__ = [
    "build_benchmark_knowledge_graph",
    "build_knowledge_graph",
    "build_residue_graph",
    "extract_variant_features",
]
