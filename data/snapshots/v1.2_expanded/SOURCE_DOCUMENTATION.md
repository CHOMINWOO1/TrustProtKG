# TrustProtKG v1.2 Expanded Snapshot Documentation

This compact offline snapshot extends the v0.8 family-aware real-derived pack
from six proteins and eighteen variants to eight proteins and twenty-four
variants. It adds AKT1 and BRAF to improve kinase-family coverage while keeping
the benchmark small enough for deterministic local tests.

The snapshot preserves public accession-style identifiers, real residue
numbering, source-style provenance fields, local PDB-style fragments, and
curated family labels suitable for leakage-aware prototype experiments.

Protein families:

- `tumor_suppressor`: TP53, BRCA1, PTEN
- `receptor_tyrosine_kinase`: EGFR, ERBB2
- `ras_gtpase`: KRAS
- `serine_threonine_kinase`: AKT1, BRAF

The train/test split is intentionally more balanced than v0.8 at the test-set
level. Each protein contributes local variants with residue-level structure
coverage, and the snapshot is paired with v1.2 diagnostics that report
label-balance, split-leakage risk, provenance completeness, dropped rows, and
coordinate coverage.

The files are intentionally tiny and are not a replacement for live database
downloads. They are fixture snapshots for offline software validation and
paper-method prototyping.
