# TrustProtKG v0.8 Family Snapshot Documentation

This compact offline snapshot extends the v0.5 real-derived pack from TP53,
BRCA1, and PTEN to six proteins by adding EGFR, ERBB2, and KRAS. It preserves
public accession-style identifiers, real residue numbering, source-style
provenance fields, and local PDB-style fragments suitable for deterministic
tests.

The snapshot adds curated protein-family metadata for family-aware prototype
experiments:

- `tumor_suppressor`: TP53, BRCA1, PTEN
- `receptor_tyrosine_kinase`: EGFR, ERBB2
- `ras_gtpase`: KRAS

The files are intentionally tiny and are not a replacement for live database
downloads. They are fixture snapshots for offline software validation and
paper-method prototyping.
