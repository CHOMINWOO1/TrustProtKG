# TrustProtKG v1.4 Evidence-Enriched Snapshot Documentation

This compact offline snapshot extends the v1.2 expanded real-derived pack with
source-style evidence enrichment fields. It keeps the same eight proteins,
twenty-four variants, four protein families, and compact PDB-style fragments,
but adds evidence fields that support provenance diagnostics and manuscript
claim annotation.

The snapshot preserves public accession-style identifiers, real residue
numbering, source-style provenance fields, local PDB-style fragments, and
curated family labels suitable for leakage-aware prototype experiments.

Evidence-enriched annotation fields:

- `review_status`: source-like review or curation state.
- `assertion_conflict`: normalized conflict flag such as `no_conflict` or
  `conflicting_assertions`.
- `annotation_date`: deterministic snapshot-local annotation date.
- `evidence_strength`: normalized `strong`, `moderate`, `weak`, or
  `conflicting` evidence category.
- `calibrated_confidence`: source-specific confidence after a simple
  deterministic calibration rule.

Protein families:

- `tumor_suppressor`: TP53, BRCA1, PTEN
- `receptor_tyrosine_kinase`: EGFR, ERBB2
- `ras_gtpase`: KRAS
- `serine_threonine_kinase`: AKT1, BRAF

The train/test split is intentionally more balanced than v0.8 at the test-set
level. Each protein contributes local variants with residue-level structure
coverage, and the snapshot is paired with v1.4 diagnostics that report
label-balance, split-leakage risk, provenance completeness, evidence strength,
assertion conflicts, source calibration, dropped rows, and coordinate coverage.

The files are intentionally tiny and are not a replacement for live database
downloads. They are fixture snapshots for offline software validation and
paper-method prototyping.
