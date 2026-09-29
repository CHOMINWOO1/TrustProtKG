# TrustProtKG v0.5 Real-Derived Snapshot Documentation

This snapshot pack is intentionally tiny and offline. It uses real public
biomedical identifiers and manually curated, real-derived rows for three human
proteins: TP53, BRCA1, and PTEN. The compact PDB files are local PDB-style
fragments that preserve real residue numbering and residue names needed by the
variant fixtures; they are not full AlphaFold downloads.

Retrieval/documentation date: 2026-06-12.

## Source Families

| Source | Purpose | URL or citation | License or usage note |
| --- | --- | --- | --- |
| UniProt | Protein accessions and reviewed metadata | https://www.uniprot.org/uniprotkb/P04637/entry, https://www.uniprot.org/uniprotkb/P38398/entry, https://www.uniprot.org/uniprotkb/P60484/entry | UniProt terms apply; rows here are a minimal attribution-preserving fixture. |
| AlphaFold DB / PDB style | Structure references and PDB-style local fragments | https://alphafold.ebi.ac.uk/entry/P04637, https://alphafold.ebi.ac.uk/entry/P38398, https://alphafold.ebi.ac.uk/entry/P60484 | AlphaFold DB terms apply; local fragments are compact test fixtures. |
| ClinVar | Missense variant labels and review-style fields | https://www.ncbi.nlm.nih.gov/clinvar/ and ClinVar publications | Attribution to ClinVar requested for redistributed/cited data. |
| Gene Ontology | Functional annotations | https://geneontology.org/ and GO term pages | GO data terms apply; rows are minimal real-derived annotations. |
| InterPro | Domain annotations | https://www.ebi.ac.uk/interpro/ | InterPro/EMBL-EBI terms apply; rows are minimal real-derived annotations. |
| Reactome | Pathway annotations | https://reactome.org/ | Reactome terms apply; rows are minimal real-derived annotations. |
| MONDO/OMIM-style disease sources | Disease association identifiers and names | https://mondo.monarchinitiative.org/ and disease resource citations | Source-specific terms apply; rows are compact research fixtures. |

## Curation Note

The files under `raw/` are not bulk exports. They are small curated snapshots
designed to exercise the TrustProtKG ingestion contract while keeping tests and
the main pipeline fully offline. For a paper-scale experiment, replace these
files with versioned real exports and regenerate the manifest checksums.
