# Public snapshot validation

Date: 2026-09-29. Checks used the staged public source on Windows with available local dependency environments. This is not a claim of a fresh cross-platform dependency installation.

## Executed checks

- Test result: **33 passed**.
- Command: `python -m pytest tests -q`.
- All retained scientific-core tests passed. The local demo produced a graph with 16 nodes and 54 edges. This is a toy-data demonstration, not clinical validation.
- All retained Python source was parsed for syntax errors.
- The primary README's local links were checked.
- Gitleaks 8.30.1 and an additional identity/address scan were used before upload. Actual credentials are excluded; explicit test-only dummy strings are not provider credentials.

## Boundaries

The source snapshot omits large raw datasets, model weights, training checkpoints, service databases, and unrelated administration/report-generation archives. Historical research values remain historical values. Paths/identities in historical notes were generalized; old artifact manifests are not a checksum guarantee for this curated publication.

## Clean Git-tree verification

The files selected by Git were exported to a separate clean directory and the stated public test scope was run again from that export. This checks that ignored local packages or uncommitted helper files are not required by those tests. No private source history is included.
