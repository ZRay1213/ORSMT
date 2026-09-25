# Frozen release data

`milp_benchmark_v3/` defines the 3,034-problem evaluation scope. Its provenance
records retain upstream URLs and content hashes. Paths beginning with `SOURCE_ROOT/`
are normalized provenance paths from the original experiment host; the external
source checkouts are not part of this repository.

`paper_core/candidates/` contains one candidate ledger for each of the six released
model sizes. `paper_core/verification/` contains the corresponding frozen ORSMT
model--code verification ledger. Every ledger has exactly one row for each scope ID.

`worked_trace/nl4opt67/` is a compact end-to-end trace used by the integration checks.

Run the following command from the repository root to verify file hashes, ID coverage,
and all released totals without network access:

```bash
python scripts/verify_release.py
```
