"""Frozen evaluation scope shipped with the public core release."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ACTIVE_SCOPE = PACKAGE / 'data/milp_benchmark_v3'
EXCLUDED_SOURCES = frozenset({'optengine_canonical'})


def verify_active_scope(folder=ACTIVE_SCOPE):
    from build_milp_benchmark import read_rows, verify_scope
    manifest = verify_scope(folder)
    rows = read_rows(Path(folder) / 'eligible.jsonl')
    if any(r['source'] in EXCLUDED_SOURCES or r['id'].startswith('optengine_canonical:')
           for r in rows):
        raise ValueError('OPT-Engine is excluded from the active MILP experiments')
    return manifest
