#!/usr/bin/env python3
"""Make a same-ID view of any candidate ledger, preserving failed outputs."""
import argparse
import collections
import json
from pathlib import Path
from build_milp_benchmark import digest, read_rows, write_json
from milp_scope_policy import ACTIVE_SCOPE, verify_active_scope


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scope', type=Path, default=ACTIVE_SCOPE)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    m = verify_active_scope(args.scope)
    allowed = {r['id'] for r in read_rows(args.scope / 'eligible.jsonl')}
    before = digest(args.input)
    seen = collections.Counter()
    kept = 0
    total = 0
    # Never overwrite a previous view or mutate/repair input rows during filtering.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as f, args.output.open('x') as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            if row['id'] in allowed:
                out.write(line if line.endswith('\n') else line + '\n')
                seen[row['id']] += 1
                kept += 1
    missing = sorted(allowed - set(seen))
    repeated = {k: n for k, n in seen.items() if n > 1}
    if digest(args.input) != before:
        raise RuntimeError('Candidate input changed while filtering; do not use this view')
    report = {'scope_manifest_sha256': digest(args.scope / 'manifest.json'),
              'expected_problem_count': m['eligible'], 'input': str(args.input), 'input_sha256': before,
              'output_sha256': digest(args.output), 'input_rows': total, 'output_rows': kept,
              'distinct_problem_count': len(seen), 'missing_ids': missing, 'repeated_ids': repeated,
              'all_problem_ids_present': not missing,
              'single_candidate_per_problem': not missing and not repeated,
              'note': 'All selected-ID records, including failed candidates, are retained unchanged. '
                      'Missing records remain missing, not fabricated results; repeated candidates need '
                      'an explicitly chosen protocol before single-candidate aggregation.'}
    write_json(args.output.with_suffix('.scope.json'), report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'missing_ids', 'repeated_ids'}}, indent=2))


if __name__ == '__main__':
    main()
