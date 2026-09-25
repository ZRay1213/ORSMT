"""Pinned source-structure review; no candidate outputs determine admission."""
import hashlib
import json
from pathlib import Path

COMMIT = 'f15bbc4477c70db85ad148df8bcc1b780bca0f8c'
BASE = 'https://raw.githubusercontent.com/optsuite/OptMATH/' + COMMIT + '/'
URL = BASE + 'benchmark/OptMATH_Bench.json'
OFFICIAL_SHA256 = 'c5dbc2bf14995b880d13e306ff54ae81b603f07b565a31e52248533848ad3a93'
REVIEW_PATH = Path(__file__).with_name('optmath_source_review.json')


def review_index(review):
    if review['commit'] != COMMIT:
        raise ValueError('Source review commit changed')
    out = {}
    groups = [(name, spec, 'eligible') for name, spec in review['families'].items()]
    groups.append(('quadratic_or_conic', review['nonlinear'], 'excluded'))
    for name, spec, status in groups:
        for i in spec['official_ids']:
            if i in out:
                raise ValueError('Repeated official review ID')
            out[i] = (name, spec, status)
    if set(out) != set(range(166)):
        raise ValueError('Review must cover the full official release')
    return out


def classify_optmath(bank, evidence):
    from build_milp_benchmark import norm
    official, source_ref = evidence.download('optmath_official.json', URL)
    if source_ref['sha256'] != OFFICIAL_SHA256:
        raise ValueError('Pinned OptMATH release hash mismatch')
    review = json.loads(REVIEW_PATH.read_text())
    reviewed = review_index(review)
    review_ref = evidence.file(REVIEW_PATH)
    if len(official) != 166 or {r['id'] for r in official} != set(reviewed):
        raise ValueError('Official release ID mismatch')
    by_text = {}
    for row in official:
        key = norm(row['en_question'])
        if key in by_text:
            raise ValueError('Ambiguous official question match')
        by_text[key] = row
    subset = [r for r in bank if r['source'] == 'optmath_bench']
    if len(subset) != 166:
        raise ValueError('Unexpected local OptMATH population')
    decisions, seen = {}, set()
    for row in subset:
        src = by_text.get(norm(row['question']))
        if src is None or src['id'] in seen:
            raise ValueError('Missing or repeated exact official question match')
        seen.add(src['id'])
        family, spec, status = reviewed[src['id']]
        refs = [source_ref, review_ref]
        if spec.get('structural_reference'):
            refs = refs + [{'url': BASE + spec['structural_reference'],
                            'role': 'structural reference, not a per-instance parameter match'}]
        decisions[row['id']] = {
            'id': row['id'], 'source': row['source'], 'status': status,
            'reason': 'reviewed_linear_source_structure' if status == 'eligible'
                      else 'explicit_quadratic_or_conic_source_constraint',
            'released_type': None, 'reviewed_type': spec['scope_type'],
            'official_id': src['id'], 'reviewed_family': family,
            'question_sha256': hashlib.sha256(row['question'].encode()).hexdigest(),
            'review_rationale': spec['rationale'], 'evidence': refs,
            'review_basis': review['review_basis']}
    if seen != set(reviewed):
        raise ValueError('Official IDs not completely matched')
    return decisions
