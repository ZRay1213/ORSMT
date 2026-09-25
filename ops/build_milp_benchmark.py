#!/usr/bin/env python3
"""Freeze a source-based MILP evaluation population without changing original data.

Admission is based on release labels, reviewed reference generators, or explicit
linear matrices. It is NOT a claim that a candidate, or a source description, has
been verified. No model outputs, answers, runtimes, or size thresholds select IDs.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import unicodedata
import urllib.request
from fractions import Fraction
from pathlib import Path
from milp_scope_policy import ACTIVE_SCOPE, EXCLUDED_SOURCES

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE
BANK_HASH = '52d8372079b5dfb419cc3bfdf4a734bff8bc69f706b7c6cb432371dc27c0bef0'
LINEAR = {'LP', 'IP', 'ILP', 'MIP', 'MILP', 'BIP', 'BILP'}
NONLINEAR = {'NLP', 'MINLP', 'QP', 'MIQP', 'QCP', 'MIQCP', 'SOCP', 'SDP'}
RELEASE_SCOPE = {
    'mamo_easy_test': ('EasyLP release', 'https://github.com/FreedomIntelligence/Mamo'),
    'mamo_complex_test': ('ComplexLP release', 'https://github.com/FreedomIntelligence/Mamo'),
    'nl4opt_test': ('NL4Opt linear optimization release', 'https://nl4opt.github.io/'),
    'nlp4lp': ('OptiMUS LP/MILP evaluation release', 'https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html'),
    'complexor': ('OptiMUS LP/MILP evaluation release', 'https://proceedings.mlr.press/v235/ahmaditeshnizi24a.html'),
}
GENERATOR_HASHES = {
    'binpacking': 'b95816975887b0e103d47ce99dbfccf4061ce1817354495dd332a9f990076c48',
    'inventory': 'baf531ed5a35ea083746c218b427a93a143c04b7ea32a4b51ab62434e6297cd0',
    'jobshop': '829ea49538e00b08e29d887c52bcc49c01252d69897b52b3832dae7cf8b4f3a1',
    'knapsack': 'b0af739d821e2f90b13e9fd80d80ea3f50a3e7fa0db8a2eee564f78f31cada98',
    'netflow': 'ce309341e68d5d98f1e8d9481e8256d64b669e9fde4df3c4664aa671bc3ad23e',
    'pollution': '3decd99464ec9794c165a500d0a28385de494bcc18bb667b6b00c5d16a170a00',
    'portfolio': '6d498e8e4f30c922cca44ea98b64e75361bd8b89c67fd6811d21d1ee5751e02a',
    'production': '9d9c1c822d78ce2d0da09bef45d96f6690e4365134d4def6d75b4f62c7e5ea7d',
    'transportation': 'f9f51d10a94b8f7ce4af7f14f98dbb5bffed6403af941e74daf42b7cd4494d0f',
    'tsp': '91de5a100f2d9b064bb711a3339f0ec24906411e3a3e6680a1b3f1f3a6fc0e77',
}
OPTIBENCH_URL = ('https://raw.githubusercontent.com/yangzhch6/ReSocratic/'
                  '27ff7ecb3c6a4955fdcb031f7bf3419c6f462d2e/data/OptiBench.json')
INDUSTRY_URL = ('https://huggingface.co/datasets/CardinalOperations/IndustryOR/resolve/'
                'bd09345a7fbbadedb900b0776e258c445d2cabf2/IndustryOR.json')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def unique_ids(rows):
    ids = [x['id'] for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate problem IDs; refusing to choose a row')
    return set(ids)


def typed(label):
    if label in LINEAR:
        return 'eligible', 'release_linear_type'
    if label in NONLINEAR:
        return 'excluded', 'released_nonlinear_or_conic_formulation'
    return 'pending', 'type_requires_reference_review'


def numeric(x):
    if isinstance(x, bool) or x is None:
        raise ValueError('not a numerical coefficient')
    return Fraction(str(x))


def text2opt_type(d):
    """Check actual sparse matrix structure, not variable names or LLM text."""
    label = d.get('meta', {}).get('optimization_type')
    if label in NONLINEAR:
        return 'excluded', 'release_' + label
    if label not in LINEAR:
        return 'pending', 'unrecognized_matrix_type'
    try:
        q = d.get('quadratic_matrix_q')
        if q is not None and any(numeric(a) != 0 for row in q for a in row):
            return 'excluded', 'nonzero_quadratic_objective'
        rows, variables = d['constraints'], d['variables']
        if not isinstance(rows, dict) or not isinstance(variables, dict) or not variables:
            raise ValueError('missing matrix')
        for row in rows.values():
            if row.get('type') != 'Constraint' or row.get('sense') not in {'<=', '>=', '='}:
                return 'pending', 'unsupported_reference_constraint'
            numeric(row['rhs'])
        for var in variables.values():
            if var.get('type') not in {'Continuous', 'Integer', 'Binary'}:
                return 'pending', 'unsupported_reference_variable'
            numeric(var['objective_linear_coefficient'])
            bounds = var['range']
            if len(bounds) != 2:
                raise ValueError('invalid bound pair')
            for b in bounds:
                if b is not None and str(b).lower() not in {'inf', '+inf', '-inf', 'infinity', '-infinity'}:
                    numeric(b)
            costs = var['resource_costs']
            if not isinstance(costs, dict) or not set(costs) <= set(rows):
                raise ValueError('unknown row reference')
            for coef in costs.values():
                numeric(coef)
    except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError):
        return 'pending', 'reference_matrix_requires_review'
    return 'eligible', 'explicit_linear_matrix'


def text_index(rows, question_key, label_key):
    out = collections.defaultdict(set)
    for r in rows:
        out[norm(r[question_key])].add(r[label_key])
    return out


def joined_type(question, index):
    labels = index.get(norm(question), set())
    if not labels:
        return 'pending', 'no_exact_release_question_match', None
    if len(labels) != 1:
        return 'pending', 'conflicting_release_labels', sorted(labels)
    label = next(iter(labels))
    status, reason = typed(label)
    return status, reason, label


class Evidence:
    def __init__(self, cache):
        self.cache = cache
        self.files = {}

    def file(self, path, url=None):
        path = Path(path)
        key = str(path)
        if key not in self.files:
            self.files[key] = {'path': key, 'sha256': digest(path)}
            if url:
                self.files[key]['url'] = url
        return self.files[key]

    def download(self, name, url):
        path = self.cache / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=40) as response:
                content = response.read()
            # Cache an unchanged, pinned official release, not generated questions.
            with path.open('xb') as f:
                f.write(content)
        ref = self.file(path, url)
        s = path.read_text()
        try:
            rows = json.loads(s)
        except json.JSONDecodeError:
            rows = [json.loads(l) for l in s.splitlines() if l.strip()]
        return rows, ref


def classify_bank(root, bank, evidence):
    from optmath_reviewed_scope import classify_optmath
    optmath_decisions = classify_optmath(bank, evidence)
    opti, opti_ref = evidence.download('optibench_official.json', OPTIBENCH_URL)
    for r in opti:
        r['_linear_label'] = {'linear-table': 'MILP', 'linear-notable': 'MILP',
                              'nonlinear-table': 'NLP', 'nonlinear-notable': 'NLP'}.get(r['type'], r['type'])
    opti_index = text_index(opti, 'question', '_linear_label')
    industry, industry_ref = evidence.download('industryor_original.jsonl', INDUSTRY_URL)
    industry_index = text_index(industry, 'en_question', 'question_type')
    bw_path = root / 'external/BWOR/data/test.jsonl'
    lo_path = root / 'external/LogiOR/logior.jsonl'
    bw = {r['id']: r for r in read_rows(bw_path)}
    lo = {r['prob_id']: r for r in read_rows(lo_path)}
    decisions = []
    for row in bank:
        source = row['source']
        status, reason, label = 'pending', 'missing_per_problem_type_evidence', None
        refs = []
        if source in EXCLUDED_SOURCES:
            status, reason = 'excluded', 'user_excluded_source'
        elif source == 'optmath_bench':
            decisions.append(optmath_decisions[row['id']])
            continue
        elif source in RELEASE_SCOPE:
            desc, url = RELEASE_SCOPE[source]
            status, reason, label = 'eligible', 'documented_linear_release', 'LP_or_MILP'
            refs = [{'url': url, 'scope': desc, 'bank_sha256': BANK_HASH}]
        elif source in {'bwor', 'logior'}:
            official = (bw if source == 'bwor' else lo)[row['native_id']]
            field = 'en_question' if source == 'bwor' else 'description'
            refs = [evidence.file(bw_path if source == 'bwor' else lo_path)]
            if norm(row['question']) == norm(official[field]):
                label = official['problem_type']
                status, reason = typed(label)
            else:
                reason = 'release_question_mismatch'
        elif source == 'optibench':
            status, reason, label = joined_type(row['question'], opti_index)
            refs = [opti_ref]
        elif source == 'industryor':
            status, reason, label = joined_type(row['question'], industry_index)
            refs = [industry_ref]
        elif source.startswith('text2opt_'):
            folder = 'Template' if source == 'text2opt_template' else 'Unstructured/resource_allocation/eval'
            path = root / 'external/Text2Opt-Bench/synthetic_dataset' / folder / row['native_id']
            refs = [evidence.file(path)]
            data = json.loads(path.read_text())
            label = data.get('meta', {}).get('optimization_type')
            descriptions = [data.get(k) for k in ('LLM_description', 'description')]
            if norm(row['question']) not in {norm(s) for s in descriptions if isinstance(s, str)}:
                reason = 'release_question_mismatch'
            else:
                status, reason = text2opt_type(data)
        decisions.append({'id': row['id'], 'source': source, 'status': status,
                          'reason': reason, 'released_type': label, 'evidence': refs})
    return decisions


def native_inventory(root, evidence):
    records = []
    extra_files = []
    path = root / 'external/DCP-Bench-Open/dcp-bench-open.jsonl'
    ref = evidence.file(path)
    for r in read_rows(path):
        records.append({'id': 'dcp:' + r['id'], 'source': 'dcp', 'status': 'pending',
                        'reason': 'native_cp_interface_requires_milp_translation', 'evidence': [ref]})
    path = root / 'external/FrontierOR/paper_meta_info.json'
    ref = evidence.file(path)
    meta = {r['paper_id']: r for r in json.loads(path.read_text())}
    # The existing native protocol has one tiny and five named large slots per
    # released family; preserve other on-disk files in a separate inventory.
    names = {'tiny_instance.json'} | {f'large_instance_{i}.json' for i in range(1, 6)}
    for p in sorted((root / 'external/FrontierOR/frontier-or').glob('*/instance/*.json')):
        family = p.parent.parent.name
        if family not in meta or p.name not in names:
            extra_files.append({'path': str(p), 'reason': 'outside_existing_native_instance_protocol'})
            continue
        label = meta.get(family, {}).get('formulation_type')
        status = 'excluded' if label in {'QP/MIQP', 'SDP', 'NLP/MINLP'} else 'pending'
        reason = ('released_nonlinear_or_conic_formulation' if status == 'excluded'
                  else 'native_reference_requires_complete_linear_capture')
        records.append({'id': f'frontieror:{family}:{p.stem}', 'source': 'frontieror',
                        'status': status, 'reason': reason, 'released_type': label,
                        'instance_path': str(p), 'evidence': [ref]})
    unique_ids(records)
    counts = collections.Counter(r['source'] for r in records)
    if counts != {'dcp': 164, 'frontieror': 1072}:
        raise ValueError(f'Native inventory changed: {counts}; review before activation')
    return records, extra_files


def write_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def write_rows(path, rows):
    with path.open('x') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def verify_scope(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    complete = json.loads((folder / 'COMPLETE.json').read_text())
    if complete['manifest_sha256'] != digest(folder / 'manifest.json'):
        raise ValueError('manifest hash mismatch')
    for name, expected in manifest['output_sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError('hash mismatch: ' + name)
    input_bank = Path(manifest['input_bank'])
    if input_bank.is_file():
        if digest(input_bank) != manifest['input_sha256']:
            raise ValueError('original bank changed')
        bank = read_rows(input_bank)
    else:
        # The public core package contains the frozen partition and its hashes,
        # but not the larger source repository from which it was selected.
        bank = []
        for status in ('eligible', 'excluded', 'pending'):
            bank.extend(read_rows(folder / (status + '.jsonl')))
    for ref in json.loads((folder / 'provenance.json').read_text()):
        path = Path(ref['path'])
        if path.is_file() and digest(path) != ref['sha256']:
            raise ValueError('reference evidence changed: ' + ref['path'])
    decisions = read_rows(folder / 'decisions.jsonl')
    if unique_ids(decisions) != unique_ids(bank):
        raise ValueError('decision IDs do not partition original bank')
    byid = {r['id']: r for r in bank}
    for status in ('eligible', 'excluded', 'pending'):
        part = read_rows(folder / (status + '.jsonl'))
        if unique_ids(part) != {r['id'] for r in decisions if r['status'] == status}:
            raise ValueError('incorrect status partition: ' + status)
        if any(row != byid[row['id']] for row in part):
            raise ValueError('source question or data was changed')
    if len(decisions) != manifest['total'] or len(read_rows(folder / 'eligible.jsonl')) != manifest['eligible']:
        raise ValueError('count mismatch')
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=ACTIVE_SCOPE)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--resume-build', action='store_true',
                    help='Resume only a failed pre-output build containing downloaded evidence alone')
    args = ap.parse_args()
    if args.verify:
        m = verify_scope(args.out)
        print(json.dumps({'verified': True, 'eligible': m['eligible'], 'by_source': m['by_source']}, indent=2))
        return
    bank_path = args.root / 'data/paper_benchmark_items_v2.jsonl'
    if digest(bank_path) != BANK_HASH:
        raise ValueError('Input bank changed; source scope must be reviewed')
    bank = read_rows(bank_path)
    unique_ids(bank)
    if args.out.exists():
        if not args.resume_build or {p.name for p in args.out.iterdir()} != {'source_evidence'}:
            raise FileExistsError('Refusing to replace an existing scope; verify it or create a new version')
    else:
        args.out.mkdir(parents=True)
    cache = args.out / 'source_evidence'
    cache.mkdir(exist_ok=args.resume_build)
    evidence = Evidence(cache)
    decisions = classify_bank(args.root, bank, evidence)
    native, extra_files = native_inventory(args.root, evidence)
    status = {r['id']: r['status'] for r in decisions}
    write_rows(args.out / 'decisions.jsonl', decisions)
    write_rows(args.out / 'native_inventory.jsonl', native)
    write_json(args.out / 'native_additional_files.json', extra_files)
    for name in ('eligible', 'excluded', 'pending'):
        write_rows(args.out / (name + '.jsonl'), (r for r in bank if status[r['id']] == name))
    write_json(args.out / 'eligible_ids.json', [r['id'] for r in bank if status[r['id']] == 'eligible'])
    write_json(args.out / 'provenance.json', list(evidence.files.values()))
    sources = sorted({r['source'] for r in bank})
    summary = {s: {k: sum(r['source'] == s and r['status'] == k for r in decisions)
                   for k in ('eligible', 'excluded', 'pending')} for s in sources}
    native_counts = {s: dict(collections.Counter(r['status'] for r in native if r['source'] == s))
                     for s in ('dcp', 'frontieror')}
    outputs = [p for p in args.out.iterdir() if p.is_file()] + list(cache.iterdir())
    manifest = {
        'schema': 'orsmt-milp-scope-v3', 'input_bank': str(bank_path), 'input_sha256': BANK_HASH,
        'total': len(bank), 'eligible': sum(r['status'] == 'eligible' for r in decisions),
        'scope': 'LP, integer linear and mixed-integer linear formulations; single linear objective',
        'selection_basis': 'reference matrices, reviewed generators, or original release type evidence',
        'not_used_for_selection': ['LLM checkpoint', 'generated code', 'verification verdict',
                                   'answer availability', 'objective agreement', 'runtime', 'problem size'],
        'candidate_support': 'Scope admission is not candidate correctness or successful translation. '
                             'Retain all selected IDs, including missing or unsupported LLM outputs.',
        'native_policy': 'Keep all native IDs inventoried. No native instance enters the active '
                         'population until its complete supported MILP representation is available.',
        'by_source': summary, 'native_inventory': native_counts,
        'reasons': dict(collections.Counter(r['reason'] for r in decisions)),
        'output_sha256': {str(p.relative_to(args.out)): digest(p) for p in outputs},
    }
    write_json(args.out / 'manifest.json', manifest)
    write_json(args.out / 'COMPLETE.json', {'manifest_sha256': digest(args.out / 'manifest.json')})
    verify_scope(args.out)
    print(json.dumps({'eligible': manifest['eligible'], 'by_source': summary,
                      'native_inventory': native_counts}, indent=2))


if __name__ == '__main__':
    main()
