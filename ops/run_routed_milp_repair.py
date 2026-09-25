"""Full-scope resumable MILP repair runner; --plan does not execute candidates or LLMs."""
import argparse
import collections
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from build_milp_benchmark import digest, read_rows, write_json
from milp_scope_policy import ACTIVE_SCOPE, verify_active_scope
from orsmt.adapters import DictModelAdapter, DictResultAdapter, GurobiReplayAdapter
from orsmt.repair_engine import ProgramExecution, repair_case
from orsmt.verify import Verifier
from orsmt.translator import ModelTranslator

CORE_DATA = PACKAGE / 'data/paper_core'
CAPTURE_MODULE = PACKAGE / 'ops/runtime/capture_gurobi.py'
TAGS = ('15b', '3b', '7b', '14b', '32b', '72b')


def native_executor(capture_module=CAPTURE_MODULE, capture_function='capture'):
    capture_module = Path(capture_module)
    sys.path.insert(0, str(capture_module.parent))
    spec = importlib.util.spec_from_file_location('repair_host_capture', capture_module)
    if spec is None or spec.loader is None:
        raise ValueError(f'cannot load capture module: {capture_module}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    capture = getattr(module, capture_function)
    if not callable(capture):
        raise ValueError(f'capture function is not callable: {capture_function}')
    ma, ra = DictModelAdapter(), DictResultAdapter()
    def execute(code):
        if not isinstance(code, str) or not code.strip():
            raise ValueError('candidate has no executable program')
        cap = capture(code, timeout=45, solve_it=True)
        if cap.get('status') != 'CAPTURED' or not cap.get('ir'):
            raise ValueError('native capture: ' + str(cap.get('status')))
        if cap['ir'].get('n_quadratic', 0):
            raise ValueError('candidate objective is not linear')
        result = cap.get('returned')
        return ProgramExecution(ma.to_ir(cap['ir']), ra.to_result(result) if result else None)
    return execute


def llm_proposer(endpoint, model, max_tokens):
    if urlparse(endpoint).hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('This runner uses an explicitly configured server-local inference endpoint')
    def propose(payload):
        body = json.dumps({'model': model, 'temperature': 0, 'max_tokens': max_tokens,
                           'messages': [{'role': 'system', 'content': 'Return one JSON object. Interpret the original optimization requirements and correct the named discrepancy.'},
                                        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}).encode()
        req = urllib.request.Request(endpoint.rstrip('/') + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=180) as response:
            answer = json.load(response)
        text = answer['choices'][0]['message']['content'].strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        proposed = json.loads(text)
        proposed['usage'] = answer.get('usage')
        return proposed
    return propose


def summarize(rows):
    """Operational alignment counts, not benchmark objective accuracy."""
    before_refuted = [r for r in rows if r.get('history') and
                      r['history'][0].get('diagnosis', {}).get('implementation', {}).get('status') == 'REFUTED']
    return {'assigned_outputs': len(rows),
            'status': dict(collections.Counter(r['status'] for r in rows)),
            'accepted_revisions': sum(bool(r.get('accepted')) for r in rows),
            'initial_model_code_refutations': len(before_refuted),
            'accepted_after_initial_refutation': sum(bool(r.get('accepted')) for r in before_refuted),
            'accepted_by_target': dict(collections.Counter(
                r['assessment']['repair_target'] for r in rows if r.get('accepted'))),
            'llm_calls': sum(r.get('llm_calls', 0) for r in rows),
            'native_calls': sum(r.get('native_calls', 0) for r in rows),
            'elapsed_seconds': sum(r.get('elapsed_seconds', 0) for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['implementation_only', 'routed'], default='routed')
    ap.add_argument('--plan', action='store_true')
    ap.add_argument('--endpoint')
    ap.add_argument('--reviser-model')
    ap.add_argument('--max-llm-calls', type=int, default=1)
    ap.add_argument('--max-tokens', type=int, default=3072)
    ap.add_argument('--capture-module', type=Path, default=CAPTURE_MODULE)
    ap.add_argument('--capture-function', default='capture')
    args = ap.parse_args()
    if args.mode == 'routed' and not args.reviser_model:
        ap.error('routed plans need an explicit reviser model to freeze the run')
    if args.mode == 'routed' and not args.plan and not (args.endpoint and args.reviser_model):
        ap.error('routed execution needs an explicit local endpoint and reviser model; use --plan otherwise')
    if args.max_llm_calls < 0 or args.max_tokens < 1:
        ap.error('invalid budget')
    scope = verify_active_scope()
    bank = read_rows(ACTIVE_SCOPE / 'eligible.jsonl')
    allowed = {r['id'] for r in bank}
    questions = {r['id']: r['question'] for r in bank}
    inputs, candidates = {}, {}
    for tag in TAGS:
        path = CORE_DATA / 'candidates' / f'dual_{tag}.jsonl'
        rows = read_rows(path)
        if len(rows) != len(allowed) or {r['id'] for r in rows} != allowed:
            raise ValueError('candidate IDs not complete and unique')
        candidates[tag] = rows
        inputs[str(path)] = digest(path)
    files = [Path(__file__), Path(__file__).with_name('milp_scope_policy.py'),
             Path(__file__).with_name('build_milp_benchmark.py'),
             args.capture_module, args.capture_module.parent / 'capture_worker.py']
    files += sorted(p for p in (PACKAGE / 'src/orsmt').rglob('*.py') if not p.name.startswith('.'))
    plan = {'scope': str(ACTIVE_SCOPE), 'scope_sha256': digest(ACTIVE_SCOPE / 'manifest.json'),
            'problem_count': scope['eligible'], 'models': list(TAGS), 'assigned_outputs': len(allowed)*len(TAGS),
            'by_source': {k: v['eligible'] for k, v in scope['by_source'].items() if v.get('eligible')},
            'mode': args.mode, 'temperature': 0, 'max_rounds': 3,
            'max_llm_calls': args.max_llm_calls if args.mode == 'routed' else 0,
            'max_tokens': args.max_tokens if args.mode == 'routed' else 0,
            'reviser_model': args.reviser_model if args.mode == 'routed' else None,
            'inputs': inputs, 'implementation_sha256': {str(p): digest(p) for p in files}}
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out / '.runner.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This output directory already has an active runner')
    path = args.out / 'PLAN.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('existing plan differs; choose a new output directory')
    else:
        write_json(path, plan)
    if args.plan:
        print(json.dumps({'plan': str(path), 'problems': len(allowed), 'models': len(TAGS),
                          'assigned_outputs': len(allowed)*len(TAGS), 'gpu_loaded': False}, indent=2))
        return
    execute = native_executor(args.capture_module, args.capture_function)
    propose = llm_proposer(args.endpoint, args.reviser_model, args.max_tokens) if args.mode == 'routed' else None
    verifier = Verifier(timeout_ms=3000, native_replay=GurobiReplayAdapter())
    translator = ModelTranslator()
    all_rows = []
    for tag, rows in candidates.items():
        target = args.out / f'repair_{tag}.jsonl'
        prior = read_rows(target) if target.exists() else []
        checkpoint = args.out / f'COMPLETE_{tag}.json'
        if checkpoint.exists():
            frozen = json.loads(checkpoint.read_text())
            if frozen['sha256'] != digest(target) or frozen['plan_sha256'] != digest(path):
                raise ValueError('completed checkpoint hash mismatch')
            if len(prior) != len(allowed) or {r['id'] for r in prior} != allowed:
                raise ValueError('completed checkpoint ID set mismatch')
        done = {r['id'] for r in prior}
        if len(prior) != len(done) or not done <= allowed:
            raise ValueError('invalid existing result IDs')
        with target.open('a') as stream:
            for row in rows:
                if row['id'] in done:
                    continue
                try:
                    translated = translator.translate(row['formulation'], format=row.get('formulation_format', 'auto'))
                    out = repair_case(case_id=row['id'], question=questions[row['id']],
                                      written=translated.model, code=row.get('code'),
                                      execute=execute, proposer=propose, verifier=verifier,
                                      written_is_specification=args.mode == 'implementation_only',
                                      max_llm_calls=args.max_llm_calls)
                    out['input_translation'] = translated.metadata()
                except Exception as exc:
                    out = {'id': row['id'], 'accepted': False, 'status': 'input_unresolved',
                           'reason': f'{type(exc).__name__}: {exc}'}
                out.update(initial_model=tag, source=row.get('source', row['id'].split(':')[0]))
                out['inference_endpoint'] = args.endpoint if args.mode == 'routed' else None
                stream.write(json.dumps(out, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                done.add(row['id'])
                if len(done) % 25 == 0:
                    print(json.dumps({'model': tag, 'completed': len(done), 'total': len(allowed),
                                      'time': datetime.now(timezone.utc).isoformat()}), flush=True)
        final = read_rows(target)
        if len(final) != len(allowed) or {r['id'] for r in final} != allowed:
            raise ValueError('incomplete final ID set')
        marker = {'rows': len(final), 'sha256': digest(target), 'plan_sha256': digest(path)}
        if checkpoint.exists():
            if json.loads(checkpoint.read_text()) != marker:
                raise ValueError('completed checkpoint changed')
        else:
            write_json(checkpoint, marker)
        print(tag, dict(collections.Counter(r['status'] for r in final)), flush=True)
        all_rows.extend(final)
    # Bind the final ledger to unchanged inputs and implementation, not just row counts.
    for filename, expected in {**inputs, **plan['implementation_sha256']}.items():
        if digest(filename) != expected:
            raise ValueError('an input or implementation changed during execution')
    verify_active_scope()
    if digest(ACTIVE_SCOPE / 'manifest.json') != plan['scope_sha256']:
        raise ValueError('scope changed during execution')
    report = {'definition': 'Alignment and repair counts; benchmark answer accuracy is evaluated separately.',
              'all': summarize(all_rows),
              'by_model': {tag: summarize([r for r in all_rows if r['initial_model'] == tag]) for tag in TAGS},
              'by_source': {source: summarize([r for r in all_rows if r['source'] == source])
                            for source in plan['by_source']}}
    report_path = args.out / 'summary.json'
    if report_path.exists():
        if json.loads(report_path.read_text()) != report:
            raise ValueError('existing summary differs')
    else:
        write_json(report_path, report)
    marker = {'rows': len(all_rows), 'plan_sha256': digest(path), 'summary_sha256': digest(report_path),
              'outputs': {f'repair_{tag}.jsonl': digest(args.out / f'repair_{tag}.jsonl') for tag in TAGS},
              'checkpoints': {f'COMPLETE_{tag}.json': digest(args.out / f'COMPLETE_{tag}.json') for tag in TAGS}}
    complete = args.out / 'COMPLETE.json'
    if complete.exists():
        if json.loads(complete.read_text()) != marker:
            raise ValueError('overall completion marker changed')
    else:
        write_json(complete, marker)


if __name__ == '__main__':
    main()
