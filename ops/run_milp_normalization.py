"""Full 12-source x seven-checkpoint paired verification and redundancy audit.

CPU only. Each retained natural candidate is translated and executed unchanged.
This is a normalization/detection experiment, not an LLM repair evaluation.
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'src'))
from build_milp_benchmark import digest, read_rows, write_json
from milp_scope_policy import ACTIVE_SCOPE, verify_active_scope
from run_routed_milp_repair import TAGS, RESULTS, CAPTURE_MODULE, native_executor
from orsmt.adapters import DictModelAdapter, GurobiReplayAdapter
from orsmt.evidence import aggregate
from orsmt.milp_codegen import model_to_dict
from orsmt.normalization import audit_redundancy, normalize_constraints
from orsmt.repair_workflow import compare_models, diagnose_repair
from orsmt.translator import ModelTranslator
from orsmt.verify import Verifier


class CaseDeadline(BaseException):
    """Not swallowed by adapter exception-to-UNDECIDED handlers."""


def timeout_handler(*_):
    raise CaseDeadline('per-candidate wall-time budget')


def evaluate(task):
    tag, row, config = task
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(config['case_seconds'])
    started = perf_counter()
    out = {'id': row['id'], 'model': tag, 'source': row['id'].split(':')[0],
           'artifacts': {}, 'llm_calls': 0, 'gpu_calls': 0}
    verifier = Verifier(config['query_ms'], native_replay=GurobiReplayAdapter())
    models = {}
    try:
        try:
            written = ModelTranslator().translate(row.get('formulation'), format=row.get('formulation_format', 'auto'), allow_llm=False)
            models['written'] = written.model
            out['written_translation'] = written.metadata()
        except TimeoutError:
            raise
        except Exception as exc:
            out['written_status'] = 'unavailable'
            out['written_reason'] = f'{type(exc).__name__}: {exc}'
        try:
            captured = native_executor()(row.get('code'))
            models['program'] = captured.model
            out['result'] = None if captured.result is None else {
                'point': {k: str(v) for k, v in captured.result.point.items()},
                'value': None if captured.result.value is None else str(captured.result.value),
                'status': captured.result.status}
            out['reporting'] = [r.to_dict() for r in verifier.verify_result(captured.model, captured.result)] if captured.result else []
        except TimeoutError:
            raise
        except Exception as exc:
            out['program_status'] = 'unavailable'
            out['program_reason'] = f'{type(exc).__name__}: {exc}'
        if len(models) == 2:
            out['implementation_before'] = [r.to_dict() for r in compare_models(verifier, models['written'], models['program'])]
        normalized = {}
        for name, model in models.items():
            result = out['artifacts'][name] = {'variables': len(model.variables), 'constraints': len(model.constraints),
                'integer_variables': sum(v.kind != 'continuous' for v in model.variables), 'original_model': model_to_dict(model)}
            # Alternate measured ordering to avoid always warming the same backend.
            order = ['incremental', 'rebuild'] if sum(row['id'].encode()) % 2 == 0 else ['rebuild', 'incremental']
            result['timing_order'] = order
            for mode in order:
                result[mode] = audit_redundancy(model, verifier, incremental=mode == 'incremental', budget_seconds=config['phase_seconds'])
            result['normalization'] = normalize_constraints(model, verifier, budget_seconds=config['phase_seconds'])
            normalized[name] = DictModelAdapter().to_ir(result['normalization']['model'])
            result['individual_query_agreement'] = sum(a['redundant'] == b['redundant'] for a,b in
                zip(result['incremental']['rows'], result['rebuild']['rows']))
        if len(normalized) == 2:
            out['implementation_after'] = [r.to_dict() for r in compare_models(verifier, normalized['written'], normalized['program'])]
        out['status'] = 'evaluated'
    except (CaseDeadline, TimeoutError) as exc:
        out.update(status='time_budget', reason=str(exc))
    except Exception as exc:
        out.update(status='evaluation_error', reason=f'{type(exc).__name__}: {exc}')
    finally:
        signal.alarm(0)
    out['elapsed_seconds'] = perf_counter()-started
    return out


def locked_plan(out, config):
    scope = verify_active_scope()
    bank = read_rows(ACTIVE_SCOPE/'eligible.jsonl')
    ids = {r['id'] for r in bank}
    inputs = {str(ACTIVE_SCOPE/'manifest.json'): digest(ACTIVE_SCOPE/'manifest.json'),
              str(ACTIVE_SCOPE/'eligible.jsonl'): digest(ACTIVE_SCOPE/'eligible.jsonl')}
    paths = {}
    for tag in TAGS:
        mp = RESULTS/'modelcode_exact_domains_v2'/f'mc_{tag}.manifest.json'
        manifest = json.loads(mp.read_text())
        if not manifest.get('complete'):
            raise ValueError('initial candidate manifest is incomplete')
        candidates = [Path(p) for p in manifest['inputs'] if '/data/bank13_dual_v3/' in p]
        if len(candidates) != 1:
            raise ValueError('ambiguous initial candidate file')
        p = candidates[0]
        if digest(p) != manifest['inputs'][str(p)]:
            raise ValueError('initial candidate hash changed')
        selected = [r['id'] for r in read_rows(p) if r['id'] in ids]
        if len(selected) != len(ids) or set(selected) != ids:
            raise ValueError('candidate scope is not exact and unique')
        paths[tag] = str(p)
        inputs[str(p)] = digest(p)
        inputs[str(mp)] = digest(mp)
    files = list((PACKAGE/'src/orsmt').rglob('*.py'))
    files += [Path(__file__), PACKAGE/'ops/build_milp_benchmark.py', PACKAGE/'ops/milp_scope_policy.py',
              PACKAGE/'ops/run_routed_milp_repair.py', CAPTURE_MODULE, CAPTURE_MODULE.parent/'capture_worker.py']
    files = [p for p in files if not p.name.startswith('._')]
    plan = {'schema': 'orsmt-milp-normalization-experiment-v1', 'config': config,
            'models': list(TAGS), 'problems': len(ids), 'assigned_outputs': len(ids)*len(TAGS),
            'sources': dict(sorted(collections.Counter(r['source'] for r in bank).items())),
            'scope_sha256': inputs[str(ACTIVE_SCOPE/'manifest.json')], 'inputs_sha256': inputs,
            'source_sha256': {str(p): digest(p) for p in files}, 'llm_calls': 0, 'gpu_devices': []}
    pp = out/'PLAN.json'
    if pp.exists() and json.loads(pp.read_text()) != plan:
        raise ValueError('plan differs; use a new output directory')
    if not pp.exists():
        write_json(pp, plan)
    return plan, ids, paths


def summarize(rows):
    result = {'assigned_outputs': len(rows), 'status': dict(collections.Counter(r['status'] for r in rows))}
    for artifact in ('written', 'program'):
        available = [r['artifacts'][artifact] for r in rows if artifact in r['artifacts']]
        audited = [r for r in available if 'incremental' in r]
        normal = [r['normalization'] for r in available if 'normalization' in r]
        result[artifact] = {'available_models': len(available), 'audited_models': len(audited),
            'constraints_in_available_models': sum(r['constraints'] for r in available),
            'individual_redundant_constraints': sum(r['incremental']['metrics']['true'] for r in audited),
            'individual_undecided_constraints': sum(r['incremental']['metrics']['undecided'] for r in audited),
            'normalizations_accepted': sum(r['accepted'] for r in normal),
            'constraints_removed': sum(r['metrics']['constraints_removed'] for r in normal),
            'irredundancy_certified_models': sum(r['metrics']['irredundancy_certified'] for r in normal)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=range(1,5))
    parser.add_argument('--query-ms', type=int, default=1000)
    parser.add_argument('--phase-seconds', type=int, default=10)
    parser.add_argument('--case-seconds', type=int, default=150)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if min(args.query_ms, args.phase_seconds, args.case_seconds) <= 0:
        parser.error('budgets must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'query_ms': args.query_ms, 'phase_seconds': args.phase_seconds, 'case_seconds': args.case_seconds,
              'workers': args.workers, 'native_replay': 'gurobi', 'order': 'original_rows', 'input_policy': 'unchanged_natural_candidates'}
    plan, ids, paths = locked_plan(args.out, config)
    print(json.dumps({'stage': 'plan_validated', 'assigned_outputs': plan['assigned_outputs'], 'workers': args.workers}), flush=True)
    if args.plan:
        return
    files = {tag: args.out/f'audit_{tag}.jsonl' for tag in TAGS}
    done = {}
    for tag, path in files.items():
        prior = read_rows(path) if path.exists() else []
        seen = {r['id'] for r in prior}
        if len(seen) != len(prior) or not seen <= ids or any(r['model'] != tag for r in prior):
            raise ValueError('invalid resume IDs')
        done[tag] = seen
    def tasks():
        for tag in TAGS:
            for row in read_rows(paths[tag]):
                if row['id'] in ids and row['id'] not in done[tag]:
                    yield tag, row, config
    streams = {tag: path.open('a') for tag, path in files.items()}
    pending_tasks = iter(tasks())
    completed = sum(map(len, done.values()))
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < args.workers*2:
                    task = next(pending_tasks, None)
                    if task is None:
                        exhausted = True
                        break
                    future = pool.submit(evaluate, task)
                    pending[future] = (task[0], task[1]['id'])
                if not pending:
                    break
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    tag, item_id = pending.pop(future)
                    result = future.result()
                    if result['id'] != item_id or result['model'] != tag:
                        raise ValueError('worker result identity differs')
                    streams[tag].write(json.dumps(result, ensure_ascii=False)+'\n')
                    streams[tag].flush()
                    os.fsync(streams[tag].fileno())
                    completed += 1
                    if completed % 25 == 0:
                        print(json.dumps({'completed': completed, 'assigned': plan['assigned_outputs'], 'model': tag}), flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    report, markers = {}, {}
    for tag, path in files.items():
        rows = read_rows(path)
        if len(rows) != len(ids) or {r['id'] for r in rows} != ids:
            raise ValueError('final scope not exact')
        report[tag] = {'all': summarize(rows), 'by_source': {s: summarize([r for r in rows if r['source'] == s]) for s in plan['sources']}}
        markers[path.name] = {'rows': len(rows), 'sha256': digest(path)}
    for p, expected in {**plan['inputs_sha256'], **plan['source_sha256']}.items():
        if digest(p) != expected:
            raise ValueError('input or source changed during execution')
    write_json(args.out/'summary.json', report)
    write_json(args.out/'COMPLETE.json', {'plan_sha256': digest(args.out/'PLAN.json'),
        'summary_sha256': digest(args.out/'summary.json'), 'outputs': markers,
        'assigned_outputs': completed, 'note': 'All assigned cases accounted for; undecided and unavailable are not passes.'})
    print(json.dumps({'stage': 'complete', 'assigned_outputs': completed}), flush=True)


if __name__ == '__main__':
    main()
