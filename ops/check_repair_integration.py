"""Real-source integration checks; these records are not a repair benchmark trial."""
import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from build_milp_benchmark import digest, write_json
from run_routed_milp_repair import native_executor
from orsmt.adapters import DictModelAdapter, GurobiReplayAdapter
from orsmt.repair_engine import repair_case
from orsmt.repair_workflow import diagnose_repair
from orsmt.evidence import aggregate, Verdict
from orsmt.milp_codegen import emit_gurobi
from orsmt.verify import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trace', type=Path, default=PACKAGE / 'data/worked_trace/nl4opt67')
    args = ap.parse_args()
    trace = args.trace
    original = json.loads((trace / '01_input.json').read_text())
    reviewed = json.loads((trace / '04_reviewed_reference.json').read_text())
    marker = json.loads((trace / 'COMPLETE.json').read_text())
    for name, expected in marker['output_sha256'].items():
        if digest(trace / name) != expected:
            raise ValueError('original worked trace changed')
    args.out.mkdir(parents=True, exist_ok=False)
    adapter, execute = DictModelAdapter(), native_executor()
    v = Verifier(native_replay=GurobiReplayAdapter())
    written = adapter.to_ir(original['formulation'])
    reference = adapter.to_ir(reviewed['formulation'])
    base = dict(case_id=original['id'], question=original['question'], written=written,
                code=original['code'], execute=execute, verifier=v)
    unassigned = repair_case(**base)
    corrected = repair_case(**base, reference=reference, reference_origin=reviewed['origin'])
    implementation_only = repair_case(**base, written_is_specification=True)
    compiled_reference = emit_gurobi(reference)
    result = execute(compiled_reference)
    check = diagnose_repair(reference, result.model, result.result, written_is_specification=True, verifier=v)
    assert unassigned['status'] == 'needs_source_review'
    assert corrected['accepted'] and corrected['assessment']['repair_target'] == 'written_model'
    assert corrected['code'] == original['code']
    assert implementation_only['accepted']
    assert aggregate(list(check.implementation)) == Verdict.CERTIFIED
    assert aggregate(list(check.reporting)) == Verdict.CERTIFIED
    for name, data in [('without_reference', unassigned), ('reviewed_reference', corrected),
                       ('implementation_only', implementation_only), ('compiled_reference', check.to_dict())]:
        write_json(args.out / (name + '.json'), data)
    summary = {'case': original['id'], 'type': 'real_source_integration_not_benchmark_trial',
               'without_reference': unassigned['status'], 'reviewed_reference': corrected['status'],
               'selected_target': corrected['assessment']['repair_target'],
               'code_unchanged_for_model_repair': corrected['code'] == original['code'],
               'code_generation_roundtrip': 'CERTIFIED',
               'implementation_only_revision': implementation_only['status'],
               'llm_calls': 0, 'gpu_loaded': False,
               'reference_origin': reviewed['origin'], 'native_result': corrected['result']}
    write_json(args.out / 'summary.json', summary)
    files = sorted(p for p in (PACKAGE / 'src/orsmt').rglob('*.py') if not p.name.startswith('.')) + [Path(__file__), Path(__file__).with_name('run_routed_milp_repair.py')]
    write_json(args.out / 'COMPLETE.json', {'input_trace_sha256': digest(trace / 'COMPLETE.json'),
              'implementation_sha256': {str(p): digest(p) for p in files},
              'output_sha256': {p.name: digest(p) for p in args.out.iterdir() if p.is_file()}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
