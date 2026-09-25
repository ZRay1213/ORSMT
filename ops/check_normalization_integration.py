"""Replay the retained NL4Opt-67 candidate; never treat this as a full experiment."""
import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'src'))
from build_milp_benchmark import digest, write_json
from run_routed_milp_repair import native_executor
from orsmt.adapters import DictModelAdapter, GurobiReplayAdapter
from orsmt.evidence import aggregate, Verdict
from orsmt.milp_codegen import emit_gurobi
from orsmt.normalization import normalize_constraints, audit_redundancy
from orsmt.repair_workflow import compare_models
from orsmt.translator import ModelTranslator
from orsmt.verify import Verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trace', type=Path, default=PACKAGE / 'data/worked_trace/nl4opt67')
    args = parser.parse_args()
    marker = json.loads((args.trace/'COMPLETE.json').read_text())
    for name, value in marker['output_sha256'].items():
        assert digest(args.trace/name) == value
    source = json.loads((args.trace/'01_input.json').read_text())
    execute = native_executor()
    original = {'written': ModelTranslator().translate(source['formulation']).model,
                'program': execute(source['code']).model}
    verifier = Verifier(3000, native_replay=GurobiReplayAdapter())
    args.out.mkdir(parents=True, exist_ok=False)
    reduced, report = {}, {}
    for name, model in original.items():
        one = audit_redundancy(model, verifier)
        rebuilt = audit_redundancy(model, verifier, incremental=False)
        assert [r['redundant'] for r in one['rows']] == [r['redundant'] for r in rebuilt['rows']]
        norm = normalize_constraints(model, verifier)
        assert norm['accepted'] and norm['metrics']['irredundancy_certified']
        reduced[name] = DictModelAdapter().to_ir(norm['model'])
        before = execute(emit_gurobi(model))
        after = execute(emit_gurobi(reduced[name]))
        assert before.result.value == after.result.value
        assert aggregate(list(compare_models(verifier, reduced[name], after.model))) == Verdict.CERTIFIED
        assert aggregate(list(verifier.verify_result(after.model, after.result))) == Verdict.CERTIFIED
        report[name] = {'individual_redundancy': one, 'rebuild': rebuilt, 'normalization': norm,
            'native_value_before': str(before.result.value), 'native_value_after': str(after.result.value)}
    before_relation = aggregate(list(compare_models(verifier, original['written'], original['program'])))
    after_relation = aggregate(list(compare_models(verifier, reduced['written'], reduced['program'])))
    assert before_relation == after_relation == Verdict.REFUTED
    summary = {'case': source['id'], 'role': 'real_candidate_integration_not_benchmark_result',
               'llm_calls': 0, 'gpu_calls': 0, 'model_code_relation_before': before_relation.value,
               'model_code_relation_after': after_relation.value, 'artifacts': report}
    write_json(args.out/'summary.json', summary)
    write_json(args.out/'COMPLETE.json', {'trace_sha256': digest(args.trace/'COMPLETE.json'),
        'outputs_sha256': {'summary.json': digest(args.out/'summary.json')},
        'source_sha256': {str(p): digest(p) for p in [Path(__file__), *(PACKAGE/'src/orsmt').rglob('*.py')] if not p.name.startswith('._')}})
    print(json.dumps({'case': source['id'], 'both_models': {name: {**r['normalization']['metrics'],
        'value_before': r['native_value_before'], 'value_after': r['native_value_after']} for name, r in report.items()},
        'disagreement_preserved': True}, indent=2))


if __name__ == '__main__':
    main()
