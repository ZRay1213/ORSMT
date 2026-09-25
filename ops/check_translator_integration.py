"""Real-candidate format roundtrip and taxonomy integration, with no LLM calls."""
import argparse
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'src'))
from build_milp_benchmark import digest, write_json
from run_routed_milp_repair import native_executor
from orsmt.adapters import GurobiReplayAdapter
from orsmt.evidence import aggregate, Verdict
from orsmt.milp_codegen import emit_gurobi
from orsmt.repair_workflow import compare_models
from orsmt.translator import ModelTranslator
from orsmt.workflow import ORSMTWorkflow
from orsmt.verify import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trace', type=Path, default=PACKAGE / 'data/worked_trace/nl4opt67')
    args = ap.parse_args()
    marker = json.loads((args.trace/'COMPLETE.json').read_text())
    for name, expected in marker['output_sha256'].items():
        if digest(args.trace/name) != expected:
            raise ValueError('worked trace changed')
    source = json.loads((args.trace/'01_input.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    translator = ModelTranslator()
    original = translator.translate(source['formulation'])
    env = {'__name__': 'format_export_only'}
    exec(emit_gurobi(original.model), env)
    native, _ = env['build_model']()
    native.write(str(args.out/'written.lp'))
    native.write(str(args.out/'written.mps'))
    v = Verifier(native_replay=GurobiReplayAdapter())
    runtime = native_executor()(source['code'])
    workflow = ORSMTWorkflow(translator=translator, verifier=v)
    inputs = {'json': json.dumps(source['formulation']), 'lp': args.out/'written.lp',
              'mps': args.out/'written.mps', 'gurobi': native}
    fingerprints, effects = {}, {}
    for fmt, artifact in inputs.items():
        translated = translator.translate(artifact, format=fmt)
        assert aggregate(list(compare_models(v, original.model, translated.model))) == Verdict.CERTIFIED
        report = workflow.check(case_id=source['id'], written=artifact, written_format=fmt,
                                program=runtime.model, result=runtime.result)
        write_json(args.out/(fmt+'_feedback.json'), report)
        fingerprints[fmt] = translated.metadata()['model_sha256']
        effects[fmt] = sorted({f['category'] for f in report['feedback']['findings']})
    native.dispose()
    assert all(x == effects['json'] for x in effects.values())
    summary = {'case': source['id'], 'type': 'real_candidate_format_roundtrip_not_generation_benchmark',
               'formats': list(inputs), 'all_format_equivalence_checks': 'CERTIFIED',
               'same_diagnostic_effects': True, 'effects': effects['json'],
               'model_fingerprints': fingerprints, 'llm_calls': 0, 'gpu_loaded': False,
               'taxonomy_metrics': report['feedback']['metrics']}
    write_json(args.out/'summary.json', summary)
    files = [Path(__file__), Path(__file__).with_name('run_routed_milp_repair.py')]
    files += sorted(p for p in (PACKAGE/'src/orsmt').rglob('*.py') if not p.name.startswith('.'))
    write_json(args.out/'COMPLETE.json', {
        'input_trace_sha256': digest(args.trace/'COMPLETE.json'),
        'implementation_sha256': {str(p): digest(p) for p in files},
        'output_sha256': {p.name: digest(p) for p in args.out.iterdir() if p.is_file()}})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
