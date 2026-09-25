import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from orsmt.ir import ModelIR, Variable, Objective, LinearExpr
from orsmt.milp_codegen import model_to_dict
from orsmt.translator import ModelTranslator
from orsmt.constructive import model_fingerprint


def model():
    return ModelIR((Variable('x', 'integer', 0, 10),), (), Objective('min', LinearExpr.from_mapping({'x': 2}, 3)))


class TranslatorTests(unittest.TestCase):
    def test_ir_and_json_are_the_same_model(self):
        t = ModelTranslator()
        j = json.dumps(model_to_dict(model()))
        a, b = t.translate(model()), t.translate('```json\n'+j+'\n```')
        self.assertEqual(model_fingerprint(a.model), model_fingerprint(b.model))
        self.assertEqual(b.original, '```json\n'+j+'\n```')
        self.assertEqual(b.mechanism, 'mechanical')

    def test_path_has_explicit_reader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.json'
            p.write_text(json.dumps(model_to_dict(model())))
            self.assertEqual(ModelTranslator().translate(p).format, 'json')

    def test_free_latex_is_an_explicit_proposal(self):
        calls = []
        def propose(request):
            calls.append(request)
            return {'formulation': model_to_dict(model()), 'assumptions': [], 'translator_id': 'unit-test-mock'}
        out = ModelTranslator(llm_translate=propose).translate(r'\min 2x+3,\quad 0\le x\le10,\ x\in\mathbb{Z}', format='latex')
        self.assertEqual(out.mechanism, 'llm_proposed')
        self.assertFalse(out.metadata()['source_alignment_certified'])
        self.assertEqual(len(calls), 1)
        self.assertIn('without correcting', calls[0]['instruction'])

    def test_no_hidden_translation_llm_call(self):
        def fail(request):
            self.fail('callback should not run')
        with self.assertRaises(ValueError):
            ModelTranslator(llm_translate=fail).translate('text', format='latex', allow_llm=False)

    def test_python_requires_capture(self):
        with self.assertRaises(ValueError):
            ModelTranslator().translate('print(1)', format='python')

    def test_unknown_text_is_not_silently_reinterpreted(self):
        with self.assertRaises(ValueError):
            ModelTranslator().translate('produce at least three units')

    def test_custom_host_adapter(self):
        t = ModelTranslator()
        t.register('host-matrix', lambda payload: model())
        self.assertEqual(t.translate(object(), format='host-matrix').mechanism, 'host_adapter')
        with self.assertRaises(ValueError):
            t.register('json', lambda x: model())

    def test_incomplete_dictionary_cannot_be_an_empty_model(self):
        with self.assertRaises(ValueError):
            ModelTranslator().translate({'variables': [], 'objective': None})

    @unittest.skipUnless(importlib.util.find_spec('gurobipy'), 'native reader requires gurobipy')
    def test_native_lp(self):
        lp = 'Minimize\n obj: 2 x\nSubject To\n c: x >= 1\nBounds\n 0 <= x <= 10\nGenerals\n x\nEnd\n'
        out = ModelTranslator().translate(lp, format='lp')
        self.assertEqual(out.model.variable_map['x'].kind, 'integer')
        self.assertEqual(out.model.objective.expression.constant, 0)
        self.assertEqual(out.model.constraints[0].rhs, 1)

    @unittest.skipUnless(importlib.util.find_spec('gurobipy'), 'native reader requires gurobipy')
    def test_lp_bare_constant_is_not_silently_a_new_variable(self):
        lp = 'Minimize\n obj: 2 x + 3\nSubject To\n c: x >= 1\nEnd\n'
        with self.assertRaises(ValueError):
            ModelTranslator().translate(lp, format='lp')

    @unittest.skipUnless(importlib.util.find_spec('gurobipy'), 'native reader requires gurobipy')
    def test_mps_and_live_model_preserve_constant(self):
        import gurobipy as gp
        with tempfile.TemporaryDirectory() as d, gp.Env(params={'OutputFlag': 0}) as env:
            with gp.Model(env=env) as m:
                x = m.addVar(lb=0, ub=10, vtype=gp.GRB.INTEGER, name='x')
                m.setObjective(2*x+3)
                m.update()
                p = Path(d)/'model.mps'
                m.write(str(p))
                direct = ModelTranslator().translate(m, format='gurobi')
                parsed = ModelTranslator().translate(p)
                self.assertEqual(parsed.model.objective.expression.constant, 3)
                self.assertEqual(model_fingerprint(direct.model), model_fingerprint(parsed.model))

    @unittest.skipUnless(importlib.util.find_spec('gurobipy'), 'native reader requires gurobipy')
    def test_repair_accepts_lp_formulation_and_supplies_taxonomy(self):
        from orsmt.ir import LinearConstraint, ResultRecord
        from orsmt.repair_engine import repair_case, ProgramExecution
        obj = Objective('min', LinearExpr.from_mapping({'x': 1}))
        variables = (Variable('x', 'integer', 0, 10),)
        wrong = ModelIR(variables, (LinearConstraint('demand', LinearExpr.from_mapping({'x': 1}), '>=', 3),), obj)
        correct = ModelIR(variables, (LinearConstraint('demand', LinearExpr.from_mapping({'x': 1}), '>=', 2),), obj)
        def propose(payload):
            self.assertEqual(payload['feedback']['schema'], 'orsmt-feedback-v1')
            self.assertTrue(payload['feedback']['findings'])
            return {'id': 'unit', 'proposed_target': 'written_model', 'source_quotes': ['at least two'],
                    'reason': 'The demand is two.', 'formulation_format': 'lp',
                    'formulation': 'Minimize\n obj: x\nSubject To\n demand: x >= 2\nBounds\n 0 <= x <= 10\nGenerals\n x\nEnd\n'}
        out = repair_case(case_id='unit', question='Produce at least two items.', written=wrong,
                         code='ORIGINAL', execute=lambda code: ProgramExecution(correct, ResultRecord({'x': 2}, 2)),
                         proposer=propose)
        self.assertTrue(out['accepted'])
        self.assertEqual(out['code'], 'ORIGINAL')
        self.assertEqual(out['history'][0]['translation']['format'], 'lp')
        self.assertEqual(out['llm_calls'], 1)  # mocked proposer, not a model experiment

    @unittest.skipUnless(importlib.util.find_spec('gurobipy'), 'native reader requires gurobipy')
    def test_native_objectives_cannot_be_silently_dropped(self):
        import gurobipy as gp
        for kind in ('piecewise', 'quadratic', 'multiple'):
            with self.subTest(kind=kind), gp.Env(params={'OutputFlag': 0}) as env:
                with gp.Model(env=env) as m:
                    x = m.addVar(lb=0, ub=10, name='x')
                    if kind == 'piecewise':
                        m.setPWLObj(x, [0, 1, 10], [0, 1, 100])
                    elif kind == 'quadratic':
                        m.setObjective(x*x)
                    else:
                        m.setObjectiveN(x, 0)
                        m.setObjectiveN(-x, 1)
                    m.update()
                    with self.assertRaises(ValueError):
                        ModelTranslator().translate(m, format='gurobi')
