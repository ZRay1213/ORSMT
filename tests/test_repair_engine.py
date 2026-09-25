import unittest
from orsmt.ir import ModelIR, Variable, LinearConstraint, LinearExpr, Objective, ResultRecord
from orsmt.repair_engine import repair_case, ProgramExecution
from orsmt.milp_codegen import emit_gurobi, model_to_dict


def model(bound):
    return ModelIR((Variable('x', 'integer', 0, 10),),
                   (LinearConstraint('minimum', LinearExpr.from_mapping({'x': 1}), '>=', bound),),
                   Objective('min', LinearExpr.from_mapping({'x': 1})))


def execution(m, value):
    return ProgramExecution(m, None if value is None else ResultRecord({'x': value}, value, 'optimal'))


class EngineTests(unittest.TestCase):
    def invoke(self, written, initial, replacement, **kwargs):
        calls = []
        def execute(code):
            calls.append(code)
            if code == 'ORIGINAL':
                return initial
            self.assertEqual(code, emit_gurobi(replacement.model))
            return replacement
        out = repair_case(case_id='case', question='At least two items.', written=written,
                          code='ORIGINAL', execute=execute, **kwargs)
        return out, calls

    def test_compile_implementation_without_llm(self):
        out, calls = self.invoke(model(2), execution(model(1), 1), execution(model(2), 2),
                                 written_is_specification=True)
        self.assertTrue(out['accepted'])
        self.assertEqual(out['llm_calls'], 0)
        self.assertEqual(len(calls), 2)

    def test_reference_repairs_written_and_preserves_code(self):
        out, calls = self.invoke(model(3), execution(model(2), 2), execution(model(2), 2),
                                 reference=model(2), reference_origin='test_reference')
        self.assertTrue(out['accepted'])
        self.assertEqual(calls, ['ORIGINAL', 'ORIGINAL'])
        self.assertTrue(out['assessment']['formal_reference_agreement'])

    def test_both_wrong_even_when_they_agree(self):
        out, _ = self.invoke(model(3), execution(model(3), 3), execution(model(2), 2),
                             reference=model(2), reference_origin='test_reference')
        self.assertTrue(out['accepted'])
        self.assertEqual(out['assessment']['repair_target'], 'both')

    def test_no_implicit_written_truth(self):
        out, calls = self.invoke(model(3), execution(model(2), 2), execution(model(2), 2))
        self.assertEqual(out['status'], 'needs_source_review')
        self.assertEqual(len(calls), 1)

    def test_llm_source_review_revises_written_model(self):
        def propose(payload):
            self.assertNotIn('answer', payload)
            self.assertNotIn('expected', payload)
            return {'id': 'case', 'proposed_target': 'written_model',
                    'source_quotes': ['At least two items.'], 'reason': 'The lower bound is two.',
                    'formulation': model_to_dict(model(2))}
        out, calls = self.invoke(model(3), execution(model(2), 2), execution(model(2), 2), proposer=propose)
        self.assertTrue(out['accepted'])
        self.assertFalse(out['natural_language_verified'])
        self.assertEqual(out['llm_calls'], 1)
        self.assertEqual(calls, ['ORIGINAL', 'ORIGINAL'])

    def test_source_review_can_choose_program(self):
        def propose(payload):
            return {'id': 'case', 'proposed_target': 'program', 'source_quotes': ['two items'],
                    'reason': 'The model states the intended bound.'}
        out, calls = self.invoke(model(2), execution(model(1), 1), execution(model(2), 2), proposer=propose)
        self.assertTrue(out['accepted'])
        self.assertNotEqual(calls[-1], 'ORIGINAL')

    def test_invalid_quote_does_not_consume_more_than_budget(self):
        def propose(payload):
            return {'id': 'case', 'proposed_target': 'both', 'source_quotes': ['invented requirement'],
                    'reason': 'invalid', 'formulation': model_to_dict(model(2))}
        out, calls = self.invoke(model(3), execution(model(1), 1), execution(model(2), 2), proposer=propose)
        self.assertFalse(out['accepted'])
        self.assertEqual(out['llm_calls'], 1)
        self.assertEqual(len(calls), 1)

    def test_result_resolved_without_program_revision(self):
        out, calls = self.invoke(model(2), execution(model(2), 3), execution(model(2), 2))
        self.assertTrue(out['accepted'])
        self.assertEqual(out['code'], 'ORIGINAL')
        self.assertEqual(out['result']['objective_value'], '2')

    def test_missing_result_is_recovered(self):
        out, _ = self.invoke(model(2), execution(model(2), None), execution(model(2), 2))
        self.assertTrue(out['accepted'])
        self.assertEqual(out['assessment']['repair_target'], 'result')

    def test_already_valid_is_not_counted_as_a_repair(self):
        out, _ = self.invoke(model(2), execution(model(2), 2), execution(model(2), 2))
        self.assertEqual(out['status'], 'already_verified')
        self.assertFalse(out['accepted'])

    def test_failed_initial_execution_is_retained(self):
        def fail(code):
            raise RuntimeError('expected test failure')
        out = repair_case(case_id='case', question='text', written=model(2), code='ORIGINAL', execute=fail)
        self.assertEqual(out['id'], 'case')
        self.assertEqual(out['status'], 'initial_execution_unresolved')
