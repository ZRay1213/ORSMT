import unittest
from dataclasses import replace
from orsmt.ir import LinearExpr, LinearConstraint, ModelIR, Objective, ResultRecord, Variable
from orsmt.repair_workflow import diagnose_repair, evaluate_revision, repair_prompt_payload


def model(bound=2):
    return ModelIR((Variable('x', 'integer', 0, 10),),
                   (LinearConstraint('minimum', LinearExpr.from_mapping({'x': 1}), '>=', bound),),
                   Objective('min', LinearExpr.from_mapping({'x': 1})))


class RepairWorkflowTests(unittest.TestCase):
    def test_disagreement_is_not_fault_ownership(self):
        self.assertEqual(diagnose_repair(model(3), model()).target, 'review')

    def test_program_is_correct(self):
        d = diagnose_repair(model(3), model(), reference=model(), reference_origin='reviewed')
        self.assertEqual(d.target, 'written_model')

    def test_written_is_correct(self):
        d = diagnose_repair(model(), model(3), reference=model(), reference_origin='reviewed')
        self.assertEqual(d.target, 'program')

    def test_both_can_be_wrong_even_when_equal(self):
        d = diagnose_repair(model(3), model(3), reference=model(), reference_origin='reviewed')
        self.assertEqual(d.target, 'both')

    def test_explicit_implementation_target(self):
        d = diagnose_repair(model(3), model(), written_is_specification=True)
        self.assertEqual(d.target, 'program')

    def test_reference_origin_required(self):
        with self.assertRaises(ValueError):
            diagnose_repair(model(), model(3), reference=model())

    def test_result_only(self):
        d = diagnose_repair(model(), model(), ResultRecord({'x': 3}, 3, 'optimal'))
        self.assertEqual(d.target, 'result')
        out = evaluate_revision(d, model(), model(), ResultRecord({'x': 2}, 2, 'optimal'))
        self.assertTrue(out['accepted'])

    def test_reference_based_model_fix(self):
        d = diagnose_repair(model(3), model(), reference=model(), reference_origin='reviewed')
        out = evaluate_revision(d, model(), model(), ResultRecord({'x': 2}, 2, 'optimal'), reference=model())
        self.assertTrue(out['accepted'])
        self.assertEqual(out['changed_artifacts'], ['written_model'])
        self.assertFalse(out['natural_language_verified'])

    def test_wrong_side_not_accepted_even_if_equal(self):
        d = diagnose_repair(model(3), model(), reference=model(), reference_origin='reviewed')
        out = evaluate_revision(d, model(3), model(3), ResultRecord({'x': 3}, 3, 'optimal'), reference=model())
        self.assertFalse(out['accepted'])
        self.assertTrue(out['model_code_consistent'])
        self.assertFalse(out['formal_reference_agreement'])

    def test_no_reference_cannot_autoadjudicate_review(self):
        d = diagnose_repair(model(3), model())
        out = evaluate_revision(d, model(), model(), ResultRecord({'x': 2}, 2, 'optimal'))
        self.assertFalse(out['accepted'])

    def test_changed_reference_rejected(self):
        d = diagnose_repair(model(3), model(), reference=model(), reference_origin='reviewed')
        with self.assertRaises(ValueError):
            evaluate_revision(d, model(3), model(3), reference=model(3))

    def test_incomplete_result_not_claimed_verified(self):
        d = diagnose_repair(model(3), model(), written_is_specification=True)
        out = evaluate_revision(d, model(3), model(3))
        self.assertTrue(out['model_code_consistent'])
        self.assertFalse(out['result_verified'])
        self.assertFalse(out['accepted'])

    def test_prompt_does_not_take_gold_label(self):
        p = repair_prompt_payload(diagnose_repair(model(3), model()), case_id='case',
                                  question='minimize x', formulation={}, code='example')
        self.assertNotIn('expected', p)
        self.assertNotIn('answer', p)
        self.assertEqual(p['action'], 'propose_fault_location_for_review')

    def test_variable_rename_is_not_guessed(self):
        other = ModelIR((Variable('y', 'integer', 0, 10),), (),
                        Objective('min', LinearExpr.from_mapping({'y': 1})))
        self.assertEqual(diagnose_repair(model(), other).target, 'review')


if __name__ == '__main__':
    unittest.main()
