import unittest
from dataclasses import replace
from fractions import Fraction
from orsmt.ir import ModelIR, Variable, Objective, LinearExpr, LinearConstraint
from orsmt.verify import Verifier
from orsmt.evidence import Verdict
from orsmt.repair_workflow import diagnose_repair


def model(scale=1, offset=0, sense='min'):
    return ModelIR((Variable('x', 'integer', 0, 10),), (),
                   Objective(sense, LinearExpr.from_mapping({'x': scale}, offset)))


class ObjectiveValueTests(unittest.TestCase):
    def test_order_is_not_value_equivalence(self):
        a, b = model(), model(2, 3)
        v = Verifier()
        self.assertEqual(v.preference_equivalence(a, b).verdict, Verdict.CERTIFIED)
        self.assertEqual(v.objective_value_equivalence(a, b).verdict, Verdict.REFUTED)
        self.assertEqual(diagnose_repair(a, b, written_is_specification=True).target, 'program')

    def test_explicit_unit_map(self):
        row = Verifier().objective_value_equivalence(model(), model(2, 3),
                    right_scale=Fraction(1, 2), right_offset=Fraction(-3, 2))
        self.assertEqual(row.verdict, Verdict.CERTIFIED)

    def test_constraint_implied_value_identity(self):
        a, b = model(), model(2)
        constraint = LinearConstraint('zero', LinearExpr.from_mapping({'x': 1}), '==', 0)
        a, b = replace(a, constraints=(constraint,)), replace(b, constraints=(constraint,))
        self.assertEqual(Verifier().objective_value_equivalence(a, b).verdict, Verdict.CERTIFIED)

    def test_no_certificate_on_empty_shared_region(self):
        a = replace(model(), variables=(Variable('x', 'integer', 0, 0),))
        b = replace(model(), variables=(Variable('x', 'integer', 1, 1),))
        self.assertEqual(Verifier().objective_value_equivalence(a, b).verdict, Verdict.UNDECIDED)

    def test_constant_offset_is_detected(self):
        self.assertEqual(Verifier().objective_value_equivalence(model(), model(1, 2)).verdict, Verdict.REFUTED)

    def test_zero_scale_rejected(self):
        with self.assertRaises(ValueError):
            Verifier().objective_value_equivalence(model(), model(), right_scale=0)
