from fractions import Fraction
import unittest
from orsmt.ir import ModelIR, Variable, LinearConstraint, LinearExpr
from orsmt.witness_feedback import evaluate_point


class WitnessFeedbackTests(unittest.TestCase):
    def test_both_formulas_get_independent_arithmetic_evaluations(self):
        vs = (Variable('a', 'integer', 0), Variable('b', 'integer', 0))
        point = {'a': Fraction(15), 'b': Fraction(10)}
        wrong = ModelIR(vs, (LinearConstraint('share', LinearExpr.from_mapping({'b': 1, 'a': -1}), '>=', '2/5'),))
        right = ModelIR(vs, (LinearConstraint('share', LinearExpr.from_mapping({'b': '3/5', 'a': '-2/5'}), '>=', 0),))
        self.assertFalse(evaluate_point(wrong, point)['feasible'])
        self.assertTrue(evaluate_point(right, point)['feasible'])
        self.assertEqual(evaluate_point(wrong, point)['constraints'][0]['lhs'], '-5')
        self.assertEqual(evaluate_point(right, point)['constraints'][0]['lhs'], '0')

    def test_partial_assignment_is_not_inferred_feasible(self):
        model = ModelIR((Variable('x'),), ())
        self.assertIsNone(evaluate_point(model, {})['feasible'])
