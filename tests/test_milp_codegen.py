import unittest
from fractions import Fraction
from orsmt.ir import ModelIR, Variable, Objective, LinearExpr, LinearConstraint
from orsmt.milp_codegen import emit_gurobi, model_to_dict
from orsmt.adapters.dict_ir import DictModelAdapter
from orsmt.constructive import model_fingerprint


class CodegenTests(unittest.TestCase):
    def test_schema_roundtrip_and_quoted_names(self):
        m = ModelIR((Variable("x'bad", 'integer', None, 10),), (),
                    Objective('max', LinearExpr.from_mapping({"x'bad": 2}, 3)))
        self.assertEqual(model_fingerprint(m), model_fingerprint(DictModelAdapter().to_ir(model_to_dict(m))))
        compile(emit_gurobi(m), '<test>', 'exec')
        self.assertIn('lb=-GRB.INFINITY', emit_gurobi(m))

    def test_nonterminating_constraint_is_scaled_exactly(self):
        m = ModelIR((Variable('x', 'continuous', Fraction(1, 3), None),),
                    (LinearConstraint('third', LinearExpr.from_mapping({'x': Fraction(1, 3)}), '<=', 2),),
                    Objective('min', LinearExpr.from_mapping({'x': 1})))
        code = emit_gurobi(m)
        self.assertIn('<= 6', code)
        self.assertIn('__bound_lb_x', code)

    def test_unrepresentable_objective_is_not_rounded(self):
        m = ModelIR((Variable('x'),), (), Objective('min', LinearExpr.from_mapping({'x': Fraction(1, 3)})))
        with self.assertRaises(ValueError):
            emit_gurobi(m)

    def test_binary_default_domain(self):
        m = ModelIR((Variable('x', 'binary'),), (), Objective('min', LinearExpr.from_mapping({'x': 1})))
        self.assertIn('lb=0, ub=1', emit_gurobi(m))
