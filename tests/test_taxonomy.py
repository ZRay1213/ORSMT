from dataclasses import replace
import unittest
from orsmt.ir import ModelIR, Variable, LinearExpr, LinearConstraint, Objective, ResultRecord
from orsmt.verify import Verifier
from orsmt.taxonomy import analyze_case, row_redundancy
from orsmt.workflow import ORSMTWorkflow
from orsmt.milp_codegen import model_to_dict


def model(rows=(), kind='integer', scale=1, offset=0):
    return ModelIR((Variable('x', kind, 0, 10),), tuple(rows),
                   Objective('min', LinearExpr.from_mapping({'x': scale}, offset)))


def row(name, op, rhs):
    return LinearConstraint(name, LinearExpr.from_mapping({'x': 1}), op, rhs)


class TaxonomyTests(unittest.TestCase):
    def test_missing_constraint_and_redundancy_are_distinct(self):
        w = model([row('demand', '>=', 2), row('weak', '>=', 1)])
        p = model()
        out = analyze_case(w, p, None, verifier=Verifier())
        self.assertIn('feasible_region_expansion', [f['category'] for f in out['findings']])
        self.assertTrue(out['redundancy']['written'][1]['redundant'])
        self.assertTrue(out['redundancy']['written'][1]['normalization_issue'])
        self.assertIn('redundant_constraint', [f['category'] for f in out['findings']])
        self.assertEqual(out['metrics']['written_constraints_preserved_by_program']['false'], 2)

    def test_extra_constraint_restricts_region(self):
        out = analyze_case(model(), model([row('extra', '<=', 3)]), None, verifier=Verifier())
        self.assertIn('feasible_region_restriction', [f['category'] for f in out['findings']])

    def test_integrality_mismatch_has_a_real_witness(self):
        out = analyze_case(model(), model(kind='continuous'), None, verifier=Verifier())
        self.assertIn('variable_domain_mismatch', [f['category'] for f in out['findings']])

    def test_structural_domain_change_may_preserve_semantics(self):
        w = model([row('fixed', '==', 0)])
        p = model([row('fixed', '==', 0)], kind='continuous')
        out = analyze_case(w, p, None, verifier=Verifier())
        self.assertEqual(out['findings'], [])
        self.assertTrue(out['structural_observations'])

    def test_objective_and_result_types(self):
        out = analyze_case(model(), model(offset=3), ResultRecord({'x': 1}, 7), verifier=Verifier())
        kinds = {f['category'] for f in out['findings']}
        self.assertTrue({'objective_value_mismatch', 'reported_value_mismatch', 'suboptimal_solution'} <= kinds)

    def test_duplicate_redundancy_is_individual(self):
        m = model([row('a', '>=', 2), row('b', '>=', 2)])
        self.assertTrue(all(r['redundant'] for r in row_redundancy(m, Verifier())))

    def test_infeasible_rest_does_not_get_redundancy_credit(self):
        m = model([row('a', '>=', 20), row('b', '>=', 30), row('c', '>=', 40)])
        self.assertTrue(all(r['redundant'] is None for r in row_redundancy(m, Verifier())))

    def test_unknown_schema_stays_in_full_row_denominator(self):
        w = model([row('a', '>=', 2)])
        p = ModelIR((Variable('y', 'integer', 0, 10),), (), Objective('min', LinearExpr.from_mapping({'y': 1})))
        out = analyze_case(w, p, None, verifier=Verifier())
        self.assertEqual(out['metrics']['written_constraints_preserved_by_program']['total'], 1)
        self.assertEqual(out['metrics']['written_constraints_preserved_by_program']['undecided'], 1)

    def test_workflow_translates_independent_artifacts(self):
        w, p = model([row('demand', '>=', 2)]), model()
        out = ORSMTWorkflow().check(case_id='unit', written=model_to_dict(w), program=p)
        self.assertNotEqual(out['translations']['written']['model_sha256'], out['translations']['program']['model_sha256'])
        self.assertEqual(out['feedback']['schema'], 'orsmt-feedback-v1')
