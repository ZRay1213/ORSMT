from dataclasses import replace
import unittest

from orsmt.adapters import DictModelAdapter
from orsmt.evidence import Verdict, aggregate
from orsmt.ir import LinearConstraint, LinearExpr, ModelIR, Objective, Variable
from orsmt.normalization import audit_redundancy, normalize_constraints
from orsmt.repair_workflow import compare_models
from orsmt.verify import Verifier


def row(name, op, rhs, coefficient=1):
    return LinearConstraint(name, LinearExpr.from_mapping({'x': coefficient}), op, rhs)


def model(rows):
    return ModelIR((Variable('x', 'integer', 0, 10),), tuple(rows),
                   Objective('min', LinearExpr.from_mapping({'x': 2}, 7)))


class NormalizationTests(unittest.TestCase):
    def test_duplicate_rows_are_not_deleted_together(self):
        m = model([row('a', '>=', 2), row('b', '>=', 2)])
        self.assertEqual(audit_redundancy(m, Verifier())['metrics']['true'], 2)
        out = normalize_constraints(m, Verifier())
        self.assertTrue(out['accepted'])
        self.assertEqual(out['metrics']['constraints_removed'], 1)
        self.assertTrue(out['metrics']['irredundancy_certified'])

    def test_implicit_redundancy_and_objective_offset(self):
        m = model([row('lower', '>=', 2), row('weak', '>=', 1), row('bound', '<=', 10)])
        out = normalize_constraints(m, Verifier())
        normalized = DictModelAdapter().to_ir(out['model'])
        self.assertEqual(out['metrics']['constraints_after'], 1)
        self.assertEqual(normalized.objective.expression.constant, 7)
        self.assertEqual(aggregate(list(compare_models(Verifier(), m, normalized))), Verdict.CERTIFIED)
        self.assertEqual(audit_redundancy(normalized, Verifier())['metrics']['true'], 0)

    def test_infeasible_models_are_not_relabelled_normalized(self):
        m = model([row('a', '>=', 8), row('b', '<=', 2), row('extra', '>=', 0)])
        out = normalize_constraints(m, Verifier())
        self.assertFalse(out['accepted'])
        self.assertEqual(out['metrics']['constraints_removed'], 0)
        self.assertEqual(audit_redundancy(m, Verifier())['metrics']['undecided'], 3)

    def test_incremental_matches_rebuild_for_equalities_and_integer_domain(self):
        m = model([row('a', '>=', '3/2'), row('b', '>=', 2), row('c', '==', 4), row('d', '<=', 7)])
        a = audit_redundancy(m, Verifier(), incremental=True)
        b = audit_redundancy(m, Verifier(), incremental=False)
        self.assertEqual([r['redundant'] for r in a['rows']], [r['redundant'] for r in b['rows']])
        first = normalize_constraints(m, Verifier())
        second = normalize_constraints(m, Verifier(), incremental=False)
        self.assertEqual(first['model'], second['model'])

    def test_zero_budget_keeps_every_row(self):
        out = normalize_constraints(model([row('a', '>=', 2), row('b', '>=', 2)]), Verifier(), budget_seconds=0)
        self.assertEqual(out['metrics']['constraints_removed'], 0)
        self.assertEqual(out['metrics']['undecided_row_queries'], 2)
        self.assertFalse(out['metrics']['irredundancy_certified'])

    def test_empty_constraint_set(self):
        out = normalize_constraints(model([]), Verifier())
        self.assertTrue(out['accepted'])
        self.assertIsNone(out['metrics']['removed_percentage'])

    def test_rejected_final_verification_does_not_publish_deletions(self):
        class IncompleteVerifier(Verifier):
            def objective_value_equivalence(self, *args, **kwargs):
                return replace(super().objective_value_equivalence(*args, **kwargs), verdict=Verdict.UNDECIDED)
        out = normalize_constraints(model([row('a', '>=', 2), row('b', '>=', 2)]), IncompleteVerifier())
        self.assertFalse(out['accepted'])
        self.assertEqual(out['metrics']['constraints_removed'], 0)


if __name__ == '__main__':
    unittest.main()
