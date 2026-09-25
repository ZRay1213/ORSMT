import importlib.util
import pathlib
import sys
import unittest

OPS = pathlib.Path(__file__).resolve().parents[1] / 'ops'
if not OPS.exists():
    OPS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
from build_milp_benchmark import joined_type, text_index, text2opt_type, typed, unique_ids


def matrix():
    return {'meta': {'optimization_type': 'MILP'}, 'quadratic_matrix_q': None,
            'constraints': {'c': {'type': 'Constraint', 'sense': '<=', 'rhs': 4}},
            'variables': {'x': {'type': 'Integer', 'range': [0, None],
                                'objective_linear_coefficient': 1, 'resource_costs': {'c': 2}}}}


class ScopeTests(unittest.TestCase):
    def test_linear_families(self):
        for label in ['LP', 'IP', 'ILP', 'MIP', 'MILP', 'BIP']:
            self.assertEqual(typed(label)[0], 'eligible')

    def test_nonlinear_families(self):
        for label in ['NLP', 'QP', 'MIQP', 'SOCP', 'SDP']:
            self.assertEqual(typed(label)[0], 'excluded')

    def test_unknown_is_not_nonlinear(self):
        for label in [None, 'Others', 'goal_programming', 'CP']:
            self.assertEqual(typed(label)[0], 'pending')

    def test_unbounded_linear(self):
        d = matrix()
        d['variables']['x']['range'] = ['-inf', 'inf']
        self.assertEqual(text2opt_type(d)[0], 'eligible')

    def test_quadratic_overrules_linear_label(self):
        d = matrix()
        d['quadratic_matrix_q'] = [[1]]
        self.assertEqual(text2opt_type(d)[0], 'excluded')

    def test_symbolic_coefficient_is_pending(self):
        d = matrix()
        d['variables']['x']['resource_costs']['c'] = '2*x'
        self.assertEqual(text2opt_type(d)[0], 'pending')

    def test_nonfinite_coefficient_is_pending(self):
        d = matrix()
        d['constraints']['c']['rhs'] = float('nan')
        self.assertEqual(text2opt_type(d)[0], 'pending')

    def test_general_constraint_not_admitted(self):
        d = matrix()
        d['constraints']['c']['type'] = 'GeneralConstraint'
        self.assertEqual(text2opt_type(d)[0], 'pending')

    def test_semi_continuous_not_admitted(self):
        d = matrix()
        d['variables']['x']['type'] = 'SemiContinuous'
        self.assertEqual(text2opt_type(d)[0], 'pending')

    def test_join_does_not_guess(self):
        idx = text_index([{'q': 'a b', 'type': 'LP'}], 'q', 'type')
        self.assertEqual(joined_type('a\n b', idx)[0], 'eligible')
        self.assertEqual(joined_type('a c', idx)[0], 'pending')

    def test_conflicting_labels(self):
        idx = text_index([{'q': 'a', 'type': 'LP'}, {'q': 'a', 'type': 'NLP'}], 'q', 'type')
        self.assertEqual(joined_type('a', idx)[0], 'pending')

    def test_duplicates_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            unique_ids([{'id': 'x'}, {'id': 'x'}])

    def test_outcomes_and_size_do_not_select(self):
        d = matrix()
        d.update(answer=None, correct=False, runtime=1e12, model='small', nvars=10**9)
        self.assertEqual(text2opt_type(d)[0], 'eligible')


if __name__ == '__main__':
    unittest.main()
