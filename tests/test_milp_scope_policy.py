import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
from milp_scope_policy import ACTIVE_SCOPE, verify_active_scope


class ActiveScopeTests(unittest.TestCase):
    def test_reviewed_version_three_is_default(self):
        self.assertEqual(ACTIVE_SCOPE.name, 'milp_benchmark_v3')

    @patch('build_milp_benchmark.read_rows')
    @patch('build_milp_benchmark.verify_scope')
    def test_old_scope_cannot_reintroduce_engine(self, verify, rows):
        verify.return_value = {'eligible': 1}
        rows.return_value = [{'id': 'optengine_canonical:1', 'source': 'optengine_canonical'}]
        with self.assertRaises(ValueError):
            verify_active_scope(Path('/historical'))

    @patch('build_milp_benchmark.read_rows')
    @patch('build_milp_benchmark.verify_scope')
    def test_retained_source_passes(self, verify, rows):
        verify.return_value = {'eligible': 1}
        rows.return_value = [{'id': 'nl4opt_test:67', 'source': 'nl4opt_test'}]
        self.assertEqual(verify_active_scope()['eligible'], 1)


if __name__ == '__main__':
    unittest.main()
