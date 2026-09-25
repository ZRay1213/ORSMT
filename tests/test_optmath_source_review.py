import copy
import importlib.util
import json
import unittest
from pathlib import Path

OPS = Path(__file__).resolve().parents[1] / 'ops'
SPEC = importlib.util.spec_from_file_location('optmath_reviewed_scope', OPS / 'optmath_reviewed_scope.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class OptMathReviewTests(unittest.TestCase):
    def setUp(self):
        self.review = json.loads(M.REVIEW_PATH.read_text())

    def test_complete_partition(self):
        ix = M.review_index(self.review)
        self.assertEqual(sum(v[2] == 'eligible' for v in ix.values()), 132)
        self.assertEqual(sum(v[2] == 'excluded' for v in ix.values()), 34)

    def test_missing_id_rejected(self):
        self.review['nonlinear']['official_ids'].pop()
        with self.assertRaises(ValueError):
            M.review_index(self.review)

    def test_duplicate_id_rejected(self):
        self.review['nonlinear']['official_ids'].append(0)
        with self.assertRaises(ValueError):
            M.review_index(self.review)

    def test_commit_change_rejected(self):
        self.review['commit'] = 'changed'
        with self.assertRaises(ValueError):
            M.review_index(self.review)
