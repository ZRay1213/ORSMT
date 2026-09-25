import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ops'))
from run_routed_milp_repair import summarize, llm_proposer


class RepairRunnerTests(unittest.TestCase):
    def test_full_denominator_and_repair_population(self):
        rows = [
            {'status': 'input_unresolved'},
            {'status': 'already_verified', 'accepted': False},
            {'status': 'verified_revision', 'accepted': True, 'llm_calls': 1,
             'history': [{'diagnosis': {'implementation': {'status': 'REFUTED'}}}],
             'assessment': {'repair_target': 'written_model'}},
            {'status': 'needs_source_review', 'accepted': False,
             'history': [{'diagnosis': {'implementation': {'status': 'REFUTED'}}}]},
        ]
        result = summarize(rows)
        self.assertEqual(result['assigned_outputs'], 4)
        self.assertEqual(result['accepted_revisions'], 1)
        self.assertEqual(result['initial_model_code_refutations'], 2)
        self.assertEqual(result['accepted_after_initial_refutation'], 1)
        self.assertEqual(result['accepted_by_target'], {'written_model': 1})
        self.assertEqual(result['llm_calls'], 1)

    def test_inference_endpoint_must_be_server_local(self):
        with self.assertRaises(ValueError):
            llm_proposer('https://example.org/v1', 'example', 20)
