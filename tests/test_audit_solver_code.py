from __future__ import annotations

import importlib.util
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "ops" / "audit_solver_code.py"


def load_script():
    spec = importlib.util.spec_from_file_location("audit_solver_code_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactBudgetTest(unittest.TestCase):
    def test_timeout_becomes_three_undecided_relation_records(self):
        module = load_script()
        module.ARTIFACT_TIMEOUT_S = 0.05
        module._audit_row_impl = lambda _row: time.sleep(1.0)

        row_id, report = module.audit_row({"id": "bench:1", "model_key": "14b"})

        self.assertEqual(row_id, "bench:1")
        self.assertEqual(len(report.records), 3)
        self.assertEqual({record.reason for record in report.records}, {"artifact_timeout"})
        self.assertEqual({record.verdict.value for record in report.records}, {"UNDECIDED"})
        self.assertEqual(report.profile.end_to_end.value, "UNDECIDED")


if __name__ == "__main__":
    unittest.main()
