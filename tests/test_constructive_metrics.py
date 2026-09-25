from __future__ import annotations

import unittest

from orsmt import (
    ContractStatus,
    CounterpartStatus,
    ReformulationVerdict,
    constructive_contract_profile,
)


class ConstructiveContractMetricTests(unittest.TestCase):
    def test_separates_proposal_contract_and_model_evidence(self) -> None:
        profile = constructive_contract_profile(
            returned=[False, True, True, True],
            admitted=[False, False, True, True],
            contract_statuses=[None, None, ContractStatus.VALID, ContractStatus.INVALID],
            model_verdicts=[None, None, ReformulationVerdict.C, ReformulationVerdict.R],
            counterpart_statuses=[None, None, None, CounterpartStatus.ABSENT],
        )

        self.assertEqual(profile.routed_pairs, 4)
        self.assertEqual(profile.returned_pairs, 3)
        self.assertEqual(profile.admitted_pairs, 2)
        self.assertEqual(profile.rejected_pairs, 1)
        self.assertEqual(profile.contract_valid, 1)
        self.assertEqual(profile.contract_invalid, 1)
        self.assertEqual(profile.model_certified, 1)
        self.assertEqual(profile.model_refuted, 1)
        self.assertEqual(profile.model_undecided, 2)
        self.assertEqual(profile.counterpart_absent, 1)
        self.assertAlmostEqual(profile.model_decision_rate_all, 0.5)

    def test_bad_map_with_counterpart_stays_model_undecided(self) -> None:
        profile = constructive_contract_profile(
            returned=[True],
            admitted=[True],
            contract_statuses=[ContractStatus.INVALID],
            model_verdicts=[ReformulationVerdict.U],
            counterpart_statuses=[CounterpartStatus.EXISTS],
        )
        self.assertEqual(profile.counterpart_exists, 1)
        self.assertEqual(profile.model_refuted, 0)

    def test_refutation_requires_lifted_counterpart_absence(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks lifted"):
            constructive_contract_profile(
                returned=[True],
                admitted=[True],
                contract_statuses=[ContractStatus.INVALID],
                model_verdicts=[ReformulationVerdict.R],
                counterpart_statuses=[CounterpartStatus.EXISTS],
            )


if __name__ == "__main__":
    unittest.main()
