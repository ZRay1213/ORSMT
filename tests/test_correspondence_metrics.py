from __future__ import annotations

import unittest

from orsmt import (
    Verdict,
    correspondence_proposal_profile,
    correspondence_route_profile,
)


class CorrespondenceProposalMetricTests(unittest.TestCase):
    def test_keeps_missing_and_ill_typed_proposals_in_denominator(self) -> None:
        profile = correspondence_proposal_profile(
            proposed=[True, True, False, True],
            well_typed=[True, False, False, True],
            verdicts=[
                Verdict.CERTIFIED,
                None,
                None,
                Verdict.REFUTED,
            ],
        )
        self.assertEqual(profile.routed_pairs, 4)
        self.assertEqual(profile.returned_pairs, 3)
        self.assertEqual(profile.missing_pairs, 1)
        self.assertEqual(profile.admitted_pairs, 2)
        self.assertEqual(profile.ill_typed_pairs, 1)
        self.assertEqual(profile.decided_pairs, 2)
        self.assertEqual(profile.certified_pairs, 1)
        self.assertEqual(profile.refuted_pairs, 1)
        self.assertEqual(profile.undecided_pairs, 0)
        self.assertAlmostEqual(profile.return_rate, 3 / 4)
        self.assertAlmostEqual(profile.proposal_coverage, 1 / 2)
        self.assertAlmostEqual(profile.well_typed_rate_returned, 2 / 3)
        self.assertAlmostEqual(profile.adjudication_decision_rate, 1)

    def test_rejects_mapper_supplied_decisions_without_a_valid_proposal(self) -> None:
        with self.assertRaises(ValueError):
            correspondence_proposal_profile(
                proposed=[False],
                well_typed=[False],
                verdicts=[Verdict.CERTIFIED],
            )

        with self.assertRaises(ValueError):
            correspondence_proposal_profile(
                proposed=[False],
                well_typed=[True],
                verdicts=[Verdict.UNDECIDED],
            )

        with self.assertRaises(ValueError):
            correspondence_proposal_profile(
                proposed=[True],
                well_typed=[True],
                verdicts=[None],
            )

    def test_route_profile_measures_incremental_decision_coverage(self) -> None:
        profile = correspondence_route_profile(
            direct_verdicts=[
                Verdict.CERTIFIED,
                Verdict.REFUTED,
                Verdict.UNDECIDED,
                Verdict.UNDECIDED,
                Verdict.UNDECIDED,
            ],
            route_eligible=[False, False, True, True, False],
            proposed=[False, False, True, True, False],
            well_typed=[False, False, True, True, False],
            proposal_verdicts=[
                None,
                None,
                Verdict.CERTIFIED,
                Verdict.UNDECIDED,
                None,
            ],
        )
        self.assertEqual(profile.route_eligible, 2)
        self.assertEqual(profile.recovered_certifications, 1)
        self.assertEqual(profile.recovered_refutations, 0)
        self.assertEqual(profile.effective_undecided, 2)
        self.assertAlmostEqual(profile.direct_decision_rate_all, 2 / 5)
        self.assertAlmostEqual(profile.effective_decision_rate_all, 3 / 5)
        self.assertAlmostEqual(profile.decision_coverage_gain, 1 / 5)

    def test_route_cannot_override_direct_or_bypass_admission(self) -> None:
        with self.assertRaises(ValueError):
            correspondence_route_profile(
                [Verdict.CERTIFIED], [True], [True], [True], [Verdict.REFUTED]
            )
        with self.assertRaises(ValueError):
            correspondence_route_profile(
                [Verdict.UNDECIDED], [False], [True], [True], [Verdict.CERTIFIED]
            )


if __name__ == "__main__":
    unittest.main()
