from __future__ import annotations

import unittest

from orsmt import (
    Verdict,
    universal_claim_evaluation,
    universal_claim_profile,
    universal_decision_audit,
)


class UniversalClaimMetricTests(unittest.TestCase):
    def test_only_parametric_proofs_and_replayed_refutations_are_lifted(self) -> None:
        concrete = [
            Verdict.CERTIFIED,
            Verdict.REFUTED,
            Verdict.UNDECIDED,
            Verdict.REFUTED,
        ]
        profile = universal_claim_profile(
            concrete,
            replayed_refutations=[False, True, False, False],
            parametric_certificates=[False, False, True, False],
        )
        self.assertEqual(profile.total_claims, 4)
        self.assertEqual(profile.parametrically_certified_claims, 1)
        self.assertEqual(profile.refuted_by_replayed_instance_claims, 1)
        self.assertEqual(profile.unresolved_claims, 2)
        self.assertEqual(profile.fixed_instance_certified_not_lifted, 1)
        self.assertEqual(profile.unreplayed_refutations, 1)
        self.assertAlmostEqual(profile.decision_coverage_all, 1 / 2)

    def test_evaluation_separates_two_proof_directions_and_abstentions(self) -> None:
        evaluation = universal_claim_evaluation(
            ground_truth_valid=[True, False, True, False, True],
            concrete_verdicts=[
                Verdict.CERTIFIED,
                Verdict.REFUTED,
                Verdict.REFUTED,
                Verdict.CERTIFIED,
                Verdict.UNDECIDED,
            ],
            replayed_refutations=[False, True, True, False, False],
            parametric_certificates=[True, False, False, True, False],
        )
        self.assertEqual(evaluation.correct_certifications, 1)
        self.assertEqual(evaluation.false_certifications, 1)
        self.assertEqual(evaluation.correct_refutations, 1)
        self.assertEqual(evaluation.false_refutations, 1)
        self.assertEqual(evaluation.unresolved_valid_claims, 1)
        self.assertEqual(evaluation.unresolved_invalid_claims, 0)
        self.assertAlmostEqual(evaluation.valid_certification_recall, 1 / 3)
        self.assertAlmostEqual(evaluation.invalid_refutation_recall, 1 / 2)
        self.assertAlmostEqual(evaluation.accuracy_all, 2 / 5)
        self.assertAlmostEqual(evaluation.balanced_accuracy, 5 / 12)
        self.assertAlmostEqual(evaluation.unsafe_decision_rate_decided, 1 / 2)

    def test_rejects_contradictory_or_unreplayable_evidence(self) -> None:
        with self.assertRaises(ValueError):
            universal_claim_profile(
                [Verdict.CERTIFIED], replayed_refutations=[True]
            )
        with self.assertRaises(ValueError):
            universal_claim_profile(
                [Verdict.REFUTED],
                replayed_refutations=[True],
                parametric_certificates=[True],
            )

    def test_reported_labels_are_audited_separately_from_evidence(self) -> None:
        audit = universal_decision_audit(
            reported_verdicts=[
                Verdict.CERTIFIED,
                Verdict.REFUTED,
                Verdict.REFUTED,
                Verdict.UNDECIDED,
            ],
            concrete_verdicts=[
                Verdict.CERTIFIED,
                Verdict.UNDECIDED,
                Verdict.REFUTED,
                Verdict.REFUTED,
            ],
            replayed_refutations=[False, False, True, True],
            parametric_certificates=[True, False, False, False],
        )
        self.assertEqual(audit.supported_certifications, 1)
        self.assertEqual(audit.supported_refutations, 1)
        self.assertEqual(audit.unsupported_refutations, 1)
        self.assertEqual(audit.evidence_available_but_abstained, 1)
        self.assertEqual(audit.contradicted_decisions, 0)
        self.assertAlmostEqual(audit.supported_rate_reported, 2 / 3)

    def test_opposite_evidence_is_a_contradiction_not_unsupported(self) -> None:
        audit = universal_decision_audit(
            reported_verdicts=[Verdict.CERTIFIED, Verdict.REFUTED],
            concrete_verdicts=[Verdict.REFUTED, Verdict.UNDECIDED],
            replayed_refutations=[True, False],
            parametric_certificates=[False, True],
        )
        self.assertEqual(audit.contradicted_certifications, 1)
        self.assertEqual(audit.contradicted_refutations, 1)
        self.assertEqual(audit.unsupported_decisions, 0)
        self.assertEqual(audit.contradiction_rate_reported, 1.0)


if __name__ == "__main__":
    unittest.main()
