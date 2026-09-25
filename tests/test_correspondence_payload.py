from __future__ import annotations

import unittest

from orsmt import (
    LinearConstraint,
    LinearExpr,
    ModelIR,
    Objective,
    ORSMTPlugin,
    Variable,
    Verdict,
    candidate_projection_json_schema,
    parse_candidate_projection,
)


def reference_model() -> ModelIR:
    return ModelIR(
        variables=(Variable("x", kind="integer", lb=0),),
        constraints=(
            LinearConstraint("cap", LinearExpr.from_mapping({"x": 1}), "<=", 4),
        ),
        objective=Objective("max", LinearExpr.from_mapping({"x": 1})),
    )


def split_model() -> ModelIR:
    return ModelIR(
        variables=(
            Variable("left", kind="integer", lb=0),
            Variable("right", kind="integer", lb=0),
        ),
        constraints=(
            LinearConstraint(
                "cap",
                LinearExpr.from_mapping({"left": 1, "right": 1}),
                "<=",
                4,
            ),
        ),
        objective=Objective(
            "max", LinearExpr.from_mapping({"left": 1, "right": 1})
        ),
    )


class CandidateProjectionPayloadTests(unittest.TestCase):
    def test_generation_schema_fixes_both_variable_vocabularies(self) -> None:
        schema = candidate_projection_json_schema(reference_model(), split_model())
        self.assertEqual(schema["required"], ["x"])
        self.assertFalse(schema["additionalProperties"])
        expression = schema["properties"]["x"]
        self.assertEqual(expression["required"], ["terms"])
        self.assertEqual(
            set(expression["properties"]["terms"]["properties"]),
            {"left", "right"},
        )
        self.assertFalse(
            expression["properties"]["terms"]["additionalProperties"]
        )

    def test_exact_json_payload_is_admitted_and_adjudicated(self) -> None:
        payload = {
            "x": {
                "terms": {"left": "1", "right": "1"},
                "constant": "0",
            }
        }
        proposal = parse_candidate_projection(reference_model(), split_model(), payload)
        self.assertTrue(proposal.returned)
        self.assertTrue(proposal.well_typed)
        report = ORSMTPlugin().audit_candidate_projection_payload(
            reference_model(), split_model(), payload=payload
        )
        self.assertEqual(report.profile.end_to_end, Verdict.CERTIFIED)
        self.assertTrue(report.metadata["proposal_well_typed"])

    def test_missing_or_unknown_variable_payload_abstains_before_smt(self) -> None:
        plugin = ORSMTPlugin()
        missing = plugin.audit_candidate_projection_payload(
            reference_model(), split_model(), payload=None
        )
        self.assertEqual(missing.profile.end_to_end, Verdict.UNDECIDED)
        self.assertEqual(
            missing.records[0].reason, "projection_proposal:not_returned"
        )

        unknown = plugin.audit_candidate_projection_payload(
            reference_model(),
            split_model(),
            payload={"x": {"terms": {"invented": 1}}},
        )
        self.assertEqual(unknown.profile.end_to_end, Verdict.UNDECIDED)
        self.assertIn("unknown_candidate_variable", unknown.records[0].reason)
        self.assertFalse(unknown.metadata["proposal_well_typed"])

    def test_constant_collapse_and_extra_fields_are_rejected(self) -> None:
        reference = reference_model()
        candidate = split_model()
        constant = parse_candidate_projection(
            reference, candidate, {"x": {"terms": {}, "constant": 0}}
        )
        self.assertFalse(constant.well_typed)
        self.assertEqual(constant.reason, "constant_only_expression:x")

        extra = parse_candidate_projection(
            reference,
            candidate,
            {"x": {"terms": {"left": 1}, "explanation": "trust me"}},
        )
        self.assertFalse(extra.well_typed)
        self.assertEqual(extra.reason, "expression_schema_mismatch:x")

    def test_composite_route_preserves_direct_and_proposal_evidence(self) -> None:
        payload = {
            "x": {
                "terms": {"left": "1", "right": "1"},
                "constant": "0",
            }
        }
        bundle = ORSMTPlugin().audit_reformulation_with_candidate_projection_payload(
            reference_model(), split_model(), payload=payload
        )
        self.assertTrue(bundle.route_eligible)
        self.assertEqual(bundle.direct.profile.end_to_end, Verdict.UNDECIDED)
        self.assertIsNotNone(bundle.proposal)
        self.assertEqual(bundle.effective.profile.end_to_end, Verdict.CERTIFIED)
        self.assertEqual(bundle.to_dict()["proposal"]["profile"]["end_to_end"], "CERTIFIED")

    def test_composite_route_never_overrides_a_direct_decision(self) -> None:
        bundle = ORSMTPlugin().audit_reformulation_with_candidate_projection_payload(
            reference_model(),
            reference_model(),
            payload={"x": {"terms": {"invented": 1}}},
        )
        self.assertFalse(bundle.route_eligible)
        self.assertIsNone(bundle.proposal)
        self.assertEqual(bundle.effective.profile.end_to_end, Verdict.CERTIFIED)


if __name__ == "__main__":
    unittest.main()
