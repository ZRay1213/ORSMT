from __future__ import annotations

import unittest

from orsmt import (
    EvidenceRecord,
    LinearExpr,
    ModelIR,
    Objective,
    ORSMTPlugin,
    Relation,
    SemanticView,
    Variable,
    Verdict,
    Verifier,
)


def model(name: str) -> ModelIR:
    return ModelIR(
        (Variable("x", "binary"),),
        (),
        Objective("max", LinearExpr.from_mapping({"x": 1})),
        name,
    )


def identity_payload():
    expression = {"affine": {"terms": {"x": 1}}}
    return {
        "forward": {"x": expression},
        "backward": {"x": expression},
        "objective": {"scale": 1, "offset": 0},
    }


class AbstainingVerifier(Verifier):
    def region_equivalence(self, left, right, relation=Relation.REFORMULATION):
        return (
            EvidenceRecord(relation, "left", Verdict.UNDECIDED, "left", reason="solver_unknown"),
            EvidenceRecord(relation, "right", Verdict.UNDECIDED, "right", reason="solver_unknown"),
        )

    def preference_equivalence(self, left, right, relation=Relation.REFORMULATION):
        return EvidenceRecord(
            relation,
            "preference",
            Verdict.UNDECIDED,
            "preference",
            reason="solver_unknown",
        )


class ConstructivePluginTests(unittest.TestCase):
    def test_standalone_route_records_schema_hashes_and_certifies(self) -> None:
        result = ORSMTPlugin().audit_constructive_reformulation_payload(
            model("left"),
            model("right"),
            payload=identity_payload(),
        )

        self.assertEqual(result.audit.reformulation_verdict.value, "C")
        self.assertTrue(result.metadata["contract_admitted"])
        self.assertEqual(len(result.metadata["reference_fingerprint"]), 64)
        self.assertEqual(result.metadata["semantic_view_source"], "deterministic_objective_anchored")

    def test_bad_contract_is_not_a_model_refutation(self) -> None:
        result = ORSMTPlugin().audit_constructive_reformulation_payload(
            model("left"),
            model("right"),
            payload={"forward": {}, "backward": {}, "objective": {}},
            semantic_view=SemanticView((("x", "x"),)),
        )

        self.assertEqual(result.audit.contract_status.value, "INVALID")
        self.assertEqual(result.audit.reformulation_verdict.value, "U")
        self.assertFalse(result.metadata["contract_admitted"])

    def test_constructive_route_recovers_only_a_direct_abstention(self) -> None:
        left = model("left")
        right = model("right")
        recovered = ORSMTPlugin(
            AbstainingVerifier(timeout_ms=3_000)
        ).audit_reformulation_with_constructive_payload(
            left,
            right,
            payload=identity_payload(),
        )
        direct = ORSMTPlugin().audit_reformulation_with_constructive_payload(
            left,
            right,
            payload=identity_payload(),
        )

        self.assertTrue(recovered.route_eligible)
        self.assertEqual(recovered.effective_verdict, Verdict.CERTIFIED)
        self.assertFalse(direct.route_eligible)
        self.assertIsNone(direct.constructive)
        self.assertEqual(direct.effective_verdict, Verdict.CERTIFIED)


if __name__ == "__main__":
    unittest.main()
