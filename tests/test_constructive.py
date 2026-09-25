from __future__ import annotations

import unittest
from fractions import Fraction

from orsmt.constructive import (
    AffineTerm,
    ConstructiveContract,
    ConstructiveVerifier,
    ContractStatus,
    CounterpartStatus,
    DSLValidationError,
    IteTerm,
    LinearComparison,
    ObjectiveTransport,
    ReformulationVerdict,
    SemanticView,
    TotalVariableMap,
    parse_constructive_contract,
    parse_term,
    model_fingerprint,
)
from orsmt.ir import LinearConstraint, LinearExpr, ModelIR, Objective, Variable


class RejectingReplay:
    def replay(self, model, point):
        return None, "native backend unavailable"


def identity_contract(*names: str, scale: int | str = 1) -> ConstructiveContract:
    assignments = {name: AffineTerm.variable(name) for name in names}
    return ConstructiveContract(
        TotalVariableMap.from_mapping(assignments),
        TotalVariableMap.from_mapping(assignments),
        ObjectiveTransport(scale),
    )


def binary_model(name: str, *, upper: int = 1) -> ModelIR:
    return ModelIR(
        (Variable("x", "binary", ub=upper),),
        (),
        Objective("max", LinearExpr.from_mapping({"x": 1})),
        name,
    )


def source_and_extended_target() -> tuple[ModelIR, ModelIR]:
    source = binary_model("source")
    target = ModelIR(
        (
            Variable("x", "binary"),
            Variable("u", "integer", lb=0, ub=2, observable=False),
        ),
        (
            LinearConstraint(
                "define_u",
                LinearExpr.from_mapping({"u": 1, "x": -2}),
                "==",
                0,
            ),
        ),
        Objective("max", LinearExpr.from_mapping({"x": 1})),
        "target",
    )
    return source, target


class ConstructiveVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.verifier = ConstructiveVerifier(timeout_ms=3_000, max_ite_depth=3)
        self.view = SemanticView((("x", "x"),))

    def test_identity_contract_certifies_reformulation(self) -> None:
        left = binary_model("left")
        right = binary_model("right")

        audit = self.verifier.verify(left, right, identity_contract("x"), self.view)

        self.assertEqual(audit.contract_status, ContractStatus.VALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.C)
        self.assertEqual(len(audit.obligations), 6)
        self.assertTrue(all(row.status == ContractStatus.VALID for row in audit.obligations))

    def test_correct_finite_ite_map_is_valid(self) -> None:
        source, target = source_and_extended_target()
        is_one = LinearComparison(
            AffineTerm.variable("x"), "==", AffineTerm.rational(1)
        )
        forward = TotalVariableMap.from_mapping(
            {
                "x": AffineTerm.variable("x"),
                "u": IteTerm(is_one, AffineTerm.rational(2), AffineTerm.rational(0)),
            }
        )
        backward = TotalVariableMap.from_mapping({"x": AffineTerm.variable("x")})
        contract = ConstructiveContract(forward, backward, ObjectiveTransport(1))

        audit = self.verifier.verify(source, target, contract, self.view)

        self.assertEqual(audit.contract_status, ContractStatus.VALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.C)

    def test_wrong_ite_guard_invalidates_contract_but_not_model(self) -> None:
        source, target = source_and_extended_target()
        wrong_guard = LinearComparison(
            AffineTerm.variable("x"), "==", AffineTerm.rational(0)
        )
        contract = ConstructiveContract(
            TotalVariableMap.from_mapping(
                {
                    "x": AffineTerm.variable("x"),
                    "u": IteTerm(
                        wrong_guard,
                        AffineTerm.rational(2),
                        AffineTerm.rational(0),
                    ),
                }
            ),
            TotalVariableMap.from_mapping({"x": AffineTerm.variable("x")}),
            ObjectiveTransport(1),
        )

        audit = self.verifier.verify(source, target, contract, self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
        failure = next(row for row in audit.obligations if row.name == "forward_feasibility")
        self.assertEqual(failure.status, ContractStatus.INVALID)
        self.assertTrue(
            any(check.status == CounterpartStatus.EXISTS for check in audit.counterpart_checks)
        )

    def test_wrong_map_cannot_by_itself_refute_equivalent_models(self) -> None:
        left = binary_model("left")
        right = binary_model("right")
        flip = AffineTerm.from_mapping({"x": -1}, constant=1)
        contract = ConstructiveContract(
            TotalVariableMap.from_mapping({"x": flip}),
            TotalVariableMap.from_mapping({"x": flip}),
            ObjectiveTransport(1),
        )

        audit = self.verifier.verify(left, right, contract, self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
        self.assertTrue(
            any(row.name == "forward_semantics" and row.status == ContractStatus.INVALID for row in audit.obligations)
        )
        self.assertTrue(audit.counterpart_checks)
        self.assertTrue(
            all(check.status == CounterpartStatus.EXISTS for check in audit.counterpart_checks)
        )

    def test_missing_target_variable_is_invalid_and_undecided(self) -> None:
        source, target = source_and_extended_target()
        contract = ConstructiveContract(
            TotalVariableMap.from_mapping({"x": AffineTerm.variable("x")}),
            TotalVariableMap.from_mapping({"x": AffineTerm.variable("x")}),
            ObjectiveTransport(1),
        )

        audit = self.verifier.verify(source, target, contract, self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
        self.assertIn("forward total map mismatch", audit.reason or "")
        self.assertIn("u", audit.reason or "")

    def test_validate_reports_unknown_mapping_variable_once(self) -> None:
        left = binary_model("left")
        right = binary_model("right")
        contract = ConstructiveContract(
            TotalVariableMap.from_mapping({"x": AffineTerm.variable("ghost")}),
            TotalVariableMap.from_mapping({"x": AffineTerm.variable("x")}),
            ObjectiveTransport(1),
        )

        audit = self.verifier.verify(left, right, contract, self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
        self.assertEqual(
            audit.reason,
            "forward map for x uses unknown variables: ['ghost']",
        )

    def test_nonpositive_objective_scale_is_invalid_not_refuted(self) -> None:
        left = binary_model("left")
        right = binary_model("right")
        for scale in (0, -1):
            with self.subTest(scale=scale):
                audit = self.verifier.verify(
                    left,
                    right,
                    identity_contract("x", scale=scale),
                    self.view,
                )
                self.assertEqual(audit.contract_status, ContractStatus.INVALID)
                self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
                self.assertIn("scale must be positive", audit.reason or "")

    def test_missing_semantic_counterpart_refutes_model(self) -> None:
        left = binary_model("left")
        right = ModelIR(
            (Variable("x", "binary"),),
            (LinearConstraint("only_zero", LinearExpr.from_mapping({"x": 1}), "==", 0),),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "right",
        )

        audit = self.verifier.verify(left, right, identity_contract("x"), self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.R)
        self.assertTrue(
            any(check.status == CounterpartStatus.ABSENT for check in audit.counterpart_checks)
        )

    def test_native_replay_failure_withholds_model_refutation(self) -> None:
        left = binary_model("left")
        right = ModelIR(
            (Variable("x", "binary"),),
            (LinearConstraint("only_zero", LinearExpr.from_mapping({"x": 1}), "==", 0),),
            Objective("max", LinearExpr.from_mapping({"x": 1})),
            "right",
        )
        verifier = ConstructiveVerifier(
            timeout_ms=3_000,
            native_replay=RejectingReplay(),
        )

        audit = verifier.verify(left, right, identity_contract("x"), self.view)

        self.assertEqual(audit.contract_status, ContractStatus.INVALID)
        self.assertEqual(audit.reformulation_verdict, ReformulationVerdict.U)
        self.assertTrue(audit.counterpart_checks)
        self.assertTrue(
            all(check.status == CounterpartStatus.UNKNOWN for check in audit.counterpart_checks)
        )

    def test_parser_enforces_exact_rational_linear_bounded_dsl(self) -> None:
        payload = {
            "ite": {
                "if": {
                    "lhs": {"affine": {"terms": {"x": 1}}},
                    "op": ">=",
                    "rhs": "1/2",
                },
                "then": {"affine": {"terms": {"x": "3/2"}, "constant": "1/3"}},
                "else": 0,
            }
        }
        term = parse_term(payload, {"x"}, max_ite_depth=1)
        self.assertIsInstance(term, IteTerm)
        self.assertEqual(term.then_term.constant, Fraction(1, 3))

        with self.assertRaises(DSLValidationError):
            parse_term(0.5, {"x"})
        with self.assertRaises(DSLValidationError):
            parse_term({"affine": {"terms": {"unknown": 1}}}, {"x"})
        with self.assertRaises(DSLValidationError):
            parse_term(payload, {"x"}, max_ite_depth=0)

    def test_whole_contract_parser_fixes_both_target_schemas(self) -> None:
        source, target = source_and_extended_target()
        payload = {
            "backward": {
                "x": {"affine": {"terms": {"x": 1}}},
            },
            "objective": {"offset": "-7/3", "scale": "3/2"},
            "forward": {
                "u": {"affine": {"terms": {"x": 2}}},
                "x": {"affine": {"terms": {"x": 1}}},
            },
        }

        contract = parse_constructive_contract(payload, source, target)

        self.assertEqual(
            tuple(name for name, _term in contract.forward.assignments),
            ("u", "x"),
        )
        self.assertEqual(contract.objective.scale, Fraction(3, 2))
        self.assertEqual(contract.objective.offset, Fraction(-7, 3))

        missing_target = {
            **payload,
            "forward": {"x": {"affine": {"terms": {"x": 1}}}},
        }
        with self.assertRaisesRegex(DSLValidationError, "total map mismatch"):
            parse_constructive_contract(missing_target, source, target)

        floating_objective = {
            **payload,
            "objective": {"scale": 1.5, "offset": 0},
        }
        with self.assertRaisesRegex(DSLValidationError, "floating-point"):
            parse_constructive_contract(floating_objective, source, target)

        with self.assertRaisesRegex(DSLValidationError, "requires exactly"):
            parse_constructive_contract({**payload, "extra": {}}, source, target)

    def test_model_fingerprint_binds_exact_semantics_not_display_name(self) -> None:
        left = binary_model("left")
        renamed = binary_model("different display name")
        restricted = ModelIR(
            left.variables,
            (LinearConstraint("cap", LinearExpr.from_mapping({"x": 1}), "==", 0),),
            left.objective,
            "left",
        )

        self.assertEqual(model_fingerprint(left), model_fingerprint(renamed))
        self.assertNotEqual(model_fingerprint(left), model_fingerprint(restricted))


if __name__ == "__main__":
    unittest.main()
