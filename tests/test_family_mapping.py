from __future__ import annotations

import unittest
from fractions import Fraction

from orsmt import (
    AffineTerm,
    DSLValidationError,
    FAMILY_SCHEMA_VERSION,
    IteTerm,
    LinearConstraint,
    LinearExpr,
    ModelIR,
    Objective,
    ORSMTPlugin,
    Variable,
    expand_family_map,
    expanded_families,
    parse_family_constructive_contract,
)


def variable(family: str, *indices: int, kind: str = "binary") -> Variable:
    suffix = "" if not indices else "[" + ",".join(map(str, indices)) + "]"
    return Variable(f"{family}{suffix}", kind)


def model(name: str, variables: tuple[Variable, ...]) -> ModelIR:
    return ModelIR(
        variables,
        (),
        Objective("max", LinearExpr.from_mapping({variables[0].name: 1})),
        name,
    )


def ref(family: str, *indices: dict[str, object]) -> dict[str, object]:
    return {"family": family, "indices": list(indices)}


def var_expr(family: str, *indices: dict[str, object]) -> dict[str, object]:
    return {"variable": ref(family, *indices)}


def identity_rule(family: str, *symbols: str) -> dict[str, object]:
    return {
        "target_family": family,
        "index_variables": list(symbols),
        "expression": var_expr(
            family, *({"index": symbol} for symbol in symbols)
        ),
    }


class FamilyMappingTests(unittest.TestCase):
    def test_expanded_families_and_array_identity(self) -> None:
        left = model("left", tuple(variable("x", i, j) for i in range(2) for j in range(2)))
        right = model("right", tuple(variable("x", i, j) for i in range(2) for j in range(2)))

        families = expanded_families(left)
        self.assertEqual(families["x"].rank, 2)
        self.assertEqual(len(families["x"].members), 4)

        mapping = expand_family_map([identity_rule("x", "i", "j")], left, right)
        self.assertEqual(set(mapping.as_dict), {f"x[{i},{j}]" for i in range(2) for j in range(2)})
        self.assertEqual(mapping.as_dict["x[1,0]"], AffineTerm.variable("x[1,0]"))

    def test_p12_style_ite_expands_per_index(self) -> None:
        source = model(
            "source",
            tuple(variable("x", i, j) for i in range(3) for j in range(3))
            + tuple(variable("u", i, kind="integer") for i in range(3)),
        )
        target = model(
            "target",
            tuple(variable("x", i, j) for i in range(3) for j in range(3))
            + tuple(variable("u", i, kind="integer") for i in range(3)),
        )
        payload = [
            identity_rule("x", "i", "j"),
            {
                "target_family": "u",
                "index_variables": ["i"],
                "expression": {
                    "ite": {
                        "condition": {
                            "lhs": var_expr("x", {"literal": 0}, {"index": "i"}),
                            "op": "==",
                            "rhs": {"constant": 1},
                        },
                        "then": {"constant": 2},
                        "else": var_expr("u", {"index": "i"}),
                    }
                },
            },
        ]

        mapping = expand_family_map(payload, source, target)
        self.assertIsInstance(mapping.as_dict["u[2]"], IteTerm)
        self.assertEqual(mapping.as_dict["u[2]"].condition.lhs, AffineTerm.variable("x[0,2]"))
        self.assertEqual(mapping.as_dict["u[2]"].else_term, AffineTerm.variable("u[2]"))

    def test_finite_sum_expands_to_exact_affine_term(self) -> None:
        source = model("source", tuple(variable("x", i) for i in range(3)))
        target = model("target", (Variable("total", "integer"),))
        mapping = expand_family_map(
            [
                {
                    "target_family": "total",
                    "index_variables": [],
                    "expression": {
                        "sum": {
                            "index": "i",
                            "over": {"family": "x", "axis": 0},
                            "expression": var_expr("x", {"index": "i"}),
                        }
                    },
                }
            ],
            source,
            target,
        )

        self.assertEqual(
            mapping.as_dict["total"],
            AffineTerm.from_mapping({"x[0]": 1, "x[1]": 1, "x[2]": 1}),
        )

    def test_binary_product_uses_binary_guard_and_rejects_integer_guard(self) -> None:
        target = model("target", (Variable("y", "integer"),))
        payload = [
            {
                "target_family": "y",
                "index_variables": [],
                "expression": {
                    "binary_product": {
                        "left": var_expr("b"),
                        "right": var_expr("v"),
                    }
                },
            }
        ]
        binary_source = model(
            "binary-source",
            (Variable("b", "binary"), Variable("v", "integer")),
        )
        integer_source = model(
            "integer-source",
            (Variable("b", "integer"), Variable("v", "integer")),
        )

        term = expand_family_map(payload, binary_source, target).as_dict["y"]
        self.assertIsInstance(term, IteTerm)
        with self.assertRaisesRegex(DSLValidationError, "guard variable is not binary"):
            expand_family_map(payload, integer_source, target)

    def test_requires_exact_total_target_family_coverage(self) -> None:
        source = model("source", (variable("x", 0), variable("u", 0, kind="integer")))
        target = model("target", (variable("x", 0), variable("u", 0, kind="integer")))

        with self.assertRaisesRegex(DSLValidationError, r"missing=\['u'\]"):
            expand_family_map([identity_rule("x", "i")], source, target)

    def test_unknown_index_and_unknown_variable_are_rejected(self) -> None:
        source = model("source", (variable("x", 0),))
        target = model("target", (variable("x", 0),))
        bad_index = identity_rule("x", "i")
        bad_index["expression"] = var_expr("x", {"index": "j"})
        with self.assertRaisesRegex(DSLValidationError, "unknown bound index"):
            expand_family_map([bad_index], source, target)

        bad_family = identity_rule("x", "i")
        bad_family["expression"] = var_expr("ghost", {"index": "i"})
        with self.assertRaisesRegex(DSLValidationError, "unknown source variable family"):
            expand_family_map([bad_family], source, target)

    def test_family_contract_and_plugin_certify_small_equivalent_models(self) -> None:
        variables = (variable("x", 0), variable("x", 1))
        left = ModelIR(
            variables,
            (LinearConstraint("capacity", LinearExpr.from_mapping({"x[0]": 1, "x[1]": 1}), "<=", 1),),
            Objective("max", LinearExpr.from_mapping({"x[0]": 2, "x[1]": 1})),
            "left",
        )
        right = ModelIR(
            variables,
            (LinearConstraint("capacity", LinearExpr.from_mapping({"x[0]": 1, "x[1]": 1}), "<=", 1),),
            Objective("max", LinearExpr.from_mapping({"x[0]": 2, "x[1]": 1})),
            "right",
        )
        payload = {
            "schema_version": FAMILY_SCHEMA_VERSION,
            "forward": [identity_rule("x", "i")],
            "backward": [identity_rule("x", "i")],
            "objective": {"scale": 1, "offset": 0},
        }

        contract = parse_family_constructive_contract(payload, left, right)
        self.assertEqual(contract.objective.scale, Fraction(1))
        result = ORSMTPlugin().audit_constructive_reformulation_payload(
            left, right, payload=payload
        )
        self.assertEqual(result.audit.reformulation_verdict.value, "C")
        self.assertEqual(result.metadata["contract_schema"], FAMILY_SCHEMA_VERSION)

    def test_non_mapping_plugin_payload_is_an_invalid_contract_not_a_crash(self) -> None:
        left = model("left", (Variable("x", "binary"),))
        right = model("right", (Variable("x", "binary"),))

        result = ORSMTPlugin().audit_constructive_reformulation_payload(
            left, right, payload=[]
        )

        self.assertEqual(result.audit.contract_status.value, "INVALID")
        self.assertEqual(result.audit.reformulation_verdict.value, "U")


if __name__ == "__main__":
    unittest.main()
