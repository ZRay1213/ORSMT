from __future__ import annotations

import unittest

from orsmt import LinearConstraint, LinearExpr, ModelIR, Objective, Variable
from orsmt.alignment import align_variable_families


def _variables(
    family: str, suffixes: tuple[str, ...], kind: str
) -> tuple[Variable, ...]:
    return tuple(Variable(f"{family}[{suffix}]", kind) for suffix in suffixes)


class UniqueFamilyMatchingTests(unittest.TestCase):
    def test_kind_resolves_crossed_simple_renames(self) -> None:
        """A same-name kind conflict must not block the unique global rename."""

        reference = ModelIR(
            _variables("x", ("0", "1"), "binary")
            + _variables("y", ("0", "1"), "integer"),
            (),
            None,
            "reference",
        )
        candidate = ModelIR(
            _variables("x", ("0", "1"), "integer")
            + _variables("z", ("0", "1"), "binary"),
            (),
            None,
            "candidate",
        )

        alignment = align_variable_families(reference, candidate)

        self.assertTrue(alignment.available)
        self.assertEqual(alignment.family_map, {"x": "y", "z": "x"})
        self.assertEqual(alignment.auxiliary_families, ())
        self.assertEqual(
            set(alignment.candidate.observable_names),
            {"x[0]", "x[1]", "y[0]", "y[1]"},
        )

    def test_expanded_suffix_signature_resolves_multiple_renames(self) -> None:
        reference = ModelIR(
            _variables("short", ("0", "1"), "continuous")
            + _variables("shifted", ("1", "2"), "continuous"),
            (),
            None,
            "reference",
        )
        candidate = ModelIR(
            _variables("alpha", ("1", "2"), "continuous")
            + _variables("beta", ("0", "1"), "continuous"),
            (),
            None,
            "candidate",
        )

        alignment = align_variable_families(reference, candidate)

        self.assertTrue(alignment.available)
        self.assertEqual(
            alignment.family_map, {"alpha": "shifted", "beta": "short"}
        )

    def test_multiple_perfect_matchings_abstain_despite_semantic_clues(self) -> None:
        """Constraints and objectives must not be used to break a structural tie."""

        reference = ModelIR(
            _variables("left", ("0", "1"), "integer")
            + _variables("right", ("0", "1"), "integer"),
            (
                LinearConstraint(
                    "left_semantics",
                    LinearExpr.from_mapping({"left[0]": 1, "left[1]": 1}),
                    "<=",
                    3,
                ),
            ),
            Objective("max", LinearExpr.from_mapping({"right[0]": 7})),
            "reference",
        )
        candidate = ModelIR(
            _variables("alpha", ("0", "1"), "integer")
            + _variables("beta", ("0", "1"), "integer"),
            (
                LinearConstraint(
                    "obvious_but_forbidden_clue",
                    LinearExpr.from_mapping({"alpha[0]": 1, "alpha[1]": 1}),
                    "<=",
                    3,
                ),
            ),
            Objective("max", LinearExpr.from_mapping({"beta[0]": 7})),
            "candidate",
        )

        alignment = align_variable_families(reference, candidate)

        self.assertFalse(alignment.available)
        self.assertIn("multiple_perfect_matchings", alignment.reason or "")
        self.assertEqual(alignment.family_map, {})

    def test_unmatched_candidate_is_auxiliary_only_after_unique_cover(self) -> None:
        reference = ModelIR(
            _variables("decision", ("0", "1"), "binary"),
            (),
            None,
            "reference",
        )
        candidate = ModelIR(
            _variables("renamed", ("0", "1"), "binary")
            + _variables("slack", ("0",), "continuous"),
            (),
            None,
            "candidate",
        )

        alignment = align_variable_families(reference, candidate)

        self.assertTrue(alignment.available)
        self.assertEqual(alignment.family_map, {"renamed": "decision"})
        self.assertEqual(alignment.auxiliary_families, ("slack",))

    def test_objective_anchored_mode_projects_matched_helper_families(self) -> None:
        reference = ModelIR(
            _variables("x", ("0", "1"), "binary")
            + _variables("u", ("0", "1"), "continuous"),
            (),
            Objective("min", LinearExpr.from_mapping({"x[0]": 2, "x[1]": 3})),
            "reference",
        )
        candidate = ModelIR(
            _variables("x", ("0", "1"), "binary")
            + _variables("u", ("0", "1"), "continuous"),
            (),
            Objective("min", LinearExpr.from_mapping({"x[0]": 2, "x[1]": 3})),
            "candidate",
        )

        alignment = align_variable_families(
            reference, candidate, semantic_mode="objective_anchored"
        )

        self.assertTrue(alignment.available)
        self.assertEqual(alignment.semantic_families, ("x",))
        self.assertEqual(alignment.reference.observable_names, ("x[0]", "x[1]"))
        self.assertEqual(alignment.candidate.observable_names, ("x[0]", "x[1]"))
        self.assertEqual(alignment.reference.auxiliary_names, ("u[0]", "u[1]"))
        self.assertEqual(alignment.candidate.auxiliary_names, ("u[0]", "u[1]"))


if __name__ == "__main__":
    unittest.main()
