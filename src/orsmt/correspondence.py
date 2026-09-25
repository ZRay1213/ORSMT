from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .ir import LinearExpr, ModelIR


@dataclass(frozen=True)
class CandidateProjectionProposal:
    """Deterministic admission result for an untrusted mapper payload."""

    returned: bool
    well_typed: bool
    projection: Mapping[str, LinearExpr] = field(default_factory=dict)
    reason: str | None = None


def candidate_projection_json_schema(
    reference: ModelIR, candidate: ModelIR
) -> dict[str, Any]:
    """Return a strict response schema for an untrusted correspondence mapper.

    The schema fixes both vocabularies before generation: every observable
    reference decision must be defined and only captured candidate variables
    may appear in an affine expression. Semantic validity is deliberately not
    encoded here; admitted responses still require exact SMT adjudication.
    """

    coefficient = {"type": ["string", "integer"]}
    candidate_terms = {
        name: dict(coefficient) for name in sorted(candidate.variable_map)
    }
    expression = {
        "type": "object",
        "additionalProperties": False,
        "required": ["terms"],
        "properties": {
            "terms": {
                "type": "object",
                "additionalProperties": False,
                "properties": candidate_terms,
            },
            "constant": dict(coefficient),
        },
    }
    reference_names = sorted(reference.observable_names)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": reference_names,
        "properties": {name: expression for name in reference_names},
    }


def parse_candidate_projection(
    reference: ModelIR,
    candidate: ModelIR,
    payload: Mapping[str, Any] | None,
) -> CandidateProjectionProposal:
    """Parse a restricted JSON-like candidate-to-reference affine map.

    The only admitted schema is one entry per observable reference decision::

        {"x": {"terms": {"x_left": "1", "x_right": "1"},
               "constant": "0"}}

    Coefficients are parsed exactly by :class:`LinearExpr`.  This function
    validates syntax and vocabulary only; it never judges semantic validity.
    """

    if payload is None:
        return CandidateProjectionProposal(False, False, reason="not_returned")
    if not isinstance(payload, Mapping):
        return CandidateProjectionProposal(True, False, reason="payload_not_mapping")

    reference_names = set(reference.observable_names)
    if not reference_names:
        return CandidateProjectionProposal(
            True, False, reason="reference_has_no_observable_decisions"
        )
    if set(payload) != reference_names:
        return CandidateProjectionProposal(
            True, False, reason="reference_schema_mismatch"
        )

    candidate_names = set(candidate.variable_map)
    projection: dict[str, LinearExpr] = {}
    for reference_name in sorted(reference_names):
        specification = payload[reference_name]
        if not isinstance(specification, Mapping):
            return CandidateProjectionProposal(
                True, False, reason=f"expression_not_mapping:{reference_name}"
            )
        if set(specification) - {"terms", "constant"}:
            return CandidateProjectionProposal(
                True, False, reason=f"expression_schema_mismatch:{reference_name}"
            )
        terms = specification.get("terms")
        if not isinstance(terms, Mapping):
            return CandidateProjectionProposal(
                True, False, reason=f"terms_not_mapping:{reference_name}"
            )
        if any(not isinstance(name, str) or not name for name in terms):
            return CandidateProjectionProposal(
                True, False, reason=f"invalid_candidate_variable:{reference_name}"
            )
        unknown = set(terms) - candidate_names
        if unknown:
            return CandidateProjectionProposal(
                True,
                False,
                reason=f"unknown_candidate_variable:{reference_name}",
            )
        try:
            expression = LinearExpr.from_mapping(
                terms, constant=specification.get("constant", 0)
            )
        except (TypeError, ValueError, ZeroDivisionError):
            return CandidateProjectionProposal(
                True, False, reason=f"invalid_exact_coefficient:{reference_name}"
            )
        if not expression.support:
            return CandidateProjectionProposal(
                True, False, reason=f"constant_only_expression:{reference_name}"
            )
        projection[reference_name] = expression

    return CandidateProjectionProposal(True, True, projection=projection)
