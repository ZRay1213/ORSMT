from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .ir import LinearConstraint, LinearExpr, ModelIR, Objective, Variable


def _family(name: str) -> tuple[str, str]:
    """Split an expanded solver name into its family and exact index suffix."""

    base, separator, suffix = name.partition("[")
    return base, f"[{suffix}" if separator else ""


@dataclass(frozen=True)
class _FamilySignature:
    """Structural information permitted for label-free family alignment."""

    cardinality: int
    suffixes: tuple[str, ...]
    kinds_by_suffix: tuple[tuple[str, str], ...]


def _families(model: ModelIR) -> dict[str, _FamilySignature]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for variable in model.variables:
        family, suffix = _family(variable.name)
        grouped.setdefault(family, []).append((suffix, variable.kind))
    families: dict[str, _FamilySignature] = {}
    for name, members in grouped.items():
        expanded = tuple(sorted(members))
        families[name] = _FamilySignature(
            cardinality=len(expanded),
            suffixes=tuple(suffix for suffix, _kind in expanded),
            kinds_by_suffix=expanded,
        )
    return families


def _compatible(left: _FamilySignature, right: _FamilySignature) -> bool:
    """Return whether two families have the same allowed structural evidence."""

    return (
        left.cardinality == right.cardinality
        and left.suffixes == right.suffixes
        and left.kinds_by_suffix == right.kinds_by_suffix
    )


def _unique_reference_cover(
    reference_names: set[str],
    candidate_names: set[str],
    reference_families: Mapping[str, _FamilySignature],
    candidate_families: Mapping[str, _FamilySignature],
) -> tuple[str, dict[str, str]]:
    """Find a unique injective candidate-to-reference family matching.

    Every reference family must be covered. Candidate families may remain
    unmatched and are subsequently treated as auxiliaries. After finding one
    cover, each selected edge is forbidden in turn; any second full cover then
    proves ambiguity. This avoids enumerating an exponential set of matchings.
    """

    if len(candidate_names) < len(reference_names):
        return "none", {}

    options = {
        reference_name: tuple(
            sorted(
                candidate_name
                for candidate_name in candidate_names
                if _compatible(
                    reference_families[reference_name],
                    candidate_families[candidate_name],
                )
            )
        )
        for reference_name in reference_names
    }
    if any(not choices for choices in options.values()):
        return "none", {}

    reference_order = tuple(
        sorted(reference_names, key=lambda name: (len(options[name]), name))
    )

    def find_cover(forbidden: tuple[str, str] | None = None) -> dict[str, str] | None:
        candidate_to_reference: dict[str, str] = {}

        def augment(reference_name: str, seen_candidates: set[str]) -> bool:
            for candidate_name in options[reference_name]:
                if forbidden == (candidate_name, reference_name):
                    continue
                if candidate_name in seen_candidates:
                    continue
                seen_candidates.add(candidate_name)
                previous_reference = candidate_to_reference.get(candidate_name)
                if previous_reference is None or augment(
                    previous_reference, seen_candidates
                ):
                    candidate_to_reference[candidate_name] = reference_name
                    return True
            return False

        for reference_name in reference_order:
            if not augment(reference_name, set()):
                return None
        return candidate_to_reference

    cover = find_cover()
    if cover is None:
        return "none", {}

    # Every distinct cover omits at least one edge from the first cover. Thus
    # testing the removal of each selected edge is a complete uniqueness test,
    # including the case where an unmatched candidate can replace a matched one.
    for candidate_name, reference_name in sorted(cover.items()):
        if find_cover((candidate_name, reference_name)) is not None:
            return "multiple", {}
    return "unique", cover


@dataclass(frozen=True)
class PairAlignment:
    """Label-free alignment used before relational reformulation queries.

    ``family_map`` maps candidate variable families to reference families.
    Candidate-only families are retained as existential auxiliaries. An
    alignment is unavailable when a reference family has no unique structural
    counterpart; callers must report that case as undecided.
    """

    reference: ModelIR | None
    candidate: ModelIR | None
    family_map: Mapping[str, str]
    auxiliary_families: tuple[str, ...] = ()
    semantic_families: tuple[str, ...] = ()
    semantic_mode: str = "all_variables"
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reference is not None and self.candidate is not None


def _rename_expression(expression: LinearExpr, names: Mapping[str, str]) -> LinearExpr:
    return LinearExpr(
        tuple((names[name], coefficient) for name, coefficient in expression.terms),
        expression.constant,
    )


def _reference_view(
    model: ModelIR, observable_families: set[str] | None = None
) -> ModelIR:
    return ModelIR(
        tuple(
            replace(
                variable,
                observable=(
                    True
                    if observable_families is None
                    else _family(variable.name)[0] in observable_families
                ),
            )
            for variable in model.variables
        ),
        model.constraints,
        model.objective,
        model.name,
        {**model.provenance, "alignment_role": "reference"},
    )


def align_variable_families(
    reference: ModelIR,
    candidate: ModelIR,
    *,
    semantic_mode: str = "all_variables",
) -> PairAlignment:
    """Infer only unique family renamings and project candidate-only auxiliaries.

    Matching uses family cardinality, the complete expanded suffix/index set,
    and variable kind at each suffix. Bounds, constraints, and objectives are
    deliberately excluded because they are semantic objects under test. No
    benchmark label or expected answer is consulted. If the remaining
    structural bipartite graph admits more than one perfect reference cover,
    alignment abstains instead of choosing a tie-break.
    """

    if semantic_mode not in {"all_variables", "objective_anchored"}:
        raise ValueError(f"unsupported semantic alignment mode: {semantic_mode}")

    reference_families = _families(reference)
    candidate_families = _families(candidate)
    family_map: dict[str, str] = {}

    # Identical family names are safe anchors only when all permitted structural
    # evidence agrees. In particular, a name match cannot override a kind
    # mismatch and prevent a unique structural rename elsewhere in the graph.
    for name in sorted(set(reference_families) & set(candidate_families)):
        if _compatible(reference_families[name], candidate_families[name]):
            family_map[name] = name

    unmatched_reference = set(reference_families) - set(family_map.values())
    unmatched_candidate = set(candidate_families) - set(family_map)

    matching_status, inferred_map = _unique_reference_cover(
        unmatched_reference,
        unmatched_candidate,
        reference_families,
        candidate_families,
    )
    if matching_status != "unique":
        detail = (
            "multiple_perfect_matchings"
            if matching_status == "multiple"
            else "no_perfect_matching"
        )
        return PairAlignment(
            None,
            None,
            family_map,
            tuple(sorted(unmatched_candidate)),
            (),
            semantic_mode,
            "ambiguous_or_missing_reference_families:"
            + ",".join(sorted(unmatched_reference))
            + ";"
            + detail,
        )

    family_map.update(inferred_map)
    unmatched_reference.difference_update(inferred_map.values())
    unmatched_candidate.difference_update(inferred_map)

    full_names: dict[str, str] = {}
    auxiliary_families = tuple(sorted(unmatched_candidate))
    if semantic_mode == "objective_anchored":
        reference_objective_families = {
            _family(name)[0]
            for name in (
                reference.objective.expression.support
                if reference.objective is not None
                else ()
            )
        }
        candidate_objective_families = {
            family_map[family]
            for name in (
                candidate.objective.expression.support
                if candidate.objective is not None
                else ()
            )
            if (family := _family(name)[0]) in family_map
        }
        semantic_families = tuple(
            sorted(reference_objective_families | candidate_objective_families)
        )
        if not semantic_families:
            return PairAlignment(
                None,
                None,
                family_map,
                auxiliary_families,
                (),
                semantic_mode,
                "objective_anchor_unavailable",
            )
        observable_reference_families = set(semantic_families)
    else:
        semantic_families = tuple(sorted(reference_families))
        observable_reference_families = set(reference_families)

    for variable in candidate.variables:
        family, suffix = _family(variable.name)
        if family in family_map:
            full_names[variable.name] = family_map[family] + suffix
        else:
            full_names[variable.name] = f"__aux__{family}{suffix}"

    if len(set(full_names.values())) != len(full_names):
        return PairAlignment(
            None,
            None,
            family_map,
            auxiliary_families,
            "candidate_name_collision_after_alignment",
        )

    aligned_variables = tuple(
        Variable(
            name=full_names[variable.name],
            kind=variable.kind,
            lb=variable.lb,
            ub=variable.ub,
            observable=family_map.get(_family(variable.name)[0])
            in observable_reference_families,
            provenance={
                **variable.provenance,
                "original_name": variable.name,
                "aligned_family": family_map.get(_family(variable.name)[0]),
            },
        )
        for variable in candidate.variables
    )
    aligned_constraints = tuple(
        LinearConstraint(
            row.name,
            _rename_expression(row.lhs, full_names),
            row.sense,
            row.rhs,
            row.group,
            row.provenance,
        )
        for row in candidate.constraints
    )
    aligned_objective = None
    if candidate.objective is not None:
        aligned_objective = Objective(
            candidate.objective.sense,
            _rename_expression(candidate.objective.expression, full_names),
            candidate.objective.provenance,
        )
    aligned_candidate = ModelIR(
        aligned_variables,
        aligned_constraints,
        aligned_objective,
        candidate.name,
        {
            **candidate.provenance,
            "alignment_role": "candidate",
            "family_map": dict(family_map),
            "auxiliary_families": auxiliary_families,
        },
    )
    aligned_reference = _reference_view(reference, observable_reference_families)

    if set(aligned_reference.observable_names) != set(aligned_candidate.observable_names):
        return PairAlignment(
            None,
            None,
            family_map,
            auxiliary_families,
            semantic_families,
            semantic_mode,
            "expanded_observable_schema_mismatch",
        )
    return PairAlignment(
        aligned_reference,
        aligned_candidate,
        family_map,
        auxiliary_families,
        semantic_families,
        semantic_mode,
    )
