from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from .constructive import (
    AffineTerm,
    ConstructiveContract,
    DSLValidationError,
    IteTerm,
    LinearComparison,
    ObjectiveTransport,
    Term,
    TotalVariableMap,
)
from .ir import ModelIR


FAMILY_SCHEMA_VERSION = "family-mapping-v1"
_FAMILY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ExpandedFamily:
    name: str
    rank: int
    members: Mapping[tuple[str, ...], str]


def _expanded_name(name: str) -> tuple[str, tuple[str, ...]]:
    family, separator, suffix = name.partition("[")
    if not _FAMILY.fullmatch(family):
        raise DSLValidationError(f"unsupported variable family name: {name!r}")
    if not separator:
        return family, ()
    if not suffix.endswith("]") or "[" in suffix[:-1] or "]" in suffix[:-1]:
        raise DSLValidationError(f"unsupported expanded variable name: {name!r}")
    body = suffix[:-1]
    if not body:
        raise DSLValidationError(f"empty variable index: {name!r}")
    indices = tuple(part.strip() for part in body.split(","))
    if any(not part for part in indices):
        raise DSLValidationError(f"empty variable index component: {name!r}")
    return family, indices


def expanded_families(model: ModelIR) -> Mapping[str, ExpandedFamily]:
    grouped: dict[str, dict[tuple[str, ...], str]] = {}
    ranks: dict[str, int] = {}
    for variable in model.variables:
        family, indices = _expanded_name(variable.name)
        if family in ranks and ranks[family] != len(indices):
            raise DSLValidationError(f"inconsistent rank in variable family {family!r}")
        ranks[family] = len(indices)
        members = grouped.setdefault(family, {})
        if indices in members:
            raise DSLValidationError(
                f"duplicate expanded index {indices!r} in family {family!r}"
            )
        members[indices] = variable.name
    return {
        family: ExpandedFamily(family, ranks[family], dict(members))
        for family, members in grouped.items()
    }


def _keys(payload: Any, required: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DSLValidationError(f"{label} requires exactly {sorted(required)}")
    return payload


def _resolve_indices(
    payload: Any,
    bindings: Mapping[str, str],
    *,
    expected_rank: int,
) -> tuple[str, ...]:
    if not isinstance(payload, list) or len(payload) != expected_rank:
        raise DSLValidationError(
            f"index expression must contain exactly {expected_rank} components"
        )
    result = []
    for atom in payload:
        if not isinstance(atom, Mapping) or len(atom) != 1:
            raise DSLValidationError("an index component is exactly {'index'} or {'literal'}")
        if set(atom) == {"index"}:
            symbol = atom["index"]
            if not isinstance(symbol, str) or symbol not in bindings:
                raise DSLValidationError(f"unknown bound index: {symbol!r}")
            result.append(bindings[symbol])
        elif set(atom) == {"literal"}:
            literal = atom["literal"]
            if not isinstance(literal, (str, int)) or isinstance(literal, bool):
                raise DSLValidationError("literal indices must be strings or integers")
            result.append(str(literal))
        else:
            raise DSLValidationError("an index component is exactly {'index'} or {'literal'}")
    return tuple(result)


def _variable_reference(
    payload: Any,
    source_families: Mapping[str, ExpandedFamily],
    bindings: Mapping[str, str],
) -> str:
    body = _keys(payload, {"family", "indices"}, "variable reference")
    family = body["family"]
    if not isinstance(family, str) or family not in source_families:
        raise DSLValidationError(f"unknown source variable family: {family!r}")
    schema = source_families[family]
    indices = _resolve_indices(body["indices"], bindings, expected_rank=schema.rank)
    try:
        return schema.members[indices]
    except KeyError as error:
        raise DSLValidationError(
            f"source variable {family}{indices!r} does not exist in the captured instance"
        ) from error


def _add_affine(terms: list[AffineTerm]) -> AffineTerm:
    coefficients: list[tuple[str, Any]] = []
    constant: Any = 0
    for term in terms:
        coefficients.extend(term.terms)
        constant += term.constant
    return AffineTerm(tuple(coefficients), constant)


def _scaled(term: AffineTerm, coefficient: Any) -> AffineTerm:
    scale = AffineTerm.rational(coefficient).constant
    return AffineTerm(
        tuple((name, scale * value) for name, value in term.terms),
        scale * term.constant,
    )


def _as_affine(term: Term, context: str) -> AffineTerm:
    if not isinstance(term, AffineTerm):
        raise DSLValidationError(f"{context} must expand to an affine term")
    return term


def _expand_expression(
    payload: Any,
    source: ModelIR,
    source_families: Mapping[str, ExpandedFamily],
    bindings: Mapping[str, str],
    *,
    depth: int,
    max_ite_depth: int,
) -> Term:
    if not isinstance(payload, Mapping) or len(payload) != 1:
        raise DSLValidationError(
            "expression must contain exactly one supported operator"
        )
    operator = next(iter(payload))
    body = payload[operator]
    if operator == "variable":
        return AffineTerm.variable(
            _variable_reference(body, source_families, bindings)
        )
    if operator == "constant":
        return AffineTerm.rational(body)
    if operator == "affine":
        affine = _keys(body, {"terms", "constant"}, "affine expression")
        if not isinstance(affine["terms"], list):
            raise DSLValidationError("affine terms must be a list")
        expanded = []
        for item in affine["terms"]:
            item = _keys(item, {"coefficient", "variable"}, "affine term")
            variable = AffineTerm.variable(
                _variable_reference(item["variable"], source_families, bindings)
            )
            expanded.append(_scaled(variable, item["coefficient"]))
        expanded.append(AffineTerm.rational(affine["constant"]))
        return _add_affine(expanded)
    if operator == "sum":
        summation = _keys(body, {"index", "over", "expression"}, "sum")
        symbol = summation["index"]
        if not isinstance(symbol, str) or not _INDEX.fullmatch(symbol):
            raise DSLValidationError("sum index must be an identifier")
        if symbol in bindings:
            raise DSLValidationError(f"sum index shadows an existing binding: {symbol}")
        over = _keys(summation["over"], {"family", "axis"}, "sum domain")
        family = over["family"]
        axis = over["axis"]
        if not isinstance(family, str) or family not in source_families:
            raise DSLValidationError(f"unknown sum-domain family: {family!r}")
        schema = source_families[family]
        if not isinstance(axis, int) or isinstance(axis, bool) or not 0 <= axis < schema.rank:
            raise DSLValidationError(f"sum axis is outside family {family!r}")
        values = sorted({indices[axis] for indices in schema.members})
        expanded_terms = []
        for value in values:
            nested = {**bindings, symbol: value}
            expanded_terms.append(
                _as_affine(
                    _expand_expression(
                        summation["expression"],
                        source,
                        source_families,
                        nested,
                        depth=depth,
                        max_ite_depth=max_ite_depth,
                    ),
                    "sum body",
                )
            )
        return _add_affine(expanded_terms)
    if operator == "binary_product":
        product = _keys(body, {"left", "right"}, "binary product")
        left = _as_affine(
            _expand_expression(
                product["left"], source, source_families, bindings,
                depth=depth, max_ite_depth=max_ite_depth,
            ),
            "binary-product left operand",
        )
        right = _as_affine(
            _expand_expression(
                product["right"], source, source_families, bindings,
                depth=depth, max_ite_depth=max_ite_depth,
            ),
            "binary-product right operand",
        )
        if len(left.terms) != 1 or left.terms[0][1] != 1 or left.constant != 0:
            raise DSLValidationError("binary-product left operand must be one variable")
        variable = source.variable_map[left.terms[0][0]]
        if variable.kind != "binary":
            raise DSLValidationError("binary-product guard variable is not binary")
        if depth >= max_ite_depth:
            raise DSLValidationError("maximum ITE depth exceeded")
        return IteTerm(
            LinearComparison(left, "==", AffineTerm.rational(1)),
            right,
            AffineTerm.rational(0),
        )
    if operator == "ite":
        if depth >= max_ite_depth:
            raise DSLValidationError("maximum ITE depth exceeded")
        ite = _keys(body, {"condition", "then", "else"}, "ITE")
        condition = _keys(ite["condition"], {"lhs", "op", "rhs"}, "condition")
        lhs = _as_affine(
            _expand_expression(
                condition["lhs"], source, source_families, bindings,
                depth=depth, max_ite_depth=max_ite_depth,
            ),
            "condition lhs",
        )
        rhs = _as_affine(
            _expand_expression(
                condition["rhs"], source, source_families, bindings,
                depth=depth, max_ite_depth=max_ite_depth,
            ),
            "condition rhs",
        )
        op = condition["op"]
        if not isinstance(op, str):
            raise DSLValidationError("condition operator must be a string")
        return IteTerm(
            LinearComparison(lhs, op, rhs),
            _expand_expression(
                ite["then"], source, source_families, bindings,
                depth=depth + 1, max_ite_depth=max_ite_depth,
            ),
            _expand_expression(
                ite["else"], source, source_families, bindings,
                depth=depth + 1, max_ite_depth=max_ite_depth,
            ),
        )
    raise DSLValidationError(f"unsupported family-map operator: {operator!r}")


def expand_family_map(
    payload: Any,
    source: ModelIR,
    target: ModelIR,
    *,
    max_ite_depth: int = 4,
) -> TotalVariableMap:
    if not isinstance(payload, list):
        raise DSLValidationError("family map must be a list of rules")
    source_families = expanded_families(source)
    target_families = expanded_families(target)
    rules: dict[str, Mapping[str, Any]] = {}
    for raw in payload:
        rule = _keys(
            raw,
            {"target_family", "index_variables", "expression"},
            "family-map rule",
        )
        family = rule["target_family"]
        if not isinstance(family, str) or family not in target_families:
            raise DSLValidationError(f"unknown target family: {family!r}")
        if family in rules:
            raise DSLValidationError(f"duplicate rule for target family {family!r}")
        indices = rule["index_variables"]
        rank = target_families[family].rank
        if (
            not isinstance(indices, list)
            or len(indices) != rank
            or any(not isinstance(name, str) or not _INDEX.fullmatch(name) for name in indices)
            or len(set(indices)) != len(indices)
        ):
            raise DSLValidationError(
                f"target family {family!r} requires {rank} unique index identifiers"
            )
        rules[family] = rule
    if set(rules) != set(target_families):
        missing = sorted(set(target_families) - set(rules))
        extra = sorted(set(rules) - set(target_families))
        raise DSLValidationError(
            f"family map mismatch: missing={missing}, extra={extra}"
        )

    assignments: dict[str, Term] = {}
    for family, schema in sorted(target_families.items()):
        rule = rules[family]
        symbols = rule["index_variables"]
        for concrete_indices, target_name in sorted(schema.members.items()):
            bindings = dict(zip(symbols, concrete_indices))
            assignments[target_name] = _expand_expression(
                rule["expression"],
                source,
                source_families,
                bindings,
                depth=0,
                max_ite_depth=max_ite_depth,
            )
    return TotalVariableMap.from_mapping(assignments)


def parse_family_constructive_contract(
    payload: Any,
    left: ModelIR,
    right: ModelIR,
    *,
    max_ite_depth: int = 4,
) -> ConstructiveContract:
    body = _keys(
        payload,
        {"schema_version", "forward", "backward", "objective"},
        "family constructive contract",
    )
    if body["schema_version"] != FAMILY_SCHEMA_VERSION:
        raise DSLValidationError(
            f"unsupported family mapping schema: {body['schema_version']!r}"
        )
    objective = _keys(body["objective"], {"scale", "offset"}, "objective")
    return ConstructiveContract(
        forward=expand_family_map(
            body["forward"], left, right, max_ite_depth=max_ite_depth
        ),
        backward=expand_family_map(
            body["backward"], right, left, max_ite_depth=max_ite_depth
        ),
        objective=ObjectiveTransport(objective["scale"], objective["offset"]),
    )
