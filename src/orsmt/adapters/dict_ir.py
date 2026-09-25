from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from decimal import Decimal
from typing import Any

from ..ir import (
    LinearConstraint,
    LinearExpr,
    ModelIR,
    Objective,
    ResultRecord,
    SourceObligation,
    Variable,
    as_fraction,
)


def _terms(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("linear terms must be a mapping or a sequence")
    parsed: dict[str, Any] = {}
    for item in raw:
        if isinstance(item, Mapping):
            name = item.get("var", item.get("name"))
            coefficient = item.get("coef", item.get("coefficient"))
        else:
            name, coefficient = item
        if name is None or coefficient is None:
            raise ValueError(f"invalid term: {item}")
        key = str(name)
        parsed[key] = as_fraction(parsed.get(key, 0)) + as_fraction(coefficient)
    return parsed


def _add(left: LinearExpr, right: LinearExpr) -> LinearExpr:
    return LinearExpr(left.terms + right.terms, left.constant + right.constant)


def _scale(expression: LinearExpr, factor: Any) -> LinearExpr:
    scalar = as_fraction(factor)
    return LinearExpr(
        tuple((name, coefficient * scalar) for name, coefficient in expression.terms),
        expression.constant * scalar,
    )


def _linear_expr(raw: Any, known_names: set[str] | None = None) -> LinearExpr:
    """Parse the linear expression variants emitted by the benchmark generators."""

    if raw is None:
        raise ValueError("missing linear expression")
    if isinstance(raw, (int, float, Fraction, Decimal)):
        return LinearExpr(constant=as_fraction(raw))
    if isinstance(raw, str):
        try:
            return LinearExpr(constant=as_fraction(raw))
        except (ValueError, ZeroDivisionError):
            if known_names is not None and raw not in known_names:
                raise ValueError(f"unknown expression symbol: {raw}")
            return LinearExpr.from_mapping({raw: 1})
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        expression = LinearExpr()
        for item in raw:
            expression = _add(expression, _linear_expr(item, known_names))
        return expression
    if not isinstance(raw, Mapping):
        raise TypeError(f"unsupported linear expression: {type(raw).__name__}")

    if "var" in raw or ("name" in raw and ("coef" in raw or "coefficient" in raw)):
        name = str(raw.get("var", raw.get("name")))
        if known_names is not None and name not in known_names:
            raise ValueError(f"unknown expression symbol: {name}")
        return LinearExpr.from_mapping({name: raw.get("coef", raw.get("coefficient", 1))})
    if "terms" in raw or raw.get("type") == "expr":
        expression = _linear_expr(raw.get("terms", ()), known_names)
        return _add(expression, LinearExpr(constant=raw.get("constant", raw.get("const", 0))))
    if "$add" in raw or "$sum" in raw:
        return _linear_expr(raw.get("$add", raw.get("$sum")), known_names)
    if "$subtract" in raw:
        values = raw["$subtract"]
        if not isinstance(values, Sequence) or len(values) != 2:
            raise ValueError("$subtract requires two operands")
        return _add(
            _linear_expr(values[0], known_names),
            _scale(_linear_expr(values[1], known_names), -1),
        )
    if "$multiply" in raw:
        values = raw["$multiply"]
        if not isinstance(values, Sequence):
            raise ValueError("$multiply requires a sequence")
        factor = Fraction(1)
        symbolic: LinearExpr | None = None
        for value in values:
            expression = _linear_expr(value, known_names)
            if expression.terms:
                if symbolic is not None:
                    raise ValueError("nonlinear product")
                symbolic = expression
            else:
                factor *= expression.constant
        return _scale(symbolic or LinearExpr(constant=1), factor)
    if "$divide" in raw:
        values = raw["$divide"]
        if not isinstance(values, Sequence) or len(values) != 2:
            raise ValueError("$divide requires two operands")
        numerator = _linear_expr(values[0], known_names)
        denominator = _linear_expr(values[1], known_names)
        if denominator.terms or denominator.constant == 0:
            raise ValueError("nonconstant or zero divisor")
        return _scale(numerator, Fraction(1, 1) / denominator.constant)
    if "$negative" in raw:
        return _scale(_linear_expr(raw["$negative"], known_names), -1)

    # A plain mapping is the conventional {variable: coefficient} form.
    if all(isinstance(name, str) for name in raw):
        if known_names is not None and not set(raw) <= known_names:
            unknown = sorted(set(raw) - known_names)
            raise ValueError(f"unknown expression symbols: {unknown}")
        return LinearExpr.from_mapping(raw)
    raise ValueError("unsupported expression object")


class DictModelAdapter:
    """Adapter for JSON-like models used by common auto-formulation pipelines."""

    def to_ir(self, model: Mapping[str, Any]) -> ModelIR:
        variables = []
        for raw in model.get("variables", ()):
            kind = str(raw.get("kind", raw.get("type", "continuous"))).lower()
            kind = {"int": "integer", "bool": "binary", "real": "continuous"}.get(kind, kind)
            variables.append(
                Variable(
                    name=str(raw["name"]),
                    kind=kind,
                    lb=raw.get("lb"),
                    ub=raw.get("ub"),
                    observable=bool(raw.get("observable", True)),
                    provenance=raw.get("provenance", {}),
                )
            )
        known_names = {variable.name for variable in variables}
        constraints = []
        for index, raw in enumerate(model.get("constraints", ())):
            lhs = raw.get("lhs", raw.get("terms", {}))
            lhs_expression = _add(
                _linear_expr(lhs, known_names),
                LinearExpr(constant=raw.get("constant", 0)),
            )
            rhs_expression = _linear_expr(raw.get("rhs"), known_names)
            moved_terms = lhs_expression.terms + tuple(
                (name, -coefficient) for name, coefficient in rhs_expression.terms
            )
            constraints.append(
                LinearConstraint(
                    name=str(raw.get("name", f"c{index}")),
                    lhs=LinearExpr(moved_terms, lhs_expression.constant),
                    sense=str(raw.get("sense", raw.get("op", "=="))),
                    rhs=rhs_expression.constant,
                    group=raw.get("group"),
                    provenance=raw.get("provenance", {}),
                )
            )
        objective = None
        raw_objective = model.get("objective")
        if raw_objective:
            terms = raw_objective.get("terms", raw_objective.get("lhs", {}))
            objective = Objective(
                str(raw_objective.get("sense", "min")),
                _add(
                    _linear_expr(terms, known_names),
                    LinearExpr(
                        constant=raw_objective.get("constant", raw_objective.get("const", 0))
                    ),
                ),
                raw_objective.get("provenance", {}),
            )
        return ModelIR(
            tuple(variables),
            tuple(constraints),
            objective,
            str(model.get("name", "model")),
            model.get("provenance", {}),
        )


class DictResultAdapter:
    def to_result(self, result: Mapping[str, Any]) -> ResultRecord:
        return ResultRecord(
            point=result.get(
                "point", result.get("solution", result.get("assignment", {}))
            ),
            value=result.get(
                "value", result.get("objective", result.get("objective_value"))
            ),
            status=result.get("status"),
            provenance=result.get("provenance", {}),
        )


class DictSourceAdapter:
    def obligations(self, source: Any) -> tuple[SourceObligation, ...]:
        if isinstance(source, Mapping):
            raw_obligations = source.get("obligations", ())
        else:
            raw_obligations = source
        obligations: list[SourceObligation] = []
        for index, raw in enumerate(raw_obligations):
            row = raw.get("constraint", raw)
            lhs = row.get("lhs", row.get("terms", {}))
            lhs_expression = _add(
                _linear_expr(lhs), LinearExpr(constant=row.get("constant", 0))
            )
            rhs_expression = _linear_expr(row.get("rhs"))
            constraint = LinearConstraint(
                name=str(row.get("name", f"source_c{index}")),
                lhs=LinearExpr(
                    lhs_expression.terms
                    + tuple((name, -coefficient) for name, coefficient in rhs_expression.terms),
                    lhs_expression.constant,
                ),
                sense=str(row.get("sense", row.get("op", "=="))),
                rhs=rhs_expression.constant,
                group=row.get("group"),
                provenance=row.get("provenance", {}),
            )
            obligations.append(
                SourceObligation(
                    claim_id=str(raw.get("claim_id", constraint.name)),
                    constraint=constraint,
                    source_span=raw.get("source_span"),
                    applicability=bool(raw.get("applicability", True)),
                    provenance=raw.get("provenance", {}),
                )
            )
        return tuple(obligations)
