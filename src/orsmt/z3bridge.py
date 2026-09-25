from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Mapping

import z3

from .ir import LinearConstraint, LinearExpr, ModelIR, Number, as_fraction


def q(value: Number) -> z3.ArithRef:
    fraction = as_fraction(value)
    return z3.Q(fraction.numerator, fraction.denominator)


def value_of(model: z3.ModelRef, expression: z3.ArithRef) -> Fraction:
    value = model.eval(expression, model_completion=True)
    if z3.is_int_value(value):
        return Fraction(value.as_long())
    if z3.is_rational_value(value):
        return Fraction(value.numerator_as_long(), value.denominator_as_long())
    raise ValueError(f"non-rational model value: {value}")


def linear_expr(expression: LinearExpr, variables: Mapping[str, z3.ArithRef]) -> z3.ArithRef:
    result: z3.ArithRef = q(expression.constant)
    for name, coefficient in expression.terms:
        result = result + q(coefficient) * variables[name]
    return result


def row_expr(constraint: LinearConstraint, variables: Mapping[str, z3.ArithRef]) -> z3.BoolRef:
    lhs = linear_expr(constraint.lhs, variables)
    rhs = q(constraint.rhs)
    if constraint.sense == "<=":
        return lhs <= rhs
    if constraint.sense == ">=":
        return lhs >= rhs
    return lhs == rhs


def make_variables(ir: ModelIR, prefix: str = "") -> dict[str, z3.ArithRef]:
    return {variable.name: z3.Real(f"{prefix}{variable.name}") for variable in ir.variables}


def model_formula(ir: ModelIR, variables: Mapping[str, z3.ArithRef]) -> z3.BoolRef:
    clauses: list[z3.BoolRef] = []
    for variable in ir.variables:
        symbol = variables[variable.name]
        if variable.kind in {"integer", "binary"}:
            clauses.append(z3.IsInt(symbol))
        if variable.kind == "binary":
            clauses.extend((symbol >= 0, symbol <= 1))
        if variable.lb is not None:
            clauses.append(symbol >= q(variable.lb))
        if variable.ub is not None:
            clauses.append(symbol <= q(variable.ub))
    clauses.extend(row_expr(constraint, variables) for constraint in ir.constraints)
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


def projected_formula(ir: ModelIR, variables: Mapping[str, z3.ArithRef]) -> z3.BoolRef:
    formula = model_formula(ir, variables)
    auxiliary = [variables[name] for name in ir.auxiliary_names]
    return z3.Exists(auxiliary, formula) if auxiliary else formula


def point_from_model(
    model: z3.ModelRef, variables: Mapping[str, z3.ArithRef], names: Iterable[str]
) -> dict[str, Fraction]:
    return {name: value_of(model, variables[name]) for name in names}


def point_constraints(
    variables: Mapping[str, z3.ArithRef], point: Mapping[str, Number]
) -> list[z3.BoolRef]:
    return [variables[name] == q(value) for name, value in point.items() if name in variables]
