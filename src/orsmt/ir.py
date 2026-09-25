from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable, Mapping


Number = int | float | str | Decimal | Fraction


def as_fraction(value: Number) -> Fraction:
    """Convert a finite decimal value without introducing binary-float noise."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _optional_fraction(value: Number | None) -> Fraction | None:
    return None if value is None else as_fraction(value)


@dataclass(frozen=True)
class Variable:
    name: str
    kind: str = "continuous"
    lb: Fraction | None = None
    ub: Fraction | None = None
    observable: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = self.kind.lower()
        if kind not in {"continuous", "integer", "binary"}:
            raise ValueError(f"unsupported variable kind: {self.kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lb", _optional_fraction(self.lb))
        object.__setattr__(self, "ub", _optional_fraction(self.ub))
        if not self.name:
            raise ValueError("variable name cannot be empty")
        if self.lb is not None and self.ub is not None and self.lb > self.ub:
            raise ValueError(f"invalid bounds for {self.name}")


@dataclass(frozen=True)
class LinearExpr:
    terms: tuple[tuple[str, Fraction], ...] = ()
    constant: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        merged: dict[str, Fraction] = {}
        for name, coefficient in self.terms:
            value = as_fraction(coefficient)
            merged[name] = merged.get(name, Fraction(0)) + value
        normalized = tuple(sorted((name, value) for name, value in merged.items() if value))
        object.__setattr__(self, "terms", normalized)
        object.__setattr__(self, "constant", as_fraction(self.constant))

    @classmethod
    def from_mapping(
        cls, terms: Mapping[str, Number] | Iterable[tuple[str, Number]], constant: Number = 0
    ) -> "LinearExpr":
        items = terms.items() if isinstance(terms, Mapping) else terms
        return cls(tuple((name, as_fraction(value)) for name, value in items), as_fraction(constant))

    @property
    def support(self) -> frozenset[str]:
        return frozenset(name for name, _ in self.terms)

    def evaluate(self, point: Mapping[str, Number]) -> Fraction:
        value = self.constant
        for name, coefficient in self.terms:
            if name not in point:
                raise KeyError(name)
            value += coefficient * as_fraction(point[name])
        return value


@dataclass(frozen=True)
class LinearConstraint:
    name: str
    lhs: LinearExpr
    sense: str
    rhs: Fraction
    group: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sense not in {"<=", ">=", "=="}:
            raise ValueError(f"unsupported constraint sense: {self.sense}")
        object.__setattr__(self, "rhs", as_fraction(self.rhs))
        if not self.name:
            raise ValueError("constraint name cannot be empty")

    @property
    def support(self) -> frozenset[str]:
        return self.lhs.support


@dataclass(frozen=True)
class Objective:
    sense: str
    expression: LinearExpr
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sense = self.sense.lower()
        if sense not in {"min", "max"}:
            raise ValueError(f"unsupported objective sense: {self.sense}")
        object.__setattr__(self, "sense", sense)


@dataclass(frozen=True)
class ModelIR:
    variables: tuple[Variable, ...]
    constraints: tuple[LinearConstraint, ...]
    objective: Objective | None = None
    name: str = "model"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = [variable.name for variable in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")
        known = set(names)
        for constraint in self.constraints:
            unknown = constraint.support - known
            if unknown:
                raise ValueError(f"constraint {constraint.name} uses unknown variables: {sorted(unknown)}")
        if self.objective:
            unknown = self.objective.expression.support - known
            if unknown:
                raise ValueError(f"objective uses unknown variables: {sorted(unknown)}")

    @property
    def variable_map(self) -> dict[str, Variable]:
        return {variable.name: variable for variable in self.variables}

    @property
    def observable_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables if variable.observable)

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables if not variable.observable)


@dataclass(frozen=True)
class SourceObligation:
    claim_id: str
    constraint: LinearConstraint
    source_span: str | None = None
    applicability: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResultRecord:
    point: Mapping[str, Fraction]
    value: Fraction | None = None
    status: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", {name: as_fraction(value) for name, value in self.point.items()})
        object.__setattr__(self, "value", _optional_fraction(self.value))
