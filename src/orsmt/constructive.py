from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import hashlib
import json
import time
from typing import Any, Iterable, Mapping, TypeAlias

import z3

from .adapters.protocols import NativeReplayAdapter
from .ir import LinearExpr, ModelIR, Number, as_fraction
from .z3bridge import linear_expr, make_variables, model_formula, point_from_model, q


class DSLValidationError(ValueError):
    """Raised when an untrusted constructive-map payload leaves the DSL."""


class ContractStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class ReformulationVerdict(str, Enum):
    C = "C"
    R = "R"
    U = "U"


class CounterpartStatus(str, Enum):
    EXISTS = "EXISTS"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


def _exact_fraction(value: Number) -> Fraction:
    if isinstance(value, bool):
        raise DSLValidationError("booleans are not rational constants")
    if isinstance(value, float):
        raise DSLValidationError("floating-point constants are forbidden; use a decimal string")
    try:
        return as_fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise DSLValidationError(f"invalid rational constant: {value!r}") from error


@dataclass(frozen=True)
class AffineTerm:
    """An exact affine expression over a fixed source-variable vocabulary."""

    terms: tuple[tuple[str, Fraction], ...] = ()
    constant: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        merged: dict[str, Fraction] = {}
        for name, coefficient in self.terms:
            if not isinstance(name, str) or not name:
                raise DSLValidationError("affine variable names must be non-empty strings")
            value = _exact_fraction(coefficient)
            merged[name] = merged.get(name, Fraction(0)) + value
        object.__setattr__(
            self,
            "terms",
            tuple(sorted((name, value) for name, value in merged.items() if value)),
        )
        object.__setattr__(self, "constant", _exact_fraction(self.constant))

    @classmethod
    def from_mapping(
        cls,
        terms: Mapping[str, Number] | Iterable[tuple[str, Number]] = (),
        constant: Number = 0,
    ) -> "AffineTerm":
        items = terms.items() if isinstance(terms, Mapping) else terms
        return cls(
            tuple((name, _exact_fraction(coefficient)) for name, coefficient in items),
            _exact_fraction(constant),
        )

    @classmethod
    def variable(cls, name: str) -> "AffineTerm":
        return cls.from_mapping({name: 1})

    @classmethod
    def rational(cls, value: Number) -> "AffineTerm":
        return cls((), _exact_fraction(value))

    @property
    def support(self) -> frozenset[str]:
        return frozenset(name for name, _coefficient in self.terms)

    def as_linear_expr(self) -> LinearExpr:
        return LinearExpr(self.terms, self.constant)


@dataclass(frozen=True)
class LinearComparison:
    lhs: AffineTerm
    op: str
    rhs: AffineTerm

    def __post_init__(self) -> None:
        if self.op not in {"<", "<=", "==", "!=", ">=", ">"}:
            raise DSLValidationError(f"unsupported comparison operator: {self.op}")

    @property
    def support(self) -> frozenset[str]:
        return self.lhs.support | self.rhs.support


@dataclass(frozen=True)
class IteTerm:
    condition: LinearComparison
    then_term: "Term"
    else_term: "Term"


Term: TypeAlias = AffineTerm | IteTerm


@dataclass(frozen=True)
class TotalVariableMap:
    """A candidate assignment for every variable of the target model."""

    assignments: tuple[tuple[str, Term], ...]

    def __post_init__(self) -> None:
        raw = self.assignments.items() if isinstance(self.assignments, Mapping) else self.assignments
        normalized = tuple(raw)
        names = [name for name, _term in normalized]
        if any(not isinstance(name, str) or not name for name in names):
            raise DSLValidationError("mapped variable names must be non-empty strings")
        if len(names) != len(set(names)):
            raise DSLValidationError("mapped variable names must be unique")
        object.__setattr__(self, "assignments", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, assignments: Mapping[str, Term]) -> "TotalVariableMap":
        return cls(tuple(assignments.items()))

    @property
    def as_dict(self) -> dict[str, Term]:
        return dict(self.assignments)


@dataclass(frozen=True)
class ObjectiveTransport:
    """score_B = scale * score_A + offset, with scale required to be positive."""

    scale: Fraction
    offset: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale", _exact_fraction(self.scale))
        object.__setattr__(self, "offset", _exact_fraction(self.offset))


@dataclass(frozen=True)
class ConstructiveContract:
    forward: TotalVariableMap
    backward: TotalVariableMap
    objective: ObjectiveTransport


@dataclass(frozen=True)
class SemanticView:
    """Trusted observable correspondence, deliberately kept outside the candidate contract."""

    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        pairs = tuple(self.pairs)
        if not pairs:
            raise DSLValidationError("semantic view cannot be empty")
        if any(
            not isinstance(left, str) or not left or not isinstance(right, str) or not right
            for left, right in pairs
        ):
            raise DSLValidationError("semantic variable names must be non-empty strings")
        if len({left for left, _right in pairs}) != len(pairs):
            raise DSLValidationError("left semantic variables must be unique")
        if len({right for _left, right in pairs}) != len(pairs):
            raise DSLValidationError("right semantic variables must be unique")
        object.__setattr__(self, "pairs", tuple(sorted(pairs)))

    @classmethod
    def by_name(cls, left: ModelIR, right: ModelIR) -> "SemanticView":
        if set(left.observable_names) != set(right.observable_names):
            raise DSLValidationError("observable schemas do not match by name")
        return cls(tuple((name, name) for name in left.observable_names))


@dataclass(frozen=True)
class ObligationResult:
    name: str
    direction: str
    status: ContractStatus
    witness: Mapping[str, Fraction] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class CounterpartCheck:
    obligation: str
    direction: str
    status: CounterpartStatus
    witness: Mapping[str, Fraction] = field(default_factory=dict)
    reason: str | None = None
    replay_backend: str | None = None
    replay_detail: str | None = None


@dataclass(frozen=True)
class ConstructiveAudit:
    contract_status: ContractStatus
    reformulation_verdict: ReformulationVerdict
    obligations: tuple[ObligationResult, ...] = ()
    counterpart_checks: tuple[CounterpartCheck, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Fraction):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def model_fingerprint(model: ModelIR) -> str:
    """Hash the exact verification semantics bound to a mapping contract."""

    payload = {
        "variables": [
            {
                "name": variable.name,
                "kind": variable.kind,
                "lb": None if variable.lb is None else str(variable.lb),
                "ub": None if variable.ub is None else str(variable.ub),
                "observable": variable.observable,
            }
            for variable in model.variables
        ],
        "constraints": [
            {
                "name": constraint.name,
                "terms": [
                    [name, str(coefficient)]
                    for name, coefficient in constraint.lhs.terms
                ],
                "constant": str(constraint.lhs.constant),
                "sense": constraint.sense,
                "rhs": str(constraint.rhs),
            }
            for constraint in model.constraints
        ],
        "objective": (
            None
            if model.objective is None
            else {
                "sense": model.objective.sense,
                "terms": [
                    [name, str(coefficient)]
                    for name, coefficient in model.objective.expression.terms
                ],
                "constant": str(model.objective.expression.constant),
            }
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_affine(payload: Any) -> AffineTerm:
    if isinstance(payload, (int, str, Decimal, Fraction)) and not isinstance(payload, bool):
        return AffineTerm.rational(payload)
    if not isinstance(payload, Mapping) or set(payload) != {"affine"}:
        raise DSLValidationError("an affine term must contain exactly the 'affine' key")
    body = payload["affine"]
    if not isinstance(body, Mapping) or not set(body) <= {"terms", "constant"}:
        raise DSLValidationError("affine body accepts only 'terms' and 'constant'")
    terms = body.get("terms", {})
    if not isinstance(terms, Mapping):
        raise DSLValidationError("affine terms must be a variable-to-rational mapping")
    return AffineTerm.from_mapping(terms, body.get("constant", 0))


def parse_term(
    payload: Any,
    allowed_variables: Iterable[str],
    *,
    max_ite_depth: int = 4,
) -> Term:
    """Parse the JSON-shaped, non-executable mapping DSL.

    Affine payloads use ``{"affine": {"terms": {"x": "1/2"},
    "constant": "3/4"}}``.  ITE payloads use ``{"ite": {"if":
    {"lhs": <affine>, "op": "<=", "rhs": <affine>}, "then": <term>,
    "else": <term>}}``.  Floats, arbitrary code, nonlinear products, dynamic
    indices, and terms deeper than ``max_ite_depth`` are rejected.
    """

    if max_ite_depth < 0:
        raise DSLValidationError("max_ite_depth must be non-negative")
    allowed = frozenset(allowed_variables)

    def parse(node: Any, depth: int) -> Term:
        if isinstance(node, float):
            raise DSLValidationError("floating-point constants are forbidden")
        if not isinstance(node, Mapping) or set(node) == {"affine"}:
            term = _parse_affine(node)
        else:
            if set(node) != {"ite"}:
                raise DSLValidationError("a term must contain exactly 'affine' or 'ite'")
            if depth >= max_ite_depth:
                raise DSLValidationError("maximum ITE depth exceeded")
            body = node["ite"]
            if not isinstance(body, Mapping) or set(body) != {"if", "then", "else"}:
                raise DSLValidationError("ITE body requires exactly 'if', 'then', and 'else'")
            condition = body["if"]
            if not isinstance(condition, Mapping) or set(condition) != {"lhs", "op", "rhs"}:
                raise DSLValidationError("comparison requires exactly 'lhs', 'op', and 'rhs'")
            if not isinstance(condition["op"], str):
                raise DSLValidationError("comparison operator must be a string")
            term = IteTerm(
                LinearComparison(
                    _parse_affine(condition["lhs"]),
                    condition["op"],
                    _parse_affine(condition["rhs"]),
                ),
                parse(body["then"], depth + 1),
                parse(body["else"], depth + 1),
            )
        unknown = term_support(term) - allowed
        if unknown:
            raise DSLValidationError(f"term uses unknown source variables: {sorted(unknown)}")
        return term

    return parse(payload, 0)


def parse_total_map(
    payload: Mapping[str, Any],
    source_variables: Iterable[str],
    target_variables: Iterable[str],
    *,
    max_ite_depth: int = 4,
) -> TotalVariableMap:
    if not isinstance(payload, Mapping):
        raise DSLValidationError("total map must be an object")
    target = frozenset(target_variables)
    if set(payload) != target:
        missing = sorted(target - set(payload))
        extra = sorted(set(payload) - target)
        raise DSLValidationError(f"total map mismatch: missing={missing}, extra={extra}")
    return TotalVariableMap.from_mapping(
        {
            name: parse_term(term, source_variables, max_ite_depth=max_ite_depth)
            for name, term in payload.items()
        }
    )


def parse_constructive_contract(
    payload: Mapping[str, Any],
    left: ModelIR,
    right: ModelIR,
    *,
    max_ite_depth: int = 4,
) -> ConstructiveContract:
    """Parse one complete contract against immutable model schemas.

    ``forward`` must define every variable of ``right`` using only variables
    from ``left``; ``backward`` has the inverse schema. The objective payload
    must contain explicit exact ``scale`` and ``offset`` values. Positivity of
    the parsed scale remains a verification condition, so a structurally valid
    payload with a non-positive scale yields ``ContractStatus.INVALID`` rather
    than being mistaken for a parser failure.
    """

    required = {"forward", "backward", "objective"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise DSLValidationError(
            "contract requires exactly 'forward', 'backward', and 'objective'"
        )
    forward_payload = payload["forward"]
    backward_payload = payload["backward"]
    objective_payload = payload["objective"]
    if not isinstance(forward_payload, Mapping):
        raise DSLValidationError("forward map must be an object")
    if not isinstance(backward_payload, Mapping):
        raise DSLValidationError("backward map must be an object")
    if not isinstance(objective_payload, Mapping) or set(objective_payload) != {
        "scale",
        "offset",
    }:
        raise DSLValidationError("objective requires exactly 'scale' and 'offset'")

    left_names = tuple(left.variable_map)
    right_names = tuple(right.variable_map)
    return ConstructiveContract(
        forward=parse_total_map(
            forward_payload,
            source_variables=left_names,
            target_variables=right_names,
            max_ite_depth=max_ite_depth,
        ),
        backward=parse_total_map(
            backward_payload,
            source_variables=right_names,
            target_variables=left_names,
            max_ite_depth=max_ite_depth,
        ),
        objective=ObjectiveTransport(
            scale=_exact_fraction(objective_payload["scale"]),
            offset=_exact_fraction(objective_payload["offset"]),
        ),
    )


def term_support(term: Term) -> frozenset[str]:
    if isinstance(term, AffineTerm):
        return term.support
    if isinstance(term, IteTerm):
        return (
            term.condition.support
            | term_support(term.then_term)
            | term_support(term.else_term)
        )
    raise DSLValidationError(f"unsupported term node: {type(term).__name__}")


def _term_depth(term: Term) -> int:
    if isinstance(term, AffineTerm):
        return 0
    if isinstance(term, IteTerm):
        return 1 + max(_term_depth(term.then_term), _term_depth(term.else_term))
    raise DSLValidationError(f"unsupported term node: {type(term).__name__}")


def _comparison_to_z3(
    comparison: LinearComparison, variables: Mapping[str, z3.ArithRef]
) -> z3.BoolRef:
    lhs = linear_expr(comparison.lhs.as_linear_expr(), variables)
    rhs = linear_expr(comparison.rhs.as_linear_expr(), variables)
    if comparison.op == "<":
        return lhs < rhs
    if comparison.op == "<=":
        return lhs <= rhs
    if comparison.op == "==":
        return lhs == rhs
    if comparison.op == "!=":
        return lhs != rhs
    if comparison.op == ">=":
        return lhs >= rhs
    return lhs > rhs


def term_to_z3(term: Term, variables: Mapping[str, z3.ArithRef]) -> z3.ArithRef:
    if isinstance(term, AffineTerm):
        return linear_expr(term.as_linear_expr(), variables)
    if isinstance(term, IteTerm):
        return z3.If(
            _comparison_to_z3(term.condition, variables),
            term_to_z3(term.then_term, variables),
            term_to_z3(term.else_term, variables),
        )
    raise DSLValidationError(f"unsupported term node: {type(term).__name__}")


class ConstructiveVerifier:
    """Verify a bidirectional constructive correspondence by counterexample queries.

    A SAT query invalidates the proposed mapping contract.  It refutes the
    reformulation only when a second, map-independent semantic-lifting query
    proves that the witnessed source point has no counterpart in the other
    model.  This is the boundary that prevents a poor map proposal from being
    reported as a model error.
    """

    def __init__(
        self,
        timeout_ms: int = 10_000,
        max_ite_depth: int = 4,
        native_replay: NativeReplayAdapter | None = None,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if max_ite_depth < 0:
            raise ValueError("max_ite_depth must be non-negative")
        self.timeout_ms = timeout_ms
        self.max_ite_depth = max_ite_depth
        self.native_replay = native_replay

    def _solver(self) -> z3.Solver:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        return solver

    @staticmethod
    def _score(ir: ModelIR, variables: Mapping[str, z3.ArithRef]) -> z3.ArithRef:
        assert ir.objective is not None
        expression = linear_expr(ir.objective.expression, variables)
        return expression if ir.objective.sense == "min" else -expression

    def _validate(
        self,
        left: ModelIR,
        right: ModelIR,
        contract: ConstructiveContract,
        semantic_view: SemanticView,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        left_names = set(left.variable_map)
        right_names = set(right.variable_map)
        forward = contract.forward.as_dict
        backward = contract.backward.as_dict
        if set(forward) != right_names:
            errors.append(
                f"forward total map mismatch: missing={sorted(right_names - set(forward))}, "
                f"extra={sorted(set(forward) - right_names)}"
            )
        if set(backward) != left_names:
            errors.append(
                f"backward total map mismatch: missing={sorted(left_names - set(backward))}, "
                f"extra={sorted(set(backward) - left_names)}"
            )
        for direction, assignments, allowed in (
            ("forward", forward, left_names),
            ("backward", backward, right_names),
        ):
            for target, term in assignments.items():
                try:
                    unknown = term_support(term) - allowed
                    if unknown:
                        message = (
                            f"{direction} map for {target} uses unknown variables: "
                            f"{sorted(unknown)}"
                        )
                        errors.append(message)
                    if _term_depth(term) > self.max_ite_depth:
                        errors.append(f"{direction} map for {target} exceeds maximum ITE depth")
                except DSLValidationError as error:
                    errors.append(f"{direction} map for {target}: {error}")
        left_semantic = {left_name for left_name, _right_name in semantic_view.pairs}
        right_semantic = {right_name for _left_name, right_name in semantic_view.pairs}
        if left_semantic != set(left.observable_names):
            errors.append("semantic view must cover every left observable exactly once")
        if right_semantic != set(right.observable_names):
            errors.append("semantic view must cover every right observable exactly once")
        if left.objective is None or right.objective is None:
            errors.append("both models must define an objective")
        if contract.objective.scale <= 0:
            errors.append("objective transport scale must be positive")
        return tuple(errors)

    def _base_status(self, ir: ModelIR, prefix: str) -> z3.CheckSatResult:
        variables = make_variables(ir, prefix)
        solver = self._solver()
        solver.add(model_formula(ir, variables))
        return solver.check()

    def _query(
        self,
        name: str,
        direction: str,
        base: ModelIR,
        variables: Mapping[str, z3.ArithRef],
        violation: z3.BoolRef,
    ) -> ObligationResult:
        solver = self._solver()
        solver.add(model_formula(base, variables), violation)
        status = solver.check()
        if status == z3.unsat:
            return ObligationResult(name, direction, ContractStatus.VALID)
        if status == z3.unknown:
            return ObligationResult(
                name,
                direction,
                ContractStatus.UNKNOWN,
                reason=solver.reason_unknown() or "solver_unknown",
            )
        return ObligationResult(
            name,
            direction,
            ContractStatus.INVALID,
            point_from_model(solver.model(), variables, variables),
            "counterexample",
        )

    def _query_conjunction(
        self,
        name: str,
        direction: str,
        base: ModelIR,
        variables: Mapping[str, z3.ArithRef],
        target: z3.BoolRef,
    ) -> ObligationResult:
        """Check entailment of a conjunction one atom at a time.

        ``base => And(target_atoms)`` is equivalent to proving
        ``base & Not(atom)`` unsatisfiable for every atom.  Keeping the base in
        one incremental solver avoids the large disjunction produced by
        ``Not(And(...))`` and localizes the first violating row.  The complete
        loop shares one wall-clock budget, so decomposition cannot multiply the
        configured timeout.
        """

        atoms = tuple(target.children()) if z3.is_and(target) else (target,)
        solver = self._solver()
        solver.add(model_formula(base, variables))
        started = time.monotonic()
        for index, atom in enumerate(atoms):
            elapsed_ms = int((time.monotonic() - started) * 1000)
            remaining_ms = self.timeout_ms - elapsed_ms
            if remaining_ms <= 0:
                return ObligationResult(
                    name,
                    direction,
                    ContractStatus.UNKNOWN,
                    reason=f"timeout_after_conjunct:{index}/{len(atoms)}",
                )
            solver.set(timeout=max(1, remaining_ms))
            solver.push()
            solver.add(z3.Not(atom))
            status = solver.check()
            if status == z3.sat:
                witness = point_from_model(solver.model(), variables, variables)
                solver.pop()
                return ObligationResult(
                    name,
                    direction,
                    ContractStatus.INVALID,
                    witness,
                    f"counterexample_conjunct:{index}/{len(atoms)}",
                )
            if status == z3.unknown:
                reason = solver.reason_unknown() or "solver_unknown"
                solver.pop()
                return ObligationResult(
                    name,
                    direction,
                    ContractStatus.UNKNOWN,
                    reason=f"{reason}_at_conjunct:{index}/{len(atoms)}",
                )
            solver.pop()
        return ObligationResult(
            name,
            direction,
            ContractStatus.VALID,
            reason=f"entailed_conjuncts:{len(atoms)}",
        )

    def _counterpart(
        self,
        obligation: ObligationResult,
        left: ModelIR,
        right: ModelIR,
        semantic_view: SemanticView,
    ) -> CounterpartCheck:
        if obligation.direction == "forward":
            source = left
            target = right
            variables = make_variables(target, f"lift::{obligation.name}::right::")
            target_semantic_point = {
                right_name: obligation.witness[left_name]
                for left_name, right_name in semantic_view.pairs
            }
            equalities = [
                variables[name] == q(value)
                for name, value in target_semantic_point.items()
            ]
        else:
            source = right
            target = left
            variables = make_variables(target, f"lift::{obligation.name}::left::")
            target_semantic_point = {
                left_name: obligation.witness[right_name]
                for left_name, right_name in semantic_view.pairs
            }
            equalities = [
                variables[name] == q(value)
                for name, value in target_semantic_point.items()
            ]

        source_replay_point = {
            name: obligation.witness[name] for name in source.variable_map
        }

        if self.native_replay is not None:
            source_feasible, source_detail = self.native_replay.replay(
                source, source_replay_point
            )
            if source_feasible is not True:
                return CounterpartCheck(
                    obligation.name,
                    obligation.direction,
                    CounterpartStatus.UNKNOWN,
                    reason="source_native_replay_failed",
                    replay_backend="native",
                    replay_detail=source_detail,
                )
        solver = self._solver()
        solver.add(model_formula(target, variables), *equalities)
        status = solver.check()
        if status == z3.sat:
            witness = point_from_model(solver.model(), variables, variables)
            if self.native_replay is not None:
                feasible, detail = self.native_replay.replay(target, witness)
                if feasible is not True:
                    return CounterpartCheck(
                        obligation.name,
                        obligation.direction,
                        CounterpartStatus.UNKNOWN,
                        witness,
                        "counterpart_native_replay_failed",
                        "native",
                        detail,
                    )
            else:
                detail = "exact target completion found"
            return CounterpartCheck(
                obligation.name,
                obligation.direction,
                CounterpartStatus.EXISTS,
                witness,
                replay_backend="native" if self.native_replay is not None else "z3-exact",
                replay_detail=detail,
            )
        if status == z3.unsat:
            if self.native_replay is not None:
                feasible, detail = self.native_replay.replay(
                    target, target_semantic_point
                )
                if feasible is not False:
                    return CounterpartCheck(
                        obligation.name,
                        obligation.direction,
                        CounterpartStatus.UNKNOWN,
                        reason="absence_native_replay_failed",
                        replay_backend="native",
                        replay_detail=detail,
                    )
            else:
                detail = "exact target completion query is unsatisfiable"
            return CounterpartCheck(
                obligation.name,
                obligation.direction,
                CounterpartStatus.ABSENT,
                reason="no semantic counterpart exists",
                replay_backend="native" if self.native_replay is not None else "z3-exact",
                replay_detail=detail,
            )
        return CounterpartCheck(
            obligation.name,
            obligation.direction,
            CounterpartStatus.UNKNOWN,
            reason=solver.reason_unknown() or "solver_unknown",
        )

    def verify(
        self,
        left: ModelIR,
        right: ModelIR,
        contract: ConstructiveContract,
        semantic_view: SemanticView,
    ) -> ConstructiveAudit:
        errors = self._validate(left, right, contract, semantic_view)
        if errors:
            return ConstructiveAudit(
                ContractStatus.INVALID,
                ReformulationVerdict.U,
                reason="; ".join(errors),
            )

        for side, ir in (("left", left), ("right", right)):
            status = self._base_status(ir, f"constructive::base::{side}::")
            if status != z3.sat:
                reason = f"{side}_model_empty" if status == z3.unsat else f"{side}_solver_unknown"
                return ConstructiveAudit(
                    ContractStatus.UNKNOWN,
                    ReformulationVerdict.U,
                    reason=reason,
                )

        left_vars = make_variables(left, "constructive::left::")
        right_vars = make_variables(right, "constructive::right::")
        forward = {
            name: term_to_z3(term, left_vars) for name, term in contract.forward.assignments
        }
        backward = {
            name: term_to_z3(term, right_vars) for name, term in contract.backward.assignments
        }
        forward_semantic_failure = z3.Or(
            *[
                forward[right_name] != left_vars[left_name]
                for left_name, right_name in semantic_view.pairs
            ]
        )
        backward_semantic_failure = z3.Or(
            *[
                backward[left_name] != right_vars[right_name]
                for left_name, right_name in semantic_view.pairs
            ]
        )
        scale = q(contract.objective.scale)
        offset = q(contract.objective.offset)
        obligations = (
            self._query_conjunction(
                "forward_feasibility",
                "forward",
                left,
                left_vars,
                model_formula(right, forward),
            ),
            self._query_conjunction(
                "backward_feasibility",
                "backward",
                right,
                right_vars,
                model_formula(left, backward),
            ),
            self._query(
                "forward_semantics",
                "forward",
                left,
                left_vars,
                forward_semantic_failure,
            ),
            self._query(
                "backward_semantics",
                "backward",
                right,
                right_vars,
                backward_semantic_failure,
            ),
            self._query(
                "forward_objective",
                "forward",
                left,
                left_vars,
                self._score(right, forward) != scale * self._score(left, left_vars) + offset,
            ),
            self._query(
                "backward_objective",
                "backward",
                right,
                right_vars,
                self._score(left, backward) != (self._score(right, right_vars) - offset) / scale,
            ),
        )
        if any(row.status == ContractStatus.INVALID for row in obligations):
            contract_status = ContractStatus.INVALID
        elif any(row.status == ContractStatus.UNKNOWN for row in obligations):
            contract_status = ContractStatus.UNKNOWN
        else:
            contract_status = ContractStatus.VALID

        liftable = {"forward_feasibility", "backward_feasibility", "forward_semantics", "backward_semantics"}
        counterpart_checks = tuple(
            self._counterpart(row, left, right, semantic_view)
            for row in obligations
            if row.status == ContractStatus.INVALID and row.name in liftable
        )
        if contract_status == ContractStatus.VALID:
            reformulation_verdict = ReformulationVerdict.C
        elif any(check.status == CounterpartStatus.ABSENT for check in counterpart_checks):
            reformulation_verdict = ReformulationVerdict.R
        else:
            reformulation_verdict = ReformulationVerdict.U
        return ConstructiveAudit(
            contract_status,
            reformulation_verdict,
            obligations,
            counterpart_checks,
            None if contract_status == ContractStatus.VALID else "contract_not_certified",
        )
