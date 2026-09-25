from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import wraps
from time import perf_counter
from typing import Iterable, Mapping

import z3

from .adapters.protocols import NativeReplayAdapter
from .evidence import EvidenceRecord, Relation, ReplayRecord, Verdict
from .ir import LinearConstraint, LinearExpr, ModelIR, ResultRecord, SourceObligation, as_fraction
from .z3bridge import (
    linear_expr,
    make_variables,
    model_formula,
    point_constraints,
    point_from_model,
    projected_formula,
    q,
    row_expr,
)


@dataclass(frozen=True)
class Completion:
    status: z3.CheckSatResult
    point: Mapping[str, Fraction]


def _timed(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        started = perf_counter()
        result = method(*args, **kwargs)
        elapsed = (perf_counter() - started) * 1_000
        if isinstance(result, EvidenceRecord):
            return replace(result, duration_ms=elapsed)
        if isinstance(result, tuple) and all(isinstance(item, EvidenceRecord) for item in result):
            return tuple(replace(item, duration_ms=elapsed) for item in result)
        return result

    return wrapped


def violation_margin(constraint: LinearConstraint, point: Mapping[str, Fraction]) -> Fraction:
    lhs = constraint.lhs.evaluate(point)
    if constraint.sense == "<=":
        return max(Fraction(0), lhs - constraint.rhs)
    if constraint.sense == ">=":
        return max(Fraction(0), constraint.rhs - lhs)
    return abs(lhs - constraint.rhs)


def normalized_violation(
    constraint: LinearConstraint, point: Mapping[str, Fraction]
) -> float:
    raw = violation_margin(constraint, point)
    activity = abs(constraint.lhs.constant - constraint.rhs)
    activity += sum(abs(coefficient * as_fraction(point[name])) for name, coefficient in constraint.lhs.terms)
    return float(raw / max(Fraction(1), activity))


class Verifier:
    """Exact relational checks for linear mixed-integer optimization models.

    A SAT result becomes a refutation only after the extracted point is replayed
    against the complete model. UNKNOWN is never collapsed into a failure.
    """

    def __init__(
        self, timeout_ms: int = 10_000, native_replay: NativeReplayAdapter | None = None
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_ms = timeout_ms
        self.native_replay = native_replay

    def _solver(self) -> z3.Solver:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        return solver

    def _complete(self, ir: ModelIR, partial: Mapping[str, Fraction]) -> Completion:
        variables = make_variables(ir, "replay::")
        unknown = set(partial) - set(variables)
        if unknown:
            return Completion(z3.unknown, {})
        solver = self._solver()
        solver.add(model_formula(ir, variables), *point_constraints(variables, partial))
        status = solver.check()
        if status != z3.sat:
            return Completion(status, {})
        return Completion(status, point_from_model(solver.model(), variables, variables))

    @staticmethod
    def _replay(passed: bool, detail: str) -> ReplayRecord:
        return ReplayRecord(passed=passed, backend="z3-exact", detail=detail)

    def _native_check(
        self,
        ir: ModelIR,
        point: Mapping[str, Fraction],
        expected_feasible: bool,
        detail: str,
    ) -> ReplayRecord:
        if self.native_replay is None:
            return self._replay(True, detail)
        try:
            feasible, native_detail = self.native_replay.replay(ir, point)
        except Exception as error:  # adapter failures do not become semantic refutations
            return ReplayRecord(
                False,
                "z3-exact+native",
                f"native replay raised {type(error).__name__}",
            )
        if feasible is None:
            return ReplayRecord(False, "z3-exact+native", native_detail or "native replay undecided")
        passed = feasible is expected_feasible
        suffix = native_detail or (
            "native feasible" if feasible else "native infeasible"
        )
        return ReplayRecord(passed, "z3-exact+native", f"{detail}; {suffix}")

    def _base_status(self, ir: ModelIR) -> z3.CheckSatResult:
        variables = make_variables(ir, "base::")
        solver = self._solver()
        solver.add(model_formula(ir, variables))
        return solver.check()

    @_timed
    def entails(
        self,
        ir: ModelIR,
        obligation: SourceObligation,
        relation: Relation = Relation.FORMALIZATION,
    ) -> EvidenceRecord:
        claim = obligation.constraint
        query = f"{ir.name} entails {obligation.claim_id}"
        fault_metadata = {"candidate_fault_if_refuted": "underconstraint"}
        if not obligation.applicability:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                source_span=obligation.source_span,
                reason="not_applicable",
                metadata=fault_metadata,
            )
        unknown = claim.support - set(ir.variable_map)
        if unknown:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                reason=f"unknown_variables:{','.join(sorted(unknown))}",
                metadata=fault_metadata,
            )

        base = self._base_status(ir)
        if base != z3.sat:
            reason = "inconsistent_model" if base == z3.unsat else "solver_unknown"
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                reason=reason,
                metadata=fault_metadata,
            )

        variables = make_variables(ir, "entail::")
        solver = self._solver()
        solver.add(model_formula(ir, variables), z3.Not(row_expr(claim, variables)))
        status = solver.check()
        if status == z3.unsat:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.CERTIFIED,
                query,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                replay=self._replay(True, "counterexample query is unsatisfiable"),
                metadata=fault_metadata,
            )
        if status == z3.unknown:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                reason="solver_unknown",
                metadata=fault_metadata,
            )

        witness = point_from_model(solver.model(), variables, ir.observable_names)
        completion = self._complete(ir, witness)
        if completion.status != z3.sat:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                replay=self._replay(False, "candidate witness did not replay"),
                reason="replay_failed",
                metadata=fault_metadata,
            )
        margin = violation_margin(claim, completion.point)
        if margin <= 0:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                replay=self._replay(False, "replayed point does not violate the obligation"),
                reason="replay_failed",
                metadata=fault_metadata,
            )
        replay = self._native_check(
            ir,
            completion.point,
            True,
            "feasible full completion violates the obligation",
        )
        if not replay.passed:
            return EvidenceRecord(
                relation,
                obligation.claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                implicated=(claim.name,),
                source_span=obligation.source_span,
                violation_margin=margin,
                normalized_violation=normalized_violation(claim, completion.point),
                replay=replay,
                reason="native_replay_failed",
                metadata=fault_metadata,
            )
        return EvidenceRecord(
            relation,
            obligation.claim_id,
            Verdict.REFUTED,
            query,
            witness=witness,
            implicated=(claim.name,),
            source_span=obligation.source_span,
            violation_margin=margin,
            normalized_violation=normalized_violation(claim, completion.point),
            replay=replay,
            metadata=fault_metadata,
        )

    def region_equivalence(
        self,
        left: ModelIR,
        right: ModelIR,
        relation: Relation = Relation.IMPLEMENTATION,
    ) -> tuple[EvidenceRecord, EvidenceRecord]:
        left_id = f"region:{left.name}\\{right.name}"
        right_id = f"region:{right.name}\\{left.name}"
        if set(left.observable_names) != set(right.observable_names):
            reason = "observable_schema_mismatch"
            return (
                EvidenceRecord(relation, left_id, Verdict.UNDECIDED, left_id, reason=reason),
                EvidenceRecord(relation, right_id, Verdict.UNDECIDED, right_id, reason=reason),
            )
        left_minus_right = self._directional_region(left, right, relation)
        right_minus_left = self._directional_region(right, left, relation)
        return (
            replace(
                left_minus_right,
                metadata={
                    **left_minus_right.metadata,
                    "region_direction": "left_minus_right",
                    "candidate_fault_if_refuted": "overconstraint",
                },
            ),
            replace(
                right_minus_left,
                metadata={
                    **right_minus_left.metadata,
                    "region_direction": "right_minus_left",
                    "candidate_fault_if_refuted": "underconstraint",
                },
            ),
        )

    @_timed
    def _directional_region(
        self, host: ModelIR, other: ModelIR, relation: Relation
    ) -> EvidenceRecord:
        claim_id = f"region:{host.name}\\{other.name}"
        query = f"exists observable x in {host.name} but not {other.name}"
        host_observables = {
            name: host.variable_map[name].kind for name in host.observable_names
        }
        other_observables = {
            name: other.variable_map[name].kind for name in other.observable_names
        }
        if not other.auxiliary_names and host_observables == other_observables:
            return self._directional_region_rows(host, other, relation, claim_id, query)

        observable = {name: z3.Real(f"region::{name}") for name in host.observable_names}
        host_vars = {
            variable.name: observable.get(variable.name, z3.Real(f"host::{variable.name}"))
            for variable in host.variables
        }
        other_vars = {
            variable.name: observable.get(variable.name, z3.Real(f"other::{variable.name}"))
            for variable in other.variables
        }
        solver = self._solver()
        solver.add(model_formula(host, host_vars), z3.Not(projected_formula(other, other_vars)))
        status = solver.check()
        if status == z3.unsat:
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.CERTIFIED,
                query,
                replay=self._replay(True, "directional set difference is empty"),
                metadata={"host": host.name, "excluded_from": other.name},
            )
        if status == z3.unknown:
            return EvidenceRecord(
                relation, claim_id, Verdict.UNDECIDED, query, reason="solver_unknown"
            )

        witness = point_from_model(solver.model(), observable, host.observable_names)
        host_replay = self._complete(host, witness)
        other_replay = self._complete(other, witness)
        passed = host_replay.status == z3.sat and other_replay.status == z3.unsat
        if not passed:
            reason = "solver_unknown" if other_replay.status == z3.unknown else "replay_failed"
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                replay=self._replay(False, "directional witness failed complete replay"),
                reason=reason,
            )
        native_host = self._native_check(host, witness, True, f"feasible in {host.name}")
        native_other = self._native_check(
            other, witness, False, f"infeasible in {other.name}"
        )
        if not (native_host.passed and native_other.passed):
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                replay=ReplayRecord(
                    False,
                    "z3-exact+native",
                    f"{native_host.detail}; {native_other.detail}",
                ),
                reason="native_replay_failed",
            )
        return EvidenceRecord(
            relation,
            claim_id,
            Verdict.REFUTED,
            query,
            witness=witness,
            replay=ReplayRecord(
                True,
                native_host.backend,
                f"{native_host.detail}; {native_other.detail}; complete directional replay",
            ),
            metadata={"host": host.name, "excluded_from": other.name},
        )

    def _directional_region_rows(
        self,
        host: ModelIR,
        other: ModelIR,
        relation: Relation,
        claim_id: str,
        query: str,
    ) -> EvidenceRecord:
        """Incrementally test a fully observable target one semantic row at a time.

        This is logically equivalent to ``host ∧ ¬other`` when variable domains
        agree.  It avoids presenting a solver with one large disjunction and,
        when satisfiable, names the first violated target row directly.
        """

        observable = {name: z3.Real(f"region::{name}") for name in host.observable_names}
        host_vars = self._vars_with_shared_observables(host, observable, "region::host::")
        solver = self._solver()
        solver.add(model_formula(host, host_vars))
        base_status = solver.check()
        if base_status != z3.sat:
            reason = "inconsistent_model" if base_status == z3.unsat else "solver_unknown"
            return EvidenceRecord(relation, claim_id, Verdict.UNDECIDED, query, reason=reason)

        host_rows = self._model_rows(host)
        target_rows = self._model_rows(other)
        host_signatures = {self._row_signature(row) for row in host_rows}
        common_rows = sum(
            self._row_signature(row) in host_signatures for row in target_rows
        )

        unknown_rows: list[str] = []
        for target in target_rows:
            # A syntactically identical normalized row is already a conjunct of
            # the host formula.  Skipping it is an exact proof step, not a
            # heuristic, and prevents unchanged large models from paying one
            # SMT call per copied row.
            if self._row_signature(target) in host_signatures:
                continue
            solver.push()
            solver.add(z3.Not(row_expr(target, observable)))
            status = solver.check()
            if status == z3.unknown:
                unknown_rows.append(target.name)
                solver.pop()
                continue
            if status == z3.unsat:
                solver.pop()
                continue

            witness = point_from_model(solver.model(), observable, host.observable_names)
            solver.pop()
            host_replay = self._complete(host, witness)
            other_replay = self._complete(other, witness)
            passed = host_replay.status == z3.sat and other_replay.status == z3.unsat
            if not passed:
                reason = "solver_unknown" if other_replay.status == z3.unknown else "replay_failed"
                return EvidenceRecord(
                    relation,
                    claim_id,
                    Verdict.UNDECIDED,
                    query,
                    witness=witness,
                    implicated=(target.name,),
                    replay=self._replay(False, "row witness failed complete replay"),
                    reason=reason,
                )
            native_host = self._native_check(host, witness, True, f"feasible in {host.name}")
            native_other = self._native_check(other, witness, False, f"infeasible in {other.name}")
            if not (native_host.passed and native_other.passed):
                return EvidenceRecord(
                    relation,
                    claim_id,
                    Verdict.UNDECIDED,
                    query,
                    witness=witness,
                    implicated=(target.name,),
                    replay=ReplayRecord(
                        False,
                        "z3-exact+native",
                        f"{native_host.detail}; {native_other.detail}",
                    ),
                    reason="native_replay_failed",
                )
            margin = violation_margin(target, witness)
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.REFUTED,
                query,
                witness=witness,
                implicated=(target.name,),
                violation_margin=margin,
                normalized_violation=normalized_violation(target, witness),
                replay=ReplayRecord(
                    True,
                    native_host.backend,
                    f"{native_host.detail}; {native_other.detail}; violated row {target.name}",
                ),
                metadata={
                    "host": host.name,
                    "excluded_from": other.name,
                    "violated_target_row": target.name,
                    "query_strategy": "incremental_rows",
                },
            )

        if unknown_rows:
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="solver_unknown",
                metadata={
                    "unknown_rows": tuple(unknown_rows),
                    "query_strategy": "incremental_rows",
                },
            )
        return EvidenceRecord(
            relation,
            claim_id,
            Verdict.CERTIFIED,
            query,
            replay=self._replay(True, "every target row is entailed incrementally"),
                metadata={
                    "host": host.name,
                    "excluded_from": other.name,
                    "query_strategy": "incremental_rows",
                    "rows_checked": len(target_rows) - common_rows,
                    "common_rows_discharged": common_rows,
                    "target_rows": len(target_rows),
                },
            )

    @staticmethod
    def _model_rows(ir: ModelIR) -> list[LinearConstraint]:
        rows: list[LinearConstraint] = []
        for variable in ir.variables:
            if variable.lb is not None:
                rows.append(
                    LinearConstraint(
                        f"bound:{variable.name}:lb",
                        LinearExpr.from_mapping({variable.name: 1}),
                        ">=",
                        variable.lb,
                    )
                )
            if variable.ub is not None:
                rows.append(
                    LinearConstraint(
                        f"bound:{variable.name}:ub",
                        LinearExpr.from_mapping({variable.name: 1}),
                        "<=",
                        variable.ub,
                    )
                )
        rows.extend(ir.constraints)
        return rows

    @staticmethod
    def _row_signature(
        row: LinearConstraint,
    ) -> tuple[str, tuple[tuple[str, Fraction], ...], Fraction]:
        """Canonicalize a linear row modulo semantically inert scaling/name.

        Inequalities are oriented as ``<= 0`` and divided by a positive
        magnitude. Equalities additionally admit a sign flip. This is used
        only for exact copied-row discharge; all nonmatches still reach SMT.
        """

        terms = dict(row.lhs.terms)
        constant = row.lhs.constant - row.rhs
        sense = row.sense
        if sense == ">=":
            terms = {name: -coefficient for name, coefficient in terms.items()}
            constant = -constant
            sense = "<="
        ordered = tuple(sorted((name, coefficient) for name, coefficient in terms.items() if coefficient))
        values = [coefficient for _name, coefficient in ordered]
        if constant:
            values.append(constant)
        if not values:
            return sense, (), Fraction(0)
        scale = abs(values[0])
        normalized_terms = tuple(
            (name, coefficient / scale) for name, coefficient in ordered
        )
        normalized_constant = constant / scale
        if sense == "==" and values[0] < 0:
            normalized_terms = tuple(
                (name, -coefficient) for name, coefficient in normalized_terms
            )
            normalized_constant = -normalized_constant
        return sense, normalized_terms, normalized_constant

    @staticmethod
    def _objective_score(ir: ModelIR, variables: Mapping[str, z3.ArithRef]) -> z3.ArithRef:
        assert ir.objective is not None
        expression = linear_expr(ir.objective.expression, variables)
        return expression if ir.objective.sense == "min" else -expression

    @staticmethod
    def _direct_preference_scale(left: ModelIR, right: ModelIR) -> Fraction | None:
        """Return the positive score scale when it is visible syntactically.

        Constants never affect ordering.  This shortcut is deliberately limited
        to objectives over the shared observable variables; auxiliary-dependent
        objectives continue through the complete SMT query below.
        """

        assert left.objective is not None and right.objective is not None
        observable = set(left.observable_names)
        if not left.objective.expression.support <= observable:
            return None
        if not right.objective.expression.support <= observable:
            return None
        left_sign = Fraction(1) if left.objective.sense == "min" else Fraction(-1)
        right_sign = Fraction(1) if right.objective.sense == "min" else Fraction(-1)
        left_terms = {
            name: coefficient * left_sign
            for name, coefficient in left.objective.expression.terms
        }
        right_terms = {
            name: coefficient * right_sign
            for name, coefficient in right.objective.expression.terms
        }
        scale: Fraction | None = None
        for name in sorted(observable):
            left_coefficient = left_terms.get(name, Fraction(0))
            right_coefficient = right_terms.get(name, Fraction(0))
            if not left_coefficient and not right_coefficient:
                continue
            if not left_coefficient or not right_coefficient:
                return None
            candidate = right_coefficient / left_coefficient
            if candidate <= 0:
                return None
            if scale is None:
                scale = candidate
            elif candidate != scale:
                return None
        return scale if scale is not None else Fraction(1)

    @staticmethod
    def _vars_with_shared_observables(
        ir: ModelIR, shared: Mapping[str, z3.ArithRef], prefix: str
    ) -> dict[str, z3.ArithRef]:
        return {
            variable.name: shared.get(variable.name, z3.Real(f"{prefix}{variable.name}"))
            for variable in ir.variables
        }

    @_timed
    def preference_equivalence(
        self,
        left: ModelIR,
        right: ModelIR,
        relation: Relation = Relation.IMPLEMENTATION,
    ) -> EvidenceRecord:
        claim_id = f"preference:{left.name}~{right.name}"
        query = f"exists x,y with different objective ordering in {left.name} and {right.name}"
        if left.objective is None or right.objective is None:
            return EvidenceRecord(
                relation, claim_id, Verdict.UNDECIDED, query, reason="missing_objective"
            )
        if set(left.observable_names) != set(right.observable_names):
            return EvidenceRecord(
                relation, claim_id, Verdict.UNDECIDED, query, reason="observable_schema_mismatch"
            )
        direct_scale = self._direct_preference_scale(left, right)
        if direct_scale is not None:
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.CERTIFIED,
                query,
                replay=self._replay(
                    True, "objectives induce the same order by an exact positive score scale"
                ),
                metadata={
                    "candidate_fault_if_refuted": "preference",
                    "query_strategy": "positive_affine_score",
                    "score_scale": direct_scale,
                },
            )
        x = {name: z3.Real(f"pref::x::{name}") for name in left.observable_names}
        y = {name: z3.Real(f"pref::y::{name}") for name in left.observable_names}
        lx = self._vars_with_shared_observables(left, x, "pref::lx::")
        ly = self._vars_with_shared_observables(left, y, "pref::ly::")
        rx = self._vars_with_shared_observables(right, x, "pref::rx::")
        ry = self._vars_with_shared_observables(right, y, "pref::ry::")
        da = self._objective_score(left, lx) - self._objective_score(left, ly)
        db = self._objective_score(right, rx) - self._objective_score(right, ry)
        disagreement = z3.Or(
            z3.And(da < 0, db >= 0),
            z3.And(da == 0, db != 0),
            z3.And(da > 0, db <= 0),
        )
        shared_region = z3.And(
            model_formula(left, lx),
            model_formula(left, ly),
            model_formula(right, rx),
            model_formula(right, ry),
        )
        solver = self._solver()
        solver.add(shared_region, disagreement)
        status = solver.check()
        if status == z3.unsat:
            # Distinguish a genuine ordering proof from a vacuous empty intersection.
            base = self._solver()
            base.add(model_formula(left, lx), model_formula(right, rx))
            base_status = base.check()
            if base_status != z3.sat:
                reason = "empty_shared_region" if base_status == z3.unsat else "solver_unknown"
                return EvidenceRecord(
                    relation, claim_id, Verdict.UNDECIDED, query, reason=reason
                )
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.CERTIFIED,
                query,
                replay=self._replay(True, "no ordering disagreement exists on the shared region"),
                metadata={"candidate_fault_if_refuted": "preference"},
            )
        if status == z3.unknown:
            return EvidenceRecord(
                relation, claim_id, Verdict.UNDECIDED, query, reason="solver_unknown"
            )

        model = solver.model()
        x_point = point_from_model(model, x, left.observable_names)
        y_point = point_from_model(model, y, left.observable_names)
        left_x = point_from_model(model, lx, lx)
        left_y = point_from_model(model, ly, ly)
        right_x = point_from_model(model, rx, rx)
        right_y = point_from_model(model, ry, ry)
        replays = (
            self._complete(left, left_x),
            self._complete(left, left_y),
            self._complete(right, right_x),
            self._complete(right, right_y),
        )
        passed = all(replay.status == z3.sat for replay in replays)
        witness = {
            **{f"observable.x.{k}": v for k, v in x_point.items()},
            **{f"observable.y.{k}": v for k, v in y_point.items()},
            **{f"left.x.{k}": v for k, v in left_x.items()},
            **{f"left.y.{k}": v for k, v in left_y.items()},
            **{f"right.x.{k}": v for k, v in right_x.items()},
            **{f"right.y.{k}": v for k, v in right_y.items()},
        }
        if not passed:
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                replay=self._replay(False, "ordering witness failed complete replay"),
                reason="replay_failed",
            )
        native_checks = [
            self._native_check(left, left_x, True, f"{left.name} accepts x preference witness"),
            self._native_check(left, left_y, True, f"{left.name} accepts y preference witness"),
            self._native_check(right, right_x, True, f"{right.name} accepts x preference witness"),
            self._native_check(right, right_y, True, f"{right.name} accepts y preference witness"),
        ]
        if not all(check.passed for check in native_checks):
            return EvidenceRecord(
                relation,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                replay=ReplayRecord(
                    False,
                    "z3-exact+native",
                    "; ".join(check.detail for check in native_checks),
                ),
                reason="native_replay_failed",
            )
        return EvidenceRecord(
            relation,
            claim_id,
            Verdict.REFUTED,
            query,
            witness=witness,
            implicated=("objective",),
            replay=ReplayRecord(
                True,
                native_checks[0].backend,
                "both points replay in both models with different ordering",
            ),
            metadata={"candidate_fault_if_refuted": "preference"},
        )

    @_timed
    def objective_value_equivalence(
        self, left: ModelIR, right: ModelIR, *,
        right_scale: Fraction = Fraction(1), right_offset: Fraction = Fraction(0),
        relation: Relation = Relation.IMPLEMENTATION,
    ) -> EvidenceRecord:
        """Compare objective values, optionally using an explicitly supplied unit map.

        This is separate from preference equivalence: positive affine changes
        preserve ordering, but need not preserve the reported cost or reward.
        The map is left_value = right_scale * right_value + right_offset.
        """
        scale, offset = as_fraction(right_scale), as_fraction(right_offset)
        if scale == 0:
            raise ValueError('objective value map must be invertible')
        claim_id = f'objective_value:{left.name}~{right.name}'
        query = f'exists shared feasible decision with unequal mapped objective values in {left.name} and {right.name}'
        metadata = {'right_scale': scale, 'right_offset': offset,
                    'candidate_fault_if_refuted': 'objective_value'}
        def record(verdict, **kwargs):
            return EvidenceRecord(relation, claim_id, verdict, query,
                                  metadata=metadata, **kwargs)
        if left.objective is None or right.objective is None:
            return record(Verdict.UNDECIDED, reason='missing_objective')
        if set(left.observable_names) != set(right.observable_names):
            return record(Verdict.UNDECIDED, reason='observable_schema_mismatch')
        shared = {name: z3.Real(f'objval::x::{name}') for name in left.observable_names}
        lv = self._vars_with_shared_observables(left, shared, 'objval::left::')
        rv = self._vars_with_shared_observables(right, shared, 'objval::right::')
        solver = self._solver()
        solver.add(model_formula(left, lv), model_formula(right, rv))
        base = solver.check()
        if base != z3.sat:
            return record(Verdict.UNDECIDED, reason='empty_shared_region' if base == z3.unsat else 'solver_unknown')
        lf = linear_expr(left.objective.expression, lv)
        rf = linear_expr(right.objective.expression, rv)
        solver.add(lf != q(scale) * rf + q(offset))
        status = solver.check()
        if status == z3.unsat:
            return record(Verdict.CERTIFIED,
                          replay=self._replay(True, 'all shared feasible completions have equal mapped objective values'))
        if status != z3.sat:
            return record(Verdict.UNDECIDED, reason='solver_unknown')
        lp = point_from_model(solver.model(), lv, lv)
        rp = point_from_model(solver.model(), rv, rv)
        a, b = left.objective.expression.evaluate(lp), right.objective.expression.evaluate(rp)
        replay_left = self._native_check(left, lp, True, 'objective-value witness feasible in left model')
        replay_right = self._native_check(right, rp, True, 'objective-value witness feasible in right model')
        metadata.update(left_value=a, right_value=b, mapped_right_value=scale*b+offset,
                        left_completion=lp, right_completion=rp)
        if not replay_left.passed or not replay_right.passed:
            return record(Verdict.UNDECIDED, reason='native_replay_failed')
        if a == scale*b+offset:
            return record(Verdict.UNDECIDED, reason='witness_value_mismatch')
        return record(Verdict.REFUTED,
                      witness={n: lp[n] for n in left.observable_names},
                      implicated=('objective',), violation_margin=abs(a-scale*b-offset),
                      replay=self._replay(True, 'complete native-replayed witness has different mapped objective values'))

    def verify_result(
        self, ir: ModelIR, result: ResultRecord
    ) -> tuple[EvidenceRecord, EvidenceRecord, EvidenceRecord]:
        feasibility = self._verify_feasibility(ir, result)
        value = self._verify_value_binding(ir, result, feasibility)
        optimality = self._verify_no_better(ir, result)
        return feasibility, value, optimality

    @_timed
    def _verify_feasibility(self, ir: ModelIR, result: ResultRecord) -> EvidenceRecord:
        claim_id = "result:feasible"
        query = f"reported point extends to a feasible point of {ir.name}"
        completion = self._complete(ir, result.point)
        if completion.status == z3.sat:
            replay = self._native_check(
                ir,
                completion.point,
                True,
                "reported values admit a complete feasible replay",
            )
            if not replay.passed:
                return EvidenceRecord(
                    Relation.REPORTING,
                    claim_id,
                    Verdict.UNDECIDED,
                    query,
                    witness=result.point,
                    replay=replay,
                    reason="native_replay_failed",
                )
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.CERTIFIED,
                query,
                witness=result.point,
                replay=replay,
            )
        if completion.status == z3.unknown:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=result.point,
                reason="solver_unknown_or_unknown_variable",
            )

        implicated: list[str] = []
        margins: list[Fraction] = []
        for constraint in ir.constraints:
            if constraint.support <= set(result.point):
                margin = violation_margin(constraint, result.point)
                if margin > 0:
                    implicated.append(constraint.name)
                    margins.append(margin)
        return EvidenceRecord(
            Relation.REPORTING,
            claim_id,
            Verdict.REFUTED,
            query,
            witness=result.point,
            implicated=tuple(implicated),
            violation_margin=max(margins) if margins else None,
            normalized_violation=(
                max(
                    normalized_violation(constraint, result.point)
                    for constraint in ir.constraints
                    if constraint.name in implicated
                )
                if implicated
                else None
            ),
            replay=self._replay(True, "no full feasible completion exists"),
        )

    @_timed
    def _verify_value_binding(
        self, ir: ModelIR, result: ResultRecord, feasibility: EvidenceRecord
    ) -> EvidenceRecord:
        claim_id = "result:value_binding"
        query = f"reported value equals the objective at the reported point in {ir.name}"
        if ir.objective is None or result.value is None:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="not_applicable",
            )
        if feasibility.verdict != Verdict.CERTIFIED:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="feasibility_not_certified",
            )
        variables = make_variables(ir, "value::")
        solver = self._solver()
        objective = linear_expr(ir.objective.expression, variables)
        solver.add(
            model_formula(ir, variables),
            *point_constraints(variables, result.point),
            objective != q(result.value),
        )
        status = solver.check()
        if status == z3.unsat:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.CERTIFIED,
                query,
                witness=result.point,
                replay=self._replay(True, "all feasible completions bind to the reported value"),
            )
        if status == z3.unknown:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="solver_unknown",
            )
        completion = point_from_model(solver.model(), variables, variables)
        actual = ir.objective.expression.evaluate(completion)
        replay = self._native_check(
            ir, completion, True, f"replayed objective is {actual}, not {result.value}"
        )
        if not replay.passed:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=completion,
                implicated=("objective",),
                replay=replay,
                reason="native_replay_failed",
            )
        return EvidenceRecord(
            Relation.REPORTING,
            claim_id,
            Verdict.REFUTED,
            query,
            witness=completion,
            implicated=("objective",),
            violation_margin=abs(actual - result.value),
            normalized_violation=(
                float(abs(actual - result.value) / max(Fraction(1), abs(actual), abs(result.value)))
            ),
            replay=replay,
        )

    @_timed
    def _verify_no_better(self, ir: ModelIR, result: ResultRecord) -> EvidenceRecord:
        claim_id = "result:no_better"
        query = f"no feasible point of {ir.name} improves on the reported value"
        if ir.objective is None or result.value is None:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="not_applicable",
            )
        if result.status and result.status.lower() not in {
            "optimal",
            "opt",
            "certified_optimal",
            "certified optimal",
        }:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="not_applicable",
            )
        variables = make_variables(ir, "better::")
        objective = linear_expr(ir.objective.expression, variables)
        better = objective < q(result.value) if ir.objective.sense == "min" else objective > q(result.value)
        solver = self._solver()
        solver.add(model_formula(ir, variables), better)
        status = solver.check()
        if status == z3.unsat:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.CERTIFIED,
                query,
                replay=self._replay(True, "strictly-better region is empty"),
            )
        if status == z3.unknown:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                reason="solver_unknown",
            )
        witness = point_from_model(solver.model(), variables, variables)
        actual = ir.objective.expression.evaluate(witness)
        margin = result.value - actual if ir.objective.sense == "min" else actual - result.value
        replay = self._native_check(
            ir, witness, True, f"feasible point improves the value to {actual}"
        )
        if not replay.passed:
            return EvidenceRecord(
                Relation.REPORTING,
                claim_id,
                Verdict.UNDECIDED,
                query,
                witness=witness,
                implicated=("objective",),
                replay=replay,
                reason="native_replay_failed",
            )
        return EvidenceRecord(
            Relation.REPORTING,
            claim_id,
            Verdict.REFUTED,
            query,
            witness=witness,
            implicated=("objective",),
            violation_margin=margin,
            normalized_violation=(
                float(margin / max(Fraction(1), abs(actual), abs(result.value)))
            ),
            replay=replay,
        )

    def audit_obligations(
        self, ir: ModelIR, obligations: Iterable[SourceObligation]
    ) -> list[EvidenceRecord]:
        return [self.entails(ir, obligation) for obligation in obligations]

    def constraint_correspondence(
        self,
        reference: ModelIR,
        candidate: ModelIR,
        relation: Relation = Relation.FORMALIZATION,
    ) -> tuple[list[EvidenceRecord], list[EvidenceRecord]]:
        """Return semantic recall and precision queries without row matching."""

        recall: list[EvidenceRecord] = []
        for row in reference.constraints:
            record = self.entails(
                candidate,
                SourceObligation(f"reference:{row.name}", row),
                relation,
            )
            recall.append(
                replace(
                    record,
                    metadata={
                        **record.metadata,
                        "metric_role": "recall",
                        "candidate_fault_if_refuted": "underconstraint",
                    },
                )
            )
        precision: list[EvidenceRecord] = []
        for row in candidate.constraints:
            record = self.entails(
                reference,
                SourceObligation(f"candidate:{row.name}", row),
                relation,
            )
            precision.append(
                replace(
                    record,
                    metadata={
                        **record.metadata,
                        "metric_role": "precision",
                        "candidate_fault_if_refuted": "overconstraint",
                    },
                )
            )
        return recall, precision
