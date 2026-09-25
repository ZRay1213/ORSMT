"""Proof-guided, sequential constraint reduction for fixed-instance MILPs.

Redundancy is an actionable normalization defect, separate from semantic
disagreement. The retained representation is deterministic for a fixed row
order; it is not a universal unique normal form for MILP reformulations.
"""
from dataclasses import replace
from time import perf_counter

import z3

from .constructive import model_fingerprint
from .evidence import EvidenceRecord, Relation, Verdict, aggregate
from .ir import SourceObligation
from .milp_codegen import model_to_dict
from .repair_workflow import compare_models
from .taxonomy import summarize_rows
from .verify import violation_margin
from .z3bridge import make_variables, model_formula, row_expr, point_from_model


class ConstraintQueries:
    """Share domain and guarded constraints, never query a vacuous base."""

    def __init__(self, model, verifier, *, incremental=True, budget_seconds=None):
        self.model, self.verifier = model, verifier
        self.incremental = incremental
        self.started = perf_counter()
        self.deadline = None if budget_seconds is None else self.started + budget_seconds
        self.variables = make_variables(model, 'redundancy::')
        self.solver = verifier._solver()
        self.guards = [z3.FreshBool('row_active') for _ in model.constraints]
        self.solver.add(model_formula(replace(model, constraints=()), self.variables))
        for guard, row in zip(self.guards, model.constraints):
            self.solver.add(z3.Implies(guard, row_expr(row, self.variables)))
        self.base = self.solver.check(*self.guards) if self._time_left() else z3.unknown

    def _time_left(self):
        if self.deadline is None:
            return True
        left = int(1000 * (self.deadline - perf_counter()))
        if left <= 0:
            return False
        self.solver.set(timeout=min(left, self.verifier.timeout_ms))
        return True

    def check(self, index, active):
        started = perf_counter()
        row = self.model.constraints[index]
        claim = f'redundancy:{index}:{row.name}'
        metadata = {'property': 'constraint_redundancy', 'row_index': index,
                    'active_row_count': len(active), 'incremental': self.incremental}

        def record(verdict, **kwargs):
            return EvidenceRecord(Relation.IMPLEMENTATION, claim, verdict,
                'remaining constraints entail the removed row', implicated=(row.name,),
                duration_ms=1000*(perf_counter()-started), metadata=metadata, **kwargs)

        if self.base != z3.sat:
            return record(Verdict.UNDECIDED, reason='infeasible_base' if self.base == z3.unsat else 'base_unknown')
        if not self._time_left():
            return record(Verdict.UNDECIDED, reason='normalization_time_budget')
        rest = replace(self.model, constraints=tuple(self.model.constraints[i] for i in sorted(active) if i != index))
        if not self.incremental:
            # Independent rebuild baseline, including its normal feasibility check.
            result = self.verifier.entails(rest, SourceObligation(claim, row), Relation.IMPLEMENTATION)
            return replace(result, metadata=metadata)
        self.solver.push()
        try:
            self.solver.add(z3.Not(row_expr(row, self.variables)))
            status = self.solver.check(*(self.guards[i] for i in sorted(active) if i != index))
            if status == z3.unsat:
                return record(Verdict.CERTIFIED,
                    replay=self.verifier._replay(True, 'remaining rows and negated row are unsatisfiable'))
            if status == z3.unknown:
                return record(Verdict.UNDECIDED, reason='solver_unknown')
            point = point_from_model(self.solver.model(), self.variables, self.variables)
        finally:
            self.solver.pop()
        margin = violation_margin(row, point)
        replay = self.verifier._native_check(rest, point, True, 'full assignment satisfies remaining rows')
        if margin <= 0 or not replay.passed:
            return record(Verdict.UNDECIDED, witness=point, replay=replay, reason='witness_replay_failed')
        return record(Verdict.REFUTED, witness=point, violation_margin=margin, replay=replay)


def audit_redundancy(model, verifier, *, incremental=True, budget_seconds=None):
    started = perf_counter()
    queries = ConstraintQueries(model, verifier, incremental=incremental, budget_seconds=budget_seconds)
    active = set(range(len(model.constraints)))
    rows = []
    for i, row in enumerate(model.constraints):
        evidence = queries.check(i, active)
        redundant = {Verdict.CERTIFIED: True, Verdict.REFUTED: False}.get(evidence.verdict)
        rows.append({'row_index': i, 'constraint': row.name, 'redundant': redundant,
                     'normalization_issue': redundant, 'evidence': evidence.to_dict()})
    return {'rows': rows, 'metrics': summarize_rows(rows, 'redundant'),
            'elapsed_seconds': perf_counter()-started, 'base_status': str(queries.base)}


def normalize_constraints(model, verifier, *, incremental=True, budget_seconds=None):
    """Delete one proved redundant row at a time; preserve all other model parts.

An essential-row witness remains valid as further constraints are removed.
Thus previously REFUTED redundancy queries need no extra residual audit.
Unknown queries retain their rows. Only a fully certified final comparison
publishes the reduced model as the accepted output.
"""
    started = perf_counter()
    queries = ConstraintQueries(model, verifier, incremental=incremental, budget_seconds=budget_seconds)
    active = set(range(len(model.constraints)))
    steps = []
    for i, row in enumerate(model.constraints):
        evidence = queries.check(i, active)
        remove = evidence.verdict == Verdict.CERTIFIED
        if remove:
            active.remove(i)
        steps.append({'row_index': i, 'constraint': row.name, 'removed': remove,
                      'evidence': evidence.to_dict()})
    proposed = replace(model, constraints=tuple(model.constraints[i] for i in sorted(active)))
    # Full-domain identity, preference and value checks also retain objective offsets.
    certificates = compare_models(verifier, replace(model, name='before'), replace(proposed, name='after'))
    accepted = aggregate(list(certificates)) == Verdict.CERTIFIED
    final = proposed if accepted else model
    removed = len(model.constraints)-len(final.constraints)
    undecided = sum(s['evidence']['verdict'] == 'UNDECIDED' for s in steps)
    return {'schema': 'orsmt-normalization-v1', 'accepted': accepted,
            'input_sha256': model_fingerprint(model), 'output_sha256': model_fingerprint(final),
            'model': model_to_dict(final), 'steps': steps,
            'final_equivalence': [r.to_dict() for r in certificates],
            'metrics': {'constraints_before': len(model.constraints), 'constraints_after': len(final.constraints),
                        'constraints_removed': removed,
                        'removed_percentage': 100*removed/len(model.constraints) if model.constraints else None,
                        'undecided_row_queries': undecided,
                        'irredundancy_certified': accepted and queries.base == z3.sat and undecided == 0},
            'elapsed_seconds': perf_counter()-started, 'base_status': str(queries.base),
            'order_policy': 'original_row_order', 'gpu_calls': 0, 'llm_calls': 0}
