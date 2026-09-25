"""Exact arithmetic facts for LLM fault-location review, without NL judgments."""
from .evidence import Verdict
from .verify import violation_margin


def evaluate_point(model, point):
    missing = sorted(set(model.variable_map)-set(point))
    if missing:
        return {'complete': False, 'missing_variables': missing, 'feasible': None}
    domain_errors = []
    for variable in model.variables:
        value = point[variable.name]
        if ((variable.kind != 'continuous' and value.denominator != 1)
                or (variable.kind == 'binary' and value not in {0, 1})
                or (variable.lb is not None and value < variable.lb)
                or (variable.ub is not None and value > variable.ub)):
            domain_errors.append(variable.name)
    rows = [{'constraint': row.name, 'lhs': str(row.lhs.evaluate(point)), 'relation': row.sense,
             'rhs': str(row.rhs), 'holds': violation_margin(row, point) == 0}
            for row in model.constraints]
    return {'complete': True, 'feasible': not domain_errors and all(r['holds'] for r in rows),
            'domain_violations': domain_errors, 'constraints': rows,
            'objective_value': str(model.objective.expression.evaluate(point)) if model.objective else None}


def witness_evaluations(written, program, records):
    facts, seen = [], set()
    for record in records:
        if record.verdict != Verdict.REFUTED or not record.witness:
            continue
        if record.metadata.get('region_direction') not in {'left_minus_right', 'right_minus_left'}:
            continue
        key = tuple(sorted(record.witness.items()))
        if key in seen:
            continue
        seen.add(key)
        facts.append({'assignment': {k: str(v) for k, v in record.witness.items()},
                      'written_model': evaluate_point(written, record.witness),
                      'program_model': evaluate_point(program, record.witness),
                      'source_satisfaction': 'requires_original_requirement_interpretation'})
    return facts
