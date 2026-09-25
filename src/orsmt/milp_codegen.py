"""Deterministic fixed-instance MILP code generation, with explicit exactness checks."""
from fractions import Fraction
from math import gcd, lcm, isfinite
from .ir import ModelIR


def number(value):
    """A native numeric literal whose shortest decimal round-trip is unchanged."""
    value = Fraction(value)
    floating = float(value)
    if not isfinite(floating) or Fraction(str(floating)) != value:
        raise ValueError(f'native coefficient not exactly round-trippable: {value}')
    return str(value.numerator) if value.denominator == 1 else repr(floating)


def model_to_dict(model: ModelIR):
    def terms(expr):
        return [{'var': n, 'coef': str(c)} for n, c in expr.terms]
    return {
        'name': model.name,
        'variables': [{'name': v.name, 'type': v.kind,
                       'lb': None if v.lb is None else str(v.lb),
                       'ub': None if v.ub is None else str(v.ub),
                       'observable': v.observable} for v in model.variables],
        'constraints': [{'name': r.name, 'lhs': terms(r.lhs), 'constant': str(r.lhs.constant),
                         'op': r.sense, 'rhs': str(r.rhs)} for r in model.constraints],
        'objective': None if model.objective is None else {
            'sense': model.objective.sense, 'lhs': terms(model.objective.expression),
            'constant': str(model.objective.expression.constant)},
    }


def _native_row(terms, rhs):
    try:
        return [(n, number(c)) for n, c in terms], number(rhs)
    except ValueError:
        # Positive exact row scaling preserves the inequality and clears thirds, etc.
        values = [c for _, c in terms] + [rhs]
        multiplier = lcm(*(v.denominator for v in values))
        integers = [int(v * multiplier) for v in values]
        divisor = 0
        for v in integers:
            divisor = gcd(divisor, abs(v))
        divisor = divisor or 1
        integers = [Fraction(v, divisor) for v in integers]
        return [(n, number(v)) for (n, _), v in zip(terms, integers)], number(integers[-1])


def emit_gurobi(model: ModelIR) -> str:
    """Emit a complete instance builder, not an AST patch to arbitrary Python.

    No objective scaling is silently introduced. Coefficients that cannot be
    represented by the native API raise an explicit error before code execution.
    """
    if model.objective is None:
        raise ValueError('MILP repair requires an explicit objective')
    lines = ['import gurobipy as gp', 'from gurobipy import GRB', '',
             'def build_model():', '    m = gp.Model("orsmt_repaired")',
             '    m.Params.OutputFlag = 0', '    m.Params.Threads = 1', '    v = {}']
    rows = [(r.name, r.lhs.terms, r.sense, r.rhs-r.lhs.constant) for r in model.constraints]
    used = {r[0] for r in rows}
    def unique(name):
        while name in used:
            name += '_'
        used.add(name)
        return name
    for variable in model.variables:
        bounds = []
        for kind, value in [('lb', variable.lb), ('ub', variable.ub)]:
            default = ('0' if kind == 'lb' else '1') if variable.kind == 'binary' else (
                '-GRB.INFINITY' if kind == 'lb' else 'GRB.INFINITY')
            if value is None:
                bounds.append(default)
                continue
            try:
                literal = number(value)
            except ValueError:
                literal = default
                rows.append((unique('__bound_' + kind + '_' + variable.name),
                             ((variable.name, Fraction(1)),), '>=' if kind == 'lb' else '<=', value))
            bounds.append(literal)
        typ = {'continuous': 'GRB.CONTINUOUS', 'integer': 'GRB.INTEGER', 'binary': 'GRB.BINARY'}[variable.kind]
        lines.append(f'    v[{variable.name!r}] = m.addVar(lb={bounds[0]}, ub={bounds[1]}, vtype={typ}, name={variable.name!r})')
    lines.append('    m.update()')
    for name, terms, sense, rhs in rows:
        native, rhs_literal = _native_row(terms, rhs)
        coefficients = ', '.join(c for _, c in native)
        variables = ', '.join(f'v[{n!r}]' for n, _ in native)
        lines.append(f'    m.addConstr(gp.LinExpr([{coefficients}], [{variables}]) {sense} {rhs_literal}, name={name!r})')
    objective = model.objective.expression
    coefficients = ', '.join(number(c) for _, c in objective.terms)
    variables = ', '.join(f'v[{n!r}]' for n, _ in objective.terms)
    sense = 'GRB.MINIMIZE' if model.objective.sense == 'min' else 'GRB.MAXIMIZE'
    lines += [f'    m.setObjective(gp.LinExpr([{coefficients}], [{variables}]) + {number(objective.constant)}, {sense})',
              '    m.update()', '    return m, v', '', 'def solve():',
              '    m, v = build_model()', '    m.Params.TimeLimit = 20', '    m.optimize()',
              '    if m.SolCount == 0:', '        return None',
              '    return {"objective_value": m.ObjVal, "assignment": {n: x.X for n, x in v.items()}}', '',
              'if __name__ == "__main__":', '    import json', '    print(json.dumps(solve()))', '']
    source = '\n'.join(lines)
    compile(source, '<orsmt-milp-codegen>', 'exec')
    return source
