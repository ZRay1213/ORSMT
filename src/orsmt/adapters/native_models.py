"""Native, pre-solve extraction; unsupported semantics fail closed.

The adapters do not solve, round solutions, infer missing domains, or convert
nonlinear models by dropping their nonlinear terms. Imports remain optional.
"""
from collections.abc import Mapping
from fractions import Fraction
import math

from ..ir import ModelIR, Variable, LinearExpr, LinearConstraint, Objective, as_fraction


def bound(value, lower):
    if value is None:
        return None
    if math.isinf(float(value)):
        if (float(value) < 0) == lower:
            return None
        raise ValueError('invalid infinite bound')
    return as_fraction(value)


def interval_rows(name, expression, lower, upper, provenance=None):
    lower, upper = bound(lower, True), bound(upper, False)
    if lower is not None and lower == upper:
        return [LinearConstraint(name, expression, '==', lower, provenance=provenance or {})]
    out = []
    if lower is not None:
        out.append(LinearConstraint(name+'::lb', expression, '>=', lower, provenance=provenance or {}))
    if upper is not None:
        out.append(LinearConstraint(name+'::ub', expression, '<=', upper, provenance=provenance or {}))
    return out


def pulp_model(model):
    import pulp
    if not isinstance(model, pulp.LpProblem):
        raise TypeError('pulp requires an LpProblem')
    if getattr(model, 'sos1', {}) or getattr(model, 'sos2', {}):
        raise ValueError('PuLP SOS constraints require an explicit reformulation')
    if model.objective is None:
        raise ValueError('PuLP objective must be explicit')
    variables = []
    native = model.variables()
    names = {id(v): v.name for v in native}
    for v in native:
        kind = 'binary' if v.isBinary() else 'integer' if v.isInteger() else 'continuous'
        if getattr(v, 'cat', None) not in {pulp.LpContinuous, pulp.LpInteger, pulp.LpBinary}:
            raise ValueError('unsupported PuLP variable domain')
        variables.append(Variable(v.name, kind, bound(v.lowBound, True), bound(v.upBound, False)))
    def expression(e):
        return LinearExpr(tuple((names[id(v)], as_fraction(c)) for v, c in e.items()), as_fraction(e.constant))
    rows = []
    constraints = model.constraints
    items = constraints.items() if hasattr(constraints, 'items') else enumerate(constraints)
    for key, row in items:
        name = str(row.name or key)
        # PuLP represents an affine expression compared against zero.
        rows.append(LinearConstraint(name, expression(row), {-1:'<=',0:'==',1:'>='}[row.sense], 0,
                                     provenance={'backend':'pulp','native_row':str(key)}))
    if model.sense not in {pulp.LpMinimize, pulp.LpMaximize}:
        raise ValueError('unsupported objective sense')
    return ModelIR(tuple(variables), tuple(rows),
                   Objective('min' if model.sense == pulp.LpMinimize else 'max', expression(model.objective)),
                   str(model.name or 'pulp'), {'adapter':'pulp','numeric_basis':'native_decimal_roundtrip'})


def pyomo_model(model):
    import pyomo.environ as pyo
    from pyomo.repn.standard_repn import generate_standard_repn
    from pyomo.core.base.block import BlockData
    if not isinstance(model, BlockData) or not model.is_constructed():
        raise TypeError('pyomo requires a constructed ConcreteModel/Block')
    allowed = {pyo.Var,pyo.Constraint,pyo.Objective,pyo.Param,pyo.Set,pyo.RangeSet,
               pyo.Block,pyo.Expression,pyo.Suffix,pyo.BuildAction,pyo.BuildCheck}
    unknown = set(model.collect_ctypes(active=True)) - allowed
    if unknown:
        raise ValueError('unsupported active Pyomo component: '+', '.join(sorted(t.__name__ for t in unknown)))
    native = list(model.component_data_objects(pyo.Var, active=True, descend_into=True))
    names = {id(v): v.name for v in native}
    variables = []
    for v in native:
        if v.is_binary():
            kind = 'binary'
        elif v.is_integer():
            kind = 'integer'
        elif v.is_continuous():
            kind = 'continuous'
        else:
            raise ValueError('unsupported Pyomo domain: '+v.name)
        # Do not replace a finite enumerated/holey domain with its interval hull.
        lo, hi, step = v.domain.get_interval()
        if step not in (0,1) or (kind == 'continuous' and step != 0):
            raise ValueError('non-interval Pyomo domain: '+v.name)
        lower, upper = v.bounds
        variables.append(Variable(v.name, kind, lower, upper))
    def expression(e):
        rep = generate_standard_repn(e, compute_values=True)
        if not rep.is_linear():
            raise ValueError('nonlinear Pyomo expression requires exact reformulation')
        if any(id(v) not in names for v in rep.linear_vars):
            raise ValueError('Pyomo expression references a variable outside the model')
        return LinearExpr(tuple((names[id(v)],as_fraction(c)) for v,c in zip(rep.linear_vars,rep.linear_coefs)),
                          as_fraction(rep.constant))
    rows = []
    for v in native:
        if v.fixed:
            if v.value is None:
                raise ValueError('fixed Pyomo variable has no value')
            rows.append(LinearConstraint('__fixed::'+v.name,LinearExpr.from_mapping({v.name:1}),'==',v.value,
                                         provenance={'backend':'pyomo','fixed_variable':v.name}))
    for c in model.component_data_objects(pyo.Constraint,active=True,descend_into=True):
        lo = pyo.value(c.lower) if c.has_lb() else None
        hi = pyo.value(c.upper) if c.has_ub() else None
        rows.extend(interval_rows(c.name,expression(c.body),lo,hi,{'backend':'pyomo','native_row':c.name}))
    objectives = list(model.component_data_objects(pyo.Objective,active=True,descend_into=True))
    if len(objectives)!=1:
        raise ValueError('exactly one active Pyomo objective is required')
    o=objectives[0]
    return ModelIR(tuple(variables),tuple(rows),Objective('min' if o.sense==pyo.minimize else 'max',expression(o.expr)),
                   str(model.name),{'adapter':'pyomo','numeric_basis':'native_decimal_roundtrip'})


def ortools_linear_model(model):
    from ortools.linear_solver import pywraplp, linear_solver_pb2
    if not isinstance(model, pywraplp.Solver):
        raise TypeError('ortools_linear requires a pywraplp.Solver')
    proto=linear_solver_pb2.MPModelProto()
    model.ExportModelToProto(proto)
    if proto.general_constraint or proto.HasField('quadratic_objective'):
        raise ValueError('OR-Tools general/quadratic constructs require exact reformulation')
    native=list(proto.variable)
    names=[v.name or f'__ortools_var_{i}' for i,v in enumerate(native)]
    variables=tuple(Variable(n,'integer' if v.is_integer else 'continuous',bound(v.lower_bound,True),bound(v.upper_bound,False))
                    for n,v in zip(names,native))
    rows=[]
    for i,c in enumerate(proto.constraint):
        if len(c.var_index)!=len(c.coefficient) or any(j<0 or j>=len(names) for j in c.var_index):
            raise ValueError('invalid OR-Tools linear row')
        e=LinearExpr(tuple((names[j],as_fraction(a)) for j,a in zip(c.var_index,c.coefficient)))
        rows.extend(interval_rows(c.name or f'row_{i}',e,c.lower_bound,c.upper_bound,
                                  {'backend':'ortools_linear','native_row_index':i}))
    objective=Objective('max' if proto.maximize else 'min',LinearExpr(tuple(
        (n,as_fraction(v.objective_coefficient)) for n,v in zip(names,native)),as_fraction(proto.objective_offset)))
    return ModelIR(variables,tuple(rows),objective,proto.name or 'ortools_linear',
                   {'adapter':'ortools_linear','numeric_basis':'native_decimal_roundtrip'})
