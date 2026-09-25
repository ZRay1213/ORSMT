import itertools
import importlib.util
import unittest
from dataclasses import replace
from fractions import Fraction
import z3
from orsmt.translator import ModelTranslator
from orsmt.ir import ModelIR, Variable, LinearExpr, LinearConstraint, Objective
from orsmt.repair_workflow import compare_models
from orsmt.verify import Verifier
from orsmt.evidence import aggregate
from orsmt.z3bridge import make_variables, model_formula

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.t=ModelTranslator()
        self.ref=ModelIR((Variable('x','integer',0,4),Variable('y','continuous',None,None)),
            (LinearConstraint('row',LinearExpr.from_mapping({'x':2,'y':1}),'<=',7),),
            Objective('max',LinearExpr.from_mapping({'x':3,'y':-1},5)))
    def equivalent(self,m,ref=None):
        rows=compare_models(Verifier(),ref or self.ref,self.t.translate(m).model)
        self.assertEqual(aggregate(list(rows)).value,'CERTIFIED', [r.to_dict() for r in rows])
    def test_pulp(self):
        import pulp as p
        m=p.LpProblem('m',p.LpMaximize);x=p.LpVariable('x',0,4,p.LpInteger);y=p.LpVariable('y')
        m+=2*x+y+1<=8;m+=3*x-y+5
        self.equivalent(m)
        m.sos1['unsupported']={x:1,y:2}
        with self.assertRaisesRegex(ValueError,'SOS'):self.t.translate(m)
    @unittest.skipUnless(importlib.util.find_spec('pyomo') is not None, 'pyomo optional dependency unavailable')
    def test_pyomo(self):
        import pyomo.environ as p
        m=p.ConcreteModel();m.x=p.Var(domain=p.Integers,bounds=(0,4));m.y=p.Var(domain=p.Reals)
        m.c=p.Constraint(expr=2*m.x+m.y+1<=8);m.o=p.Objective(expr=3*m.x-m.y+5,sense=p.maximize)
        self.equivalent(m)
        m.x.fix(2)
        fixed=replace(self.ref,constraints=self.ref.constraints+(LinearConstraint('fixed',LinearExpr.from_mapping({'x':1}),'==',2),))
        self.equivalent(m,fixed)
        m.x.unfix();m.o.set_value(m.x*m.y)
        with self.assertRaisesRegex(ValueError,'nonlinear'):self.t.translate(m)
    @unittest.skipUnless(importlib.util.find_spec('pyomo') is not None, 'pyomo optional dependency unavailable')
    def test_pyomo_holey_and_logic_reject(self):
        import pyomo.environ as p
        m=p.ConcreteModel();m.x=p.Var(domain=p.Set(initialize=[0,2,5]));m.o=p.Objective(expr=m.x)
        with self.assertRaises(ValueError):self.t.translate(m)
        m=p.ConcreteModel();m.x=p.Var();m.o=p.Objective(expr=m.x);m.b=p.BooleanVar();m.l=p.LogicalConstraint(expr=m.b)
        with self.assertRaisesRegex(ValueError,'unsupported'):self.t.translate(m)
    def test_ortools_linear(self):
        from ortools.linear_solver import pywraplp
        m=pywraplp.Solver.CreateSolver('SCIP');x=m.IntVar(0,4,'x');y=m.NumVar(-m.infinity(),m.infinity(),'y')
        m.Add(2*x+y+1<=8);m.Maximize(3*x-y+5)
        self.equivalent(m)
    def cp_exhaustive(self,m,variables,values):
        from ortools.sat.python import cp_model
        ir=self.t.translate(m).model;syms=make_variables(ir);s=z3.Solver();s.add(model_formula(ir,syms))
        for point in itertools.product(*values):
            clone=m.clone()
            for var,value in zip(variables,point):clone.add(clone.get_int_var_from_proto_index(var.index)==value)
            solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
            status=solver.solve(clone)
            self.assertIn(status,[cp_model.OPTIMAL,cp_model.INFEASIBLE])
            s.push();s.add(*[syms[v.name]==a for v,a in zip(variables,point)])
            self.assertEqual(s.check()==z3.sat,status==cp_model.OPTIMAL,point);s.pop()
            if status==cp_model.OPTIMAL:
                actual=ir.objective.expression.constant
                for n,c in ir.objective.expression.terms:actual+=c*dict(zip([v.name for v in variables],point))[n]
                self.assertAlmostEqual(float(actual),solver.objective_value)
        return ir
    def test_cp_domains_enforcement_and_objective(self):
        from ortools.sat.python import cp_model as p
        m=p.CpModel();x=m.new_int_var_from_domain(p.Domain.from_values([0,2,3]),'x');b=m.new_bool_var('b')
        m.add(x>=2).only_enforce_if(b.Not());m.add(x<=2).only_enforce_if(b);m.maximize(2*x+3)
        ir=self.cp_exhaustive(m,[x,b],[range(4),range(2)])
        self.assertTrue(ir.auxiliary_names);self.assertEqual(ir.objective.sense,'max')
    def test_cp_boolean_and_assumptions(self):
        from ortools.sat.python import cp_model as p
        for kind in ('add_bool_or','add_bool_and','add_at_most_one','add_exactly_one','add_bool_xor'):
            with self.subTest(kind=kind):
                m=p.CpModel();a,b,c=[m.new_bool_var(n) for n in 'abc'];getattr(m,kind)([a,b.Not(),c]);m.add_assumption(a.Not());m.minimize(a+2*b+3*c)
                self.cp_exhaustive(m,[a,b,c],[range(2)]*3)
    def test_cp_alldiff_max_abs(self):
        from ortools.sat.python import cp_model as p
        m=p.CpModel();x=m.new_int_var(-1,1,'x');y=m.new_int_var(-1,1,'y');z=m.new_int_var(0,2,'z');a=m.new_int_var(0,1,'a')
        m.add_all_different([x,y]);m.add_max_equality(z,[x+1,y]);m.add_abs_equality(a,x);m.minimize(z+0.5*a+2.25)
        self.cp_exhaustive(m,[x,y,z,a],[range(-1,2),range(-1,2),range(3),range(2)])
    def test_cp_unsupported(self):
        from ortools.sat.python import cp_model as p
        m=p.CpModel();x=m.new_int_var(0,2,'x');y=m.new_int_var(0,4,'y');m.add_multiplication_equality(y,[x,x]);m.minimize(y)
        with self.assertRaisesRegex(ValueError,'unsupported'):self.t.translate(m)
    def test_exact_json_decimal(self):
        m=self.t.translate('{"variables":[{"name":"x","lb":0}],"constraints":[{"lhs":{"x":1},"op":"<=","rhs":0.123456789123456789}],"objective":{"sense":"min","lhs":{"x":1}}}').model
        self.assertEqual(m.constraints[0].rhs,Fraction('0.123456789123456789'))
        with self.assertRaisesRegex(ValueError,'duplicate'):self.t.translate('{"variables":[],"variables":[],"constraints":[],"objective":{}}')

if __name__=='__main__':unittest.main()
