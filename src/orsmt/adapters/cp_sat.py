"""Exact finite-domain CP-SAT lowering into MILP with explicit auxiliaries.

Supports linear domain unions, Boolean clauses/cardinality/XOR, enforcement,
all-different and max/abs. Scheduling, table, circuit and nonlinear products
fail closed. Every big-M is derived from the complete finite variable bounds.
"""
from fractions import Fraction
from ..ir import ModelIR, Variable, LinearExpr, LinearConstraint, Objective, as_fraction


def has(proto, field):
    helper=getattr(proto,'has_'+field,None)
    return helper() if helper else proto.HasField(field)


def add(a,b,scale=1):
    return LinearExpr(a.terms+tuple((n,c*scale) for n,c in b.terms),a.constant+b.constant*scale)


class CPSATLowering:
    def __init__(self, model, max_rows=100000):
        from ortools.sat.python import cp_model
        if not isinstance(model,cp_model.CpModel):
            raise TypeError('ortools_cp_sat requires a CpModel')
        error=model.validate()
        if error:
            raise ValueError('invalid CP-SAT model: '+error)
        self.p=model.Proto()
        self.variables=[];self.rows=[];self.domains={};self.max_rows=max_rows
        self.names=[v.name or f'__cp_var_{i}' for i,v in enumerate(self.p.variables)]
        if len(set(self.names))!=len(self.names):
            raise ValueError('CP-SAT variable names must be unique for semantic correspondence')
        for n,v in zip(self.names,self.p.variables):
            d=list(v.domain)
            if not d or len(d)%2:
                raise ValueError('invalid finite CP-SAT domain')
            self.variables.append(Variable(n,'binary' if d[0]>=0 and d[-1]<=1 else 'integer',d[0],d[-1]))
            self.domains[n]=(Fraction(d[0]),Fraction(d[-1]))
        for i,v in enumerate(self.p.variables):
            if len(v.domain)>2:
                self.domain('domain::'+str(i),self.ref(i),list(v.domain),[])

    def ref(self,i,literal=False):
        j=i if i>=0 else -i-1
        if j>=len(self.names):
            raise ValueError('invalid variable reference')
        return LinearExpr.from_mapping({self.names[j]:1 if i>=0 else -1},1 if literal and i<0 else 0)

    def expr(self,p):
        e=LinearExpr(constant=int(p.offset))
        if len(p.vars)!=len(p.coeffs):
            raise ValueError('mismatched CP-SAT expression')
        for i,c in zip(p.vars,p.coeffs):
            e=add(e,self.ref(i),int(c))
        return e

    def limits(self,e):
        lo=hi=e.constant
        for n,c in e.terms:
            a,b=self.domains[n]
            lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
        return lo,hi

    def auxiliary(self):
        n='__orsmt_cp_aux_'+str(len(self.variables))
        while n in self.domains:
            n+='_'
        self.variables.append(Variable(n,'binary',0,1,observable=False,
                                       provenance={'purpose':'exact_cp_sat_lowering'}))
        self.domains[n]=(Fraction(0),Fraction(1))
        return LinearExpr.from_mapping({n:1})

    def row(self,name,e,sense,rhs,guards):
        if len(self.rows)>=self.max_rows:
            raise ValueError('CP-SAT lowering row budget exceeded')
        if sense=='==':
            self.row(name+'::lo',e,'>=',rhs,guards);self.row(name+'::hi',e,'<=',rhs,guards);return
        lo,hi=self.limits(e)
        M=max(Fraction(0),hi-rhs if sense=='<=' else rhs-lo)
        slack=LinearExpr(constant=len(guards))
        for g in guards:
            slack=add(slack,g,-1)
        lowered=add(e,slack,-M if sense=='<=' else M)
        self.rows.append(LinearConstraint(name,lowered,sense,rhs,
            provenance={'adapter':'ortools_cp_sat','lowering':'bounded_indicator' if guards else 'linear',
                        'big_m':str(M) if guards else None,'source_constraint':name.split('::')[0]}))

    def domain(self,name,e,d,guards):
        if not d:
            self.row(name+'::false',LinearExpr(),'>=',1,guards);return
        if len(d)==2:
            self.row(name+'::lo',e,'>=',int(d[0]),guards)
            self.row(name+'::hi',e,'<=',int(d[1]),guards);return
        selectors=[self.auxiliary() for _ in range(len(d)//2)]
        total=LinearExpr()
        for s in selectors:total=add(total,s)
        self.row(name+'::choose',total,'==',1,[])
        for j,s in enumerate(selectors):
            self.domain(name+'::piece'+str(j),e,d[2*j:2*j+2],guards+[s])

    def finish(self):
        for i,c in enumerate(self.p.constraints):
            name=c.name or f'constraint_{i}'
            guards=[self.ref(int(g),True) for g in c.enforcement_literal]
            if has(c,'linear'):
                e=LinearExpr()
                for v,a in zip(c.linear.vars,c.linear.coeffs):e=add(e,self.ref(int(v)),int(a))
                self.domain(name,e,list(c.linear.domain),guards)
            elif any(has(c,k) for k in ('bool_or','bool_and','at_most_one','exactly_one','bool_xor')):
                kind=next(k for k in ('bool_or','bool_and','at_most_one','exactly_one','bool_xor') if has(c,k))
                literals=[self.ref(int(v),True) for v in getattr(c,kind).literals]
                total=LinearExpr()
                for lit in literals:total=add(total,lit)
                if kind=='bool_xor':
                    odd=[j for j in range(1,len(literals)+1,2)]
                    self.domain(name,total,[v for j in odd for v in (j,j)],guards)
                else:
                    sense,rhs={'bool_or':('>=',1),'bool_and':('==',len(literals)),
                               'at_most_one':('<=',1),'exactly_one':('==',1)}[kind]
                    self.row(name,total,sense,rhs,guards)
            elif has(c,'all_diff'):
                exprs=[self.expr(e) for e in c.all_diff.exprs]
                for j,a in enumerate(exprs):
                    for k,b in enumerate(exprs[:j]):
                        e=add(a,b,-1);lo,hi=self.limits(e)
                        d=[]
                        if lo<=-1:d.extend([int(lo),-1])
                        if hi>=1:d.extend([1,int(hi)])
                        self.domain(name+f'::{j}_{k}',e,d,guards)
            elif has(c,'lin_max'):
                target=self.expr(c.lin_max.target)
                exprs=[self.expr(e) for e in c.lin_max.exprs]
                if not exprs:raise ValueError('empty maximum')
                selectors=[self.auxiliary() for _ in exprs]
                total=LinearExpr()
                for s in selectors:total=add(total,s)
                self.row(name+'::choose',total,'==',1,[])
                for j,(e,s) in enumerate(zip(exprs,selectors)):
                    diff=add(target,e,-1)
                    self.row(name+f'::max{j}',diff,'>=',0,guards)
                    self.row(name+f'::attain{j}',diff,'<=',0,guards+[s])
            else:
                kind=[k[4:] for k in dir(c) if k.startswith('has_') and getattr(c,k)()]
                if not kind and hasattr(c,'WhichOneof'):kind=[c.WhichOneof('constraint')]
                raise ValueError('unsupported CP-SAT constraint: '+str(kind))
        for i,lit in enumerate(self.p.assumptions):
            self.row(f'assumption_{i}',self.ref(int(lit),True),'==',1,[])
        if has(self.p,'floating_point_objective'):
            o=self.p.floating_point_objective
            e=LinearExpr(constant=as_fraction(o.offset))
            for v,a in zip(o.vars,o.coeffs):e=add(e,self.ref(int(v)),as_fraction(a))
            objective=Objective('max' if o.maximize else 'min',e)
        elif has(self.p,'objective'):
            o=self.p.objective;factor=as_fraction(o.scaling_factor) or Fraction(1)
            raw=LinearExpr()
            for v,a in zip(o.vars,o.coeffs):raw=add(raw,self.ref(int(v)),int(a))
            if o.domain:self.domain('objective_domain',raw,list(o.domain),[])
            objective=Objective('max' if factor<0 else 'min',
                LinearExpr(tuple((n,a*factor) for n,a in raw.terms),(raw.constant+as_fraction(o.offset))*factor))
        else:
            raise ValueError('CP-SAT objective must be explicit')
        return ModelIR(tuple(self.variables),tuple(self.rows),objective,self.p.name or 'cp_sat',
            {'adapter':'ortools_cp_sat','numeric_basis':'integer_domains_exact_rational_objective',
             'auxiliary_projection_required':any(not v.observable for v in self.variables),
             'native_constraints':len(self.p.constraints),'lowered_constraints':len(self.rows),
             'variable_indices':dict(enumerate(self.names))})


def cp_sat_model(model):
    return CPSATLowering(model).finish()
