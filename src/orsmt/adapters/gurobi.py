from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from threading import local
from typing import Mapping

from ..ir import ModelIR


@dataclass(frozen=True)
class GurobiReplayAdapter:
    """Replay a partial SMT witness on the complete captured MILP with Gurobi.

    Gurobi is an optional host dependency. A license/import/solver failure is
    returned as ``None`` so the verifier emits ``UNDECIDED`` rather than a
    semantic refutation.
    """

    time_limit_s: float = 10.0
    _thread_state: local = field(default_factory=local, init=False, repr=False, compare=False)

    def _environment(self, gp):
        environment = getattr(self._thread_state, "environment", None)
        if environment is None:
            environment = gp.Env(empty=True)
            environment.setParam("OutputFlag", 0)
            environment.start()
            self._thread_state.environment = environment
        return environment

    def replay(
        self, model: ModelIR, point: Mapping[str, object]
    ) -> tuple[bool | None, str]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as error:  # optional integration
            return None, f"gurobi unavailable: {type(error).__name__}"

        native = None
        try:
            native = gp.Model("orsmt_replay", env=self._environment(gp))
            native.Params.OutputFlag = 0
            native.Params.Threads = 1
            native.Params.TimeLimit = self.time_limit_s
            variables = {}
            for variable in model.variables:
                vtype = {
                    "binary": GRB.BINARY,
                    "integer": GRB.INTEGER,
                    "continuous": GRB.CONTINUOUS,
                }[variable.kind]
                lb = -GRB.INFINITY if variable.lb is None else float(variable.lb)
                ub = GRB.INFINITY if variable.ub is None else float(variable.ub)
                variables[variable.name] = native.addVar(
                    lb=lb, ub=ub, vtype=vtype, name=variable.name
                )
            native.update()
            for constraint in model.constraints:
                expression = gp.LinExpr(float(constraint.lhs.constant))
                for name, coefficient in constraint.lhs.terms:
                    expression += float(coefficient) * variables[name]
                rhs = float(constraint.rhs)
                if constraint.sense == "<=":
                    native.addConstr(expression <= rhs, name=constraint.name)
                elif constraint.sense == ">=":
                    native.addConstr(expression >= rhs, name=constraint.name)
                else:
                    native.addConstr(expression == rhs, name=constraint.name)
            for name, value in point.items():
                if name not in variables:
                    return None, f"witness contains unknown variable {name}"
                exact = value if isinstance(value, Fraction) else Fraction(str(value))
                native.addConstr(variables[name] == float(exact), name=f"fix::{name}")
            native.setObjective(0.0, GRB.MINIMIZE)
            native.optimize()
            if native.Status == GRB.INFEASIBLE:
                return False, "native captured MILP is infeasible under the witness"
            if native.SolCount > 0:
                return True, "native captured MILP admits a completion of the witness"
            return None, f"native replay status={native.Status} without a solution"
        except Exception as error:
            return None, f"native replay failed: {type(error).__name__}: {str(error)[:160]}"
        finally:
            if native is not None:
                native.dispose()
