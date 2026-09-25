from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..ir import LinearConstraint, LinearExpr, ModelIR, Objective, Variable, as_fraction


class GurobiModelAdapter:
    """Lower a fully expanded, pre-optimization Gurobi model to exact ORSMT IR.

    The adapter accepts linear objectives and linear rows. Unsupported
    quadratic, general, or SOS constructs fail explicitly at the adapter
    boundary so callers can retain the item as ``UNDECIDED``.
    """

    def __init__(self, *, observable: Mapping[str, bool] | None = None) -> None:
        self.observable = dict(observable or {})

    @staticmethod
    def _bound(value: float) -> Any | None:
        return None if abs(float(value)) >= 1e90 else as_fraction(str(value))

    def to_ir(self, model: Any, *, name: str | None = None) -> ModelIR:
        model.update()
        unsupported = {
            "quadratic_constraints": int(getattr(model, "NumQConstrs", 0)),
            "general_constraints": int(getattr(model, "NumGenConstrs", 0)),
            "sos_constraints": int(getattr(model, "NumSOS", 0)),
        }
        active = [kind for kind, count in unsupported.items() if count]
        if active:
            raise ValueError("unsupported Gurobi constructs: " + ",".join(active))

        type_map = {"B": "binary", "I": "integer", "C": "continuous"}
        variables = []
        for raw in model.getVars():
            if raw.VType not in type_map:
                raise ValueError(f"unsupported Gurobi variable type: {raw.VType}")
            variables.append(
                Variable(
                    raw.VarName,
                    type_map[raw.VType],
                    self._bound(raw.LB),
                    self._bound(raw.UB),
                    self.observable.get(raw.VarName, True),
                    {"gurobi_name": raw.VarName},
                )
            )

        used_names: set[str] = set()
        constraints = []
        senses = {"<": "<=", ">": ">=", "=": "=="}
        for index, raw in enumerate(model.getConstrs()):
            base_name = raw.ConstrName or f"c{index}"
            row_name = base_name if base_name not in used_names else f"{base_name}#{index}"
            used_names.add(row_name)
            row = model.getRow(raw)
            terms = tuple(
                (row.getVar(i).VarName, as_fraction(str(row.getCoeff(i))))
                for i in range(row.size())
            )
            constraints.append(
                LinearConstraint(
                    row_name,
                    LinearExpr(terms, as_fraction(str(row.getConstant()))),
                    senses[raw.Sense],
                    as_fraction(str(raw.RHS)),
                    provenance={"gurobi_name": raw.ConstrName},
                )
            )

        raw_objective = model.getObjective()
        if raw_objective.__class__.__name__ == "QuadExpr":
            raise ValueError("quadratic objective is unsupported")
        objective = Objective(
            "min" if int(model.ModelSense) == 1 else "max",
            LinearExpr(
                tuple(
                    (
                        raw_objective.getVar(i).VarName,
                        as_fraction(str(raw_objective.getCoeff(i))),
                    )
                    for i in range(raw_objective.size())
                ),
                as_fraction(str(raw_objective.getConstant())),
            ),
            {"backend": "gurobi"},
        )
        model_name = name or str(getattr(model, "ModelName", "gurobi_model"))
        return ModelIR(
            tuple(variables),
            tuple(constraints),
            objective,
            model_name,
            {"backend": "gurobi", "fully_expanded": True},
        )
