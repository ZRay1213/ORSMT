from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .ir import LinearConstraint, LinearExpr, ModelIR, Variable


def with_observable_projection(
    model: ModelIR,
    projection: Mapping[str, LinearExpr],
    *,
    name: str | None = None,
) -> ModelIR:
    """Expose affine semantic decisions through exact definitional equalities.

    A mapper may propose, for example, ``x = x_left + x_right``. This function
    does not trust that proposal: it lowers it into the model so the ordinary
    projected-region and preference queries can adjudicate it.
    """

    if not projection:
        raise ValueError("observable projection cannot be empty")
    known = set(model.variable_map)
    synthetic_names = {
        public: f"__orsmt_observable__{public}" for public in projection
    }
    if len(set(synthetic_names.values())) != len(synthetic_names):
        raise ValueError("observable projection names must be unique")
    collision = set(synthetic_names.values()) & known
    if collision:
        raise ValueError(
            f"observable projection collides with model variables: {sorted(collision)}"
        )

    definitions: list[LinearConstraint] = []
    synthetic_variables: list[Variable] = []
    for public, expression in sorted(projection.items()):
        if not public:
            raise ValueError("observable projection name cannot be empty")
        unknown = expression.support - known
        if unknown:
            raise ValueError(
                f"observable projection {public} uses unknown variables: {sorted(unknown)}"
            )
        synthetic = synthetic_names[public]
        synthetic_variables.append(
            Variable(
                synthetic,
                "continuous",
                observable=True,
                provenance={"semantic_name": public, "projection": True},
            )
        )
        lhs = LinearExpr.from_mapping(
            (
                (synthetic, 1),
                *((variable, -coefficient) for variable, coefficient in expression.terms),
            )
        )
        definitions.append(
            LinearConstraint(
                f"projection:{public}",
                lhs,
                "==",
                expression.constant,
                group="observable_projection",
                provenance={"semantic_name": public},
            )
        )

    hidden_variables = tuple(
        replace(variable, observable=False) for variable in model.variables
    )
    provenance = dict(model.provenance)
    provenance["observable_projection"] = {
        public: {
            "terms": [
                (variable, str(coefficient))
                for variable, coefficient in expression.terms
            ],
            "constant": str(expression.constant),
        }
        for public, expression in sorted(projection.items())
    }
    return ModelIR(
        hidden_variables + tuple(synthetic_variables),
        model.constraints + tuple(definitions),
        model.objective,
        name or f"{model.name}[projected]",
        provenance,
    )
