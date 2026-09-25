from __future__ import annotations

import argparse
import json

from orsmt import (
    LinearConstraint,
    LinearExpr,
    ModelIR,
    Objective,
    ORSMTPlugin,
    ResultRecord,
    SourceObligation,
    Variable,
    Verifier,
)
from orsmt.adapters import GurobiReplayAdapter


def model(name: str, share_coefficients: dict[str, int]) -> ModelIR:
    return ModelIR(
        variables=(
            Variable("brownies", "integer", lb=0),
            Variable("other", "integer", lb=0),
        ),
        constraints=(
            LinearConstraint(
                "all_items", LinearExpr.from_mapping({"brownies": 1, "other": 1}), "==", 8
            ),
            LinearConstraint(
                "brownie_share", LinearExpr.from_mapping(share_coefficients), ">=", 0
            ),
        ),
        objective=Objective(
            "max", LinearExpr.from_mapping({"brownies": 2, "other": 4})
        ),
        name=name,
    )


correct = model("source", {"brownies": 3, "other": -2})
written = model("written", {"brownies": 5, "other": -2})
runtime = model("runtime", {"brownies": 5, "other": -2})
obligation = SourceObligation(
    "share_at_least_40_percent",
    LinearConstraint(
        "brownie_share", LinearExpr.from_mapping({"brownies": 3, "other": -2}), ">=", 0
    ),
    source_span="at least 40% of all items are brownies",
)
result = ResultRecord(
    {"brownies": 3, "other": 5}, value=26, status="optimal"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--native-gurobi",
    action="store_true",
    help="require each SMT witness to reproduce on a Gurobi reconstruction",
)
args = parser.parse_args()
verifier = Verifier(
    native_replay=GurobiReplayAdapter() if args.native_gurobi else None
)

report = ORSMTPlugin(verifier).audit(
    source_obligations=(obligation,),
    source_model=correct,
    written_model=written,
    runtime_model=runtime,
    result=result,
    metadata={"example": "brownie-share"},
)

print(json.dumps(report.to_dict(), indent=2))
