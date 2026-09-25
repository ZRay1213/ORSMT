#!/usr/bin/env python3
"""Build the runtime model of one candidate program and print it as typed IR.

Runs as its own process because the program is model-written: it can loop forever, exhaust memory, or
raise anywhere, and none of that should take down the batch.  gurobipy's optimize is intercepted so the
model is recorded exactly as the program built it, before any presolve or reformulation touches it.

With "solve" the interception records the model and then lets the call through, so the program runs to
completion and prints the decision and objective value it would have returned.  That is the only way to
obtain the third artifact -- the returned result -- from the same execution that produced the model, and
without it the program-to-result relation cannot be checked on natural outputs at all.  Solving is what
the size-limited licence bounds, so a model too large to solve still yields a captured model and simply
reports no result.  Reads {"code": ..., "solve": bool} on stdin, prints one JSON object on stdout.
"""
from __future__ import annotations
import io, json, os, resource, sys, contextlib


def dec(x):
    from decimal import Decimal
    return str(Decimal(repr(float(x))))


def gurobi_to_ir(model):
    """Lower a captured gurobipy model into the dict IR, refusing anything not linear."""
    import gurobipy as gp
    model.update()
    variables, names = [], {}
    for v in model.getVars():
        vt = v.VType
        typ = "bool" if vt == gp.GRB.BINARY else "int" if vt in (gp.GRB.INTEGER,) else "real"
        lb = None if v.LB <= -gp.GRB.INFINITY else float(v.LB)
        ub = None if v.UB >= gp.GRB.INFINITY else float(v.UB)
        variables.append({"name": v.VarName, "type": typ, "lb": lb, "ub": ub})
        names[v.index] = v.VarName
    if model.NumQConstrs or model.NumGenConstrs or model.NumSOS:
        return None, (f"nonlinear_rows:q={model.NumQConstrs},"
                      f"gen={model.NumGenConstrs},sos={model.NumSOS}")
    constraints = []
    for i, c in enumerate(model.getConstrs()):
        row = model.getRow(c)
        lhs = [{"coef": dec(row.getCoeff(k)), "var": row.getVar(k).VarName} for k in range(row.size())]
        op = {"<": "<=", ">": ">=", "=": "=="}.get(c.Sense, c.Sense)
        constraints.append({"kind": "linear", "name": c.ConstrName or f"c{i}", "lhs": lhs,
                            "op": op, "rhs": dec(c.RHS)})
    obj = model.getObjective()
    n_quad = 0
    if hasattr(obj, "getLinExpr"):
        n_quad = int(obj.size()); lin = obj.getLinExpr()
    else:
        lin = obj
    terms = [{"coef": dec(lin.getCoeff(k)), "var": lin.getVar(k).VarName} for k in range(lin.size())]
    sense = "min" if model.ModelSense == gp.GRB.MINIMIZE else "max"
    ir = {"variables": variables, "constraints": constraints,
          "objective": {"sense": sense, "lhs": terms, "const": dec(lin.getConstant())},
          "n_quadratic": n_quad}
    return ir, None



def prose_result(printed, var_names):
    """Fallback for programs that report in prose rather than in the contracted JSON line."""
    import os, re
    if os.environ.get("ORSMT_PROSE_RESULT") != "1":
        return None
    text = "\n".join(printed)
    # a released harness line, when present, is exactly what that system scored
    hl = re.findall(r"(?im)^\s*Just print the best solution:\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*$", text)
    if hl:
        obj = float(hl[-1]); tag = "program_output"
    else:
        nums = re.findall(r"(?i)(?:objective|optimal|minimum|maximum|minimi[sz]ed|maximi[sz]ed)[^\n]*?[:=]\s*\$?\s*(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", text)
        if not nums:
            return None
        obj = float(nums[-1]); tag = "program_output_prose"
    # printed decisions are formatted and rounded by the program; the decision the program returned is the
    # solver's, which the caller fills from attributes when this dict carries no assignment
    return {"objective_value": obj, "assignment": {}, "from": tag}

def main():
    payload = json.loads(sys.stdin.read())
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    os.environ["GRB_LICENSE_FILE"] = os.environ.get("GRB_LICENSE_FILE", "")
    out = {"status": "UNKNOWN"}
    try:
        import gurobipy as gp
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "NO_GUROBI", "error": str(e)[:200]})); return

    captured = []
    orig_opt = gp.Model.optimize
    orig_env = gp.Env
    want_solve = bool(payload.get("solve"))
    solve_error = []

    def intercept(self, *a, **k):
        self.update()
        captured.append(self)
        if not want_solve:
            raise RuntimeError("__captured__")
        try:
            self.setParam("OutputFlag", 0)
            self.setParam("TimeLimit", float(payload.get("time_limit", 20)))
        except Exception:  # noqa: BLE001
            pass
        try:
            return orig_opt(self, *a, **k)
        except Exception as e:  # noqa: BLE001
            solve_error.append(f"{type(e).__name__}: {str(e)[:160]}")
            raise RuntimeError("__captured__")

    gp.Model.optimize = intercept
    buf = io.StringIO()
    try:
        g = {"__name__": "__main__", "__builtins__": __builtins__}
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(compile(payload["code"], "<candidate>", "exec"), g)  # noqa: S102
        out["status"] = "NO_OPTIMIZE_CALL"
    except RuntimeError as e:
        if "__captured__" not in str(e):
            out["status"] = "EXEC_ERROR"; out["error"] = f"RuntimeError: {str(e)[:200]}"
    except SystemExit:
        out["status"] = "NO_OPTIMIZE_CALL"
    except BaseException as e:  # noqa: BLE001
        out["status"] = "EXEC_ERROR"; out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        gp.Model.optimize = orig_opt
        gp.Env = orig_env

    if want_solve:
        out["solve_error"] = solve_error[0] if solve_error else None
        printed = buf.getvalue().strip().splitlines()
        for line in reversed(printed[-12:]):
            line = line.strip()
            if line.startswith("{") and "objective_value" in line:
                try:
                    r = json.loads(line)
                    out["returned"] = {"objective_value": r.get("objective_value"),
                                       "assignment": r.get("assignment")}
                except json.JSONDecodeError:
                    pass
                break
        if "returned" not in out and captured:
            m = captured[0]
            try:
                names = [v.VarName for v in m.getVars()]
            except Exception:  # noqa: BLE001
                names = []
            pr = prose_result(printed, names)
            if pr is not None:
                try:
                    if not pr["assignment"] and m.SolCount > 0:
                        pr["assignment"] = {v.VarName: float(v.X) for v in m.getVars()}; pr["from"] = "program_output_prose_attr_assignment"
                except Exception:  # noqa: BLE001
                    pass
                out["returned"] = pr
            try:
                if "returned" not in out and m.SolCount > 0:
                    out["returned"] = {"objective_value": float(m.ObjVal),
                                       "assignment": {v.VarName: float(v.X) for v in m.getVars()},
                                       "from": "model_attributes"}
                out["gurobi_status"] = int(m.Status)
            except Exception:  # noqa: BLE001
                pass
    if not captured:
        # the program may have built a model and never optimized it
        out.setdefault("status", "NO_MODEL")
        print(json.dumps(out)); return
    try:
        ir, err = gurobi_to_ir(captured[0])
        if ir is None:
            out["status"] = "UNSUPPORTED"; out["error"] = err
        else:
            out["status"] = "CAPTURED"; out["ir"] = ir
            out["n_models_built"] = len(captured)
    except Exception as e:  # noqa: BLE001
        out["status"] = "LOWER_ERROR"; out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
