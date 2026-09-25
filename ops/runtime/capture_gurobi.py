"""Capture the Gurobi model and result produced by one generated program.

The worker runs in a child process. This is process isolation, not a security
sandbox; callers must place untrusted generated code in an external sandbox.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


WORKER = Path(__file__).with_name("capture_worker.py")


def capture(code: str, *, timeout: float = 45, solve_it: bool = False) -> dict:
    if not isinstance(code, str) or not code.strip():
        return {"status": "NO_PROGRAM", "error": "program artifact unavailable"}
    payload = json.dumps({"code": code, "solve": bool(solve_it)})
    try:
        completed = subprocess.run(
            [sys.executable, str(WORKER)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": f"capture exceeded {timeout} seconds"}
    output = completed.stdout.strip().splitlines()
    if not output:
        detail = completed.stderr.strip()[-500:]
        return {"status": "WORKER_ERROR", "error": detail or f"exit={completed.returncode}"}
    try:
        result = json.loads(output[-1])
    except json.JSONDecodeError as error:
        return {"status": "WORKER_ERROR", "error": f"invalid worker JSON: {error}"}
    if completed.returncode and result.get("status") == "CAPTURED":
        result = dict(result, status="WORKER_ERROR", error=f"exit={completed.returncode}")
    return result


__all__ = ["capture"]
