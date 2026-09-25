#!/usr/bin/env python3
"""Audit written formulations against models captured from real solver programs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
from multiprocessing import Pool
from pathlib import Path

from orsmt import (
    AuditReport,
    EvidenceRecord,
    ORSMTPlugin,
    Relation,
    Verdict,
    Verifier,
    audit_scorecard,
    pipeline_profile,
    write_reports_atomic,
)
from orsmt.adapters import (
    CallableRuntimeAdapter,
    DictModelAdapter,
    DictResultAdapter,
    DictSourceAdapter,
    GurobiReplayAdapter,
)


CAPTURE = None
MODEL_ADAPTER = None
RESULT_ADAPTER = None
SOURCE_ADAPTER = None
PLUGIN = None
ARTIFACT_TIMEOUT_S = None


class ArtifactBudgetExceeded(BaseException):
    """Hard per-artifact budget; BaseException prevents adapter error swallowing."""


def load_capture(path: Path, function_name: str):
    spec = importlib.util.spec_from_file_location("orsmt_host_capture", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load capture module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"capture function is not callable: {function_name}")
    return function


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def initialize_worker(
    capture_module: str,
    capture_function: str,
    timeout_ms: int,
    native_gurobi: bool,
    artifact_timeout_s: float,
) -> None:
    global CAPTURE, MODEL_ADAPTER, RESULT_ADAPTER, SOURCE_ADAPTER, PLUGIN
    global ARTIFACT_TIMEOUT_S
    CAPTURE = load_capture(Path(capture_module), capture_function)
    MODEL_ADAPTER = DictModelAdapter()
    RESULT_ADAPTER = DictResultAdapter()
    SOURCE_ADAPTER = DictSourceAdapter()
    native_replay = GurobiReplayAdapter() if native_gurobi else None
    PLUGIN = ORSMTPlugin(Verifier(timeout_ms=timeout_ms, native_replay=native_replay))
    ARTIFACT_TIMEOUT_S = artifact_timeout_s


def _audit_row_impl(row: dict):
    assert CAPTURE is not None and MODEL_ADAPTER is not None
    assert RESULT_ADAPTER is not None and SOURCE_ADAPTER is not None and PLUGIN is not None
    code = row.get("code")
    captured = (
        CAPTURE(code, solve_it=True)
        if code
        else {"status": "NO_PROGRAM", "error": "program artifact unavailable"}
    )
    runtime_adapter = CallableRuntimeAdapter(lambda _code: captured, MODEL_ADAPTER)
    returned = captured.get("returned") if isinstance(captured, dict) else None
    report = PLUGIN.audit_artifact(
        source_specification=row.get("source_obligations"),
        source_reference=row.get("source_model", row.get("reference_model")),
        written_artifact=row.get("written_model", row.get("formulation")),
        executable_artifact=code,
        returned_artifact=returned,
        source_adapter=SOURCE_ADAPTER,
        model_adapter=MODEL_ADAPTER,
        runtime_adapter=runtime_adapter,
        result_adapter=RESULT_ADAPTER,
        metadata={
            "id": str(row["id"]),
            "source": row.get("source") or str(row["id"]).split(":", 1)[0],
            "model_key": row.get("model_key"),
            "parse_note": row.get("parse_note"),
            "capture_status": captured.get("status") if isinstance(captured, dict) else None,
        },
    )
    return str(row["id"]), report


def _timeout_report(row: dict, timeout_s: float) -> tuple[str, AuditReport]:
    row_id = str(row["id"])
    records = tuple(
        EvidenceRecord(
            relation=relation,
            claim_id=f"{relation.value}:artifact_budget",
            verdict=Verdict.UNDECIDED,
            query="artifact audit completes within its wall-clock budget",
            reason="artifact_timeout",
            duration_ms=timeout_s * 1000.0,
        )
        for relation in (
            Relation.FORMALIZATION,
            Relation.IMPLEMENTATION,
            Relation.REPORTING,
        )
    )
    report = AuditReport(
        records,
        pipeline_profile(records),
        {
            "id": row_id,
            "source": row.get("source") or row_id.split(":", 1)[0],
            "model_key": row.get("model_key"),
            "artifact_timeout_s": timeout_s,
        },
    )
    return row_id, report


def audit_row(row: dict):
    timeout_s = float(ARTIFACT_TIMEOUT_S or 0)
    if timeout_s <= 0 or not hasattr(signal, "setitimer"):
        return _audit_row_impl(row)

    def expire(_signum, _frame):
        raise ArtifactBudgetExceeded()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return _audit_row_impl(row)
    except ArtifactBudgetExceeded:
        return _timeout_report(row, timeout_s)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-module", type=Path, required=True)
    parser.add_argument("--capture-function", default="capture")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    parser.add_argument(
        "--artifact-timeout-s",
        type=float,
        default=120.0,
        help="hard wall-clock budget per artifact; timeout yields explicit UNDECIDED records",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--native-gurobi", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate input ids")

    with Pool(
        processes=args.workers,
        initializer=initialize_worker,
        initargs=(
            str(args.capture_module),
            args.capture_function,
            args.timeout_ms,
            args.native_gurobi,
            args.artifact_timeout_s,
        ),
    ) as pool:
        reports = list(pool.imap(audit_row, rows, chunksize=1))
    write_reports_atomic(args.output, reports)
    scorecard = audit_scorecard([report.records for _, report in reports])
    summary = {
        "rows": len(reports),
        "id_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
        "scorecard": scorecard.to_dict(),
    }
    summary_path = args.output.with_suffix(".summary.json")
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, summary_path)


if __name__ == "__main__":
    main()
