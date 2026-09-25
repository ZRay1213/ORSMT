from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import DictModelAdapter, DictResultAdapter, DictSourceAdapter
from .ledger import read_jsonl, validate_exact_ids, write_reports_atomic
from .plugin import ORSMTPlugin
from .verify import Verifier


def audit_rows(rows, timeout_ms: int = 10_000):
    model_adapter = DictModelAdapter()
    result_adapter = DictResultAdapter()
    source_adapter = DictSourceAdapter()
    plugin = ORSMTPlugin(Verifier(timeout_ms=timeout_ms))
    reports = []
    for row in rows:
        row_id = str(row["id"])
        source_name = row.get("source") or row_id.split(":", 1)[0]
        written_raw = row.get("written_model", row.get("formulation"))
        runtime_raw = row.get("runtime_model", row.get("captured_model"))
        source_raw = row.get("source_model", row.get("reference_model"))
        obligation_raw = row.get("source_obligations")
        adapter_errors = {}

        def adapt(name, adapter, raw):
            if not raw:
                return None
            try:
                return adapter(raw)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                adapter_errors[name] = f"{type(error).__name__}:{error}"
                return None

        source_model = adapt("source_model", model_adapter.to_ir, source_raw)
        written_model = adapt("written_model", model_adapter.to_ir, written_raw)
        runtime_model = adapt("runtime_model", model_adapter.to_ir, runtime_raw)
        result = adapt("result", result_adapter.to_result, row.get("result"))
        obligations = adapt("source_obligations", source_adapter.obligations, obligation_raw)
        report = plugin.audit(
            source_obligations=obligations or (),
            source_model=source_model,
            written_model=written_model,
            runtime_model=runtime_model,
            result=result,
            metadata={
                "id": row_id,
                "source": source_name,
                "model_key": row.get("model_key"),
                "parse_note": row.get("parse_note"),
                "truncated": row.get("truncated"),
                "adapter_errors": adapter_errors,
            },
        )
        reports.append((row_id, report))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch relational ORSMT audit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-ids")
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate input ids")
    if args.expected_ids:
        expected = [line.strip() for line in Path(args.expected_ids).read_text().splitlines() if line.strip()]
        validate_exact_ids(expected, ids)
    reports = audit_rows(rows, args.timeout_ms)
    write_reports_atomic(args.output, reports)


if __name__ == "__main__":
    main()
