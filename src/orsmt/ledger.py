from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .plugin import AuditReport


def validate_exact_ids(expected_ids: Iterable[str], actual_ids: Iterable[str]) -> None:
    expected = list(expected_ids)
    actual = list(actual_ids)
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate artifact ids")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected ids")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:10]
        extra = sorted(set(actual) - set(expected))[:10]
        raise ValueError(f"artifact id mismatch; missing={missing}, extra={extra}")


def write_reports_atomic(
    path: str | os.PathLike[str], reports: Iterable[tuple[str, AuditReport]]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(reports)
    ids = [str(row_id) for row_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate report ids")
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row_id, report in rows:
                payload: dict[str, Any] = {"id": str(row_id), **report.to_dict()}
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: str | os.PathLike[str]) -> list[Mapping[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
