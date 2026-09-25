#!/usr/bin/env python3
"""Verify the frozen six-model ORSMT release without network access."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/paper_core/manifest.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")

    scope_rows = rows(ROOT / "data/milp_benchmark_v3/eligible.jsonl")
    scope_ids = [str(row["id"]) for row in scope_rows]
    expected_problem_count = int(manifest["problem_count"])
    if len(scope_ids) != expected_problem_count or len(set(scope_ids)) != expected_problem_count:
        raise ValueError("scope IDs are incomplete or duplicated")
    expected_ids = set(scope_ids)

    totals = Counter()
    by_model: dict[str, dict[str, int]] = {}
    for model in manifest["models"]:
        candidate_rows = rows(ROOT / f"data/paper_core/candidates/dual_{model}.jsonl")
        verification_rows = rows(ROOT / f"data/paper_core/verification/mc_{model}.jsonl")
        for label, current in (("candidate", candidate_rows), ("verification", verification_rows)):
            ids = [str(row["id"]) for row in current]
            if len(ids) != expected_problem_count or len(set(ids)) != expected_problem_count:
                raise ValueError(f"{model} {label} IDs are incomplete or duplicated")
            if set(ids) != expected_ids:
                raise ValueError(f"{model} {label} IDs differ from the frozen scope")

        states = Counter(str(row.get("state", "UNKNOWN")) for row in verification_rows)
        summary = {
            "equivalent": states["EQUIVALENT"],
            "refuted": states["REFUTED"],
            "remaining": len(verification_rows) - states["EQUIVALENT"] - states["REFUTED"],
        }
        if summary != manifest["expected_by_model"][model]:
            raise ValueError(f"unexpected counts for {model}: {summary}")
        by_model[model] = summary
        totals.update(summary)

    expected_totals = Counter(manifest["expected_totals"])
    if totals != expected_totals:
        raise ValueError(f"unexpected six-model totals: {dict(totals)}")
    if expected_problem_count * len(manifest["models"]) != manifest["assigned_outputs"]:
        raise ValueError("assigned-output denominator is inconsistent")

    print(json.dumps({
        "status": "PASS",
        "problems": expected_problem_count,
        "models": manifest["models"],
        "assigned_outputs": manifest["assigned_outputs"],
        "totals": dict(totals),
        "by_model": by_model,
    }, indent=2))


if __name__ == "__main__":
    main()
