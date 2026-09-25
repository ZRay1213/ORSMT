from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter


def read_rows(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def id_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    inputs = list(read_rows(args.input))
    ledger = list(read_rows(args.ledger))
    input_ids = [str(row["id"]) for row in inputs]
    output_ids = [str(row["id"]) for row in ledger]
    sources = Counter(str(row["metadata"].get("source")) for row in ledger)
    error_sources = Counter()
    error_types = Counter()
    relation_sets = Counter()
    end_to_end = Counter()
    for row in ledger:
        metadata = row["metadata"]
        errors = metadata.get("adapter_errors", {})
        if errors:
            error_sources[str(metadata.get("source"))] += 1
            for message in errors.values():
                error_types[message.split(":", 1)[0]] += 1
        relation_sets[tuple(sorted(row["profile"]["by_relation"]))] += 1
        end_to_end[row["profile"]["end_to_end"]] += 1

    summary = {
        "input_rows": len(inputs),
        "ledger_rows": len(ledger),
        "unique_input_ids": len(set(input_ids)),
        "unique_ledger_ids": len(set(output_ids)),
        "exact_id_set": set(input_ids) == set(output_ids),
        "input_id_hash": id_hash(input_ids),
        "ledger_id_hash": id_hash(output_ids),
        "sources": dict(sorted(sources.items())),
        "adapter_error_rows": sum(error_sources.values()),
        "adapter_error_sources": dict(sorted(error_sources.items())),
        "adapter_error_types": dict(sorted(error_types.items())),
        "relation_sets": {",".join(key): value for key, value in relation_sets.items()},
        "end_to_end": dict(sorted(end_to_end.items())),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not (
        summary["input_rows"] == summary["ledger_rows"]
        and summary["unique_input_ids"] == summary["input_rows"]
        and summary["unique_ledger_ids"] == summary["ledger_rows"]
        and summary["exact_id_set"]
        and summary["input_id_hash"] == summary["ledger_id_hash"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
