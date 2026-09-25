from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping


class Verdict(str, Enum):
    CERTIFIED = "CERTIFIED"
    REFUTED = "REFUTED"
    UNDECIDED = "UNDECIDED"


class Relation(str, Enum):
    FORMALIZATION = "formalization"
    IMPLEMENTATION = "implementation"
    REPORTING = "reporting"
    REFORMULATION = "reformulation"


@dataclass(frozen=True)
class ReplayRecord:
    passed: bool
    backend: str
    detail: str = ""


@dataclass(frozen=True)
class EvidenceRecord:
    relation: Relation
    claim_id: str
    verdict: Verdict
    query: str
    witness: Mapping[str, Fraction] = field(default_factory=dict)
    implicated: tuple[str, ...] = ()
    source_span: str | None = None
    violation_margin: Fraction | None = None
    normalized_violation: float | None = None
    replay: ReplayRecord | None = None
    reason: str | None = None
    duration_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Fraction):
                return str(value)
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def aggregate(records: list[EvidenceRecord]) -> Verdict:
    applicable = [record for record in records if record.reason != "not_applicable"]
    if not applicable:
        return Verdict.UNDECIDED
    if any(record.verdict == Verdict.REFUTED for record in applicable):
        return Verdict.REFUTED
    if all(record.verdict == Verdict.CERTIFIED for record in applicable):
        return Verdict.CERTIFIED
    return Verdict.UNDECIDED
