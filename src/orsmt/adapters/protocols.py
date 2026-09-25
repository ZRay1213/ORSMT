from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..evidence import EvidenceRecord
from ..ir import ModelIR, ResultRecord, SourceObligation


@runtime_checkable
class SourceAdapter(Protocol):
    def obligations(self, source: Any) -> Sequence[SourceObligation]: ...


@runtime_checkable
class ModelAdapter(Protocol):
    def to_ir(self, model: Any) -> ModelIR: ...


@runtime_checkable
class RuntimeAdapter(Protocol):
    def capture(self, executable: Any) -> ModelIR: ...


@runtime_checkable
class ResultAdapter(Protocol):
    def to_result(self, result: Any) -> ResultRecord: ...


@runtime_checkable
class NativeReplayAdapter(Protocol):
    def replay(self, model: Any, point: Mapping[str, object]) -> tuple[bool | None, str]: ...


@runtime_checkable
class RepairAdapter(Protocol):
    def propose(self, artifact: Any, evidence: Sequence[EvidenceRecord]) -> Any: ...
