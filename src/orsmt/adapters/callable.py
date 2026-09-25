from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..ir import ModelIR
from .protocols import ModelAdapter


@dataclass(frozen=True)
class CallableRuntimeAdapter:
    """Wrap a host pipeline's pre-solve capture function as an ORSMT adapter.

    The callable may return a ``ModelIR`` directly, a serialized model mapping,
    or a capture envelope such as ``{"status": "CAPTURED", "ir": ...}``.
    """

    capture_fn: Callable[[Any], Any]
    model_adapter: ModelAdapter
    accepted_statuses: frozenset[str] = field(
        default_factory=lambda: frozenset({"CAPTURED", "OK"})
    )

    def capture(self, executable: Any) -> ModelIR:
        payload = self.capture_fn(executable)
        if isinstance(payload, ModelIR):
            return payload
        if not isinstance(payload, Mapping):
            raise TypeError("runtime capture must return ModelIR or a mapping")
        if "status" in payload and str(payload["status"]).upper() not in self.accepted_statuses:
            detail = payload.get("error", payload.get("reason", "capture failed"))
            raise ValueError(f"runtime capture status={payload['status']}: {detail}")
        raw_model = payload.get("ir", payload.get("model", payload))
        return self.model_adapter.to_ir(raw_model)
