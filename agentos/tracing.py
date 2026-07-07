"""Observability for Agent OS: structured events, spans, and metrics.

Every meaningful state transition emits a typed ``Event``. Events with a
``span`` build a per-root-task span tree carrying OpenTelemetry GenAI
semantic-convention attributes (``gen_ai.system``, ``gen_ai.request.model``,
token counts, cost) so a trace can later be exported to an OTLP backend.

The tracer is pure-stdlib and thread-safe; it doubles as the immutable audit
log. A PII/secret redactor can be attached so sensitive values never land in
stored events.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

Redactor = Callable[[str], str]


@dataclass
class Event:
    seq: int
    ts: float
    kind: str                      # e.g. task.submitted, task.completed, llm.call, tool.call
    task_id: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "kind": self.kind,
                "task_id": self.task_id, "attrs": self.attrs}


@dataclass
class Span:
    name: str
    task_id: int | None
    start: float
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end is None else round((self.end - self.start) * 1000, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_id": self.task_id,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
            "children": [c.as_dict() for c in self.children],
        }


class Tracer:
    def __init__(self, redactor: Redactor | None = None, clock: Callable[[], float] | None = None) -> None:
        self._events: list[Event] = []
        self._spans: dict[int, Span] = {}   # root task id -> span
        self._lock = threading.Lock()
        self._seq = 0
        self._redactor = redactor
        # Injectable clock keeps tests deterministic; defaults to wall time.
        self._clock = clock or time.time
        self.counters: dict[str, float] = {}

    # --- events --------------------------------------------------------------

    def emit(self, event: str, task_id: int | None = None, **attrs: Any) -> Event:
        if self._redactor:
            attrs = {k: (self._redactor(v) if isinstance(v, str) else v) for k, v in attrs.items()}
        with self._lock:
            self._seq += 1
            ev = Event(seq=self._seq, ts=self._clock(), kind=event, task_id=task_id, attrs=attrs)
            self._events.append(ev)
        return ev

    def events(self, task_id: int | None = None) -> list[Event]:
        with self._lock:
            evs = list(self._events)
        return [e for e in evs if task_id is None or e.task_id == task_id]

    # --- spans ---------------------------------------------------------------

    def start_span(self, name: str, task_id: int | None = None, **attrs: Any) -> Span:
        span = Span(name=name, task_id=task_id, start=self._clock(), attrs=dict(attrs))
        if task_id is not None:
            with self._lock:
                self._spans.setdefault(task_id, span)
        return span

    def end_span(self, span: Span, **attrs: Any) -> None:
        span.end = self._clock()
        span.attrs.update(attrs)

    def trace(self, task_id: int) -> dict[str, Any] | None:
        """Return the span tree + events for a root task, OTel-flavored."""
        span = self._spans.get(task_id)
        return {
            "task_id": task_id,
            "span": span.as_dict() if span else None,
            "events": [e.as_dict() for e in self.events(task_id)],
        }

    # --- metrics -------------------------------------------------------------

    def incr(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self.counters)
