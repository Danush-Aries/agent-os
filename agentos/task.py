"""The unit of work in Agent OS: a Task, analogous to a process in a real OS."""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any

_counter = itertools.count(1)


class Priority(enum.IntEnum):
    """Named priority levels. Higher value = scheduled first."""

    LOW = 0
    NORMAL = 10
    HIGH = 20
    CRITICAL = 30


class TaskStatus(str, enum.Enum):
    PENDING = "pending"                    # waiting for dependencies / a free agent
    RUNNING = "running"                    # currently being handled by an agent
    AWAITING_APPROVAL = "awaiting_approval"  # paused for a human decision (HITL)
    COMPLETED = "completed"                # finished successfully; result is set
    FAILED = "failed"                      # handler raised / retries exhausted
    BLOCKED = "blocked"                    # a dependency failed, so this can never run


@dataclass
class Task:
    """A single job for an agent to handle.

    ``kind`` routes the task to an agent. ``deps`` are task ids that must reach
    COMPLETED before this task is runnable (how pipelines/DAGs are expressed).
    ``priority`` breaks ties among runnable tasks (higher first).

    Reliability knobs: ``max_retries`` with exponential ``backoff_base`` (+jitter
    applied by the kernel), a wall-clock ``timeout_s``, and ``requires_approval``
    which routes the task through the human-in-the-loop gate before it runs.
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = Priority.NORMAL
    deps: list[int] = field(default_factory=list)
    max_retries: int = 0
    backoff_base: float = 0.05
    timeout_s: float | None = None
    requires_approval: bool = False
    id: int = field(default_factory=lambda: next(_counter))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    owner: str | None = None      # name of the agent that handled it
    attempts: int = 0             # how many times the handler has been invoked
    approved: bool | None = None  # HITL decision, once made

    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "payload": self.payload,
            "priority": int(self.priority),
            "deps": list(self.deps),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "owner": self.owner,
            "attempts": self.attempts,
            "requires_approval": self.requires_approval,
            "approved": self.approved,
        }
