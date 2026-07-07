"""Human-in-the-loop: pause high-stakes work for an approval decision.

A task (or a tool) flagged ``requires_approval`` is moved to AWAITING_APPROVAL
and handed to the ``ApprovalGate``. A gate resolves a decision via an injectable
policy: auto-approve/deny (deterministic, used in tests), or a pending queue
that a human answers through the CLI/REST interface.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from .task import Task

# A policy inspects a task and returns True (approve), False (deny), or
# None (defer — leave it pending for an out-of-band human decision).
ApprovalPolicy = Callable[[Task], "bool | None"]


def auto_approve(_: Task) -> bool:
    return True


def auto_deny(_: Task) -> bool:
    return False


@dataclass
class PendingApproval:
    task_id: int
    kind: str
    summary: str


class ApprovalGate:
    def __init__(self, policy: ApprovalPolicy = auto_approve) -> None:
        self.policy = policy
        self._pending: dict[int, PendingApproval] = {}
        self._decisions: dict[int, bool] = {}
        self._lock = threading.Lock()

    def request(self, task: Task) -> bool | None:
        """Return the decision, or None if it must wait for a human."""
        decision = self.policy(task)
        if decision is None:
            with self._lock:
                # Honor an already-recorded out-of-band decision if present.
                if task.id in self._decisions:
                    return self._decisions.pop(task.id)
                self._pending[task.id] = PendingApproval(
                    task_id=task.id, kind=task.kind,
                    summary=str(task.payload)[:200],
                )
            return None
        return decision

    # --- out-of-band resolution (CLI / REST) ---------------------------------

    def resolve(self, task_id: int, approved: bool) -> None:
        with self._lock:
            self._pending.pop(task_id, None)
            self._decisions[task_id] = approved

    def take_decision(self, task_id: int) -> bool | None:
        with self._lock:
            return self._decisions.pop(task_id, None)

    def pending(self) -> list[PendingApproval]:
        with self._lock:
            return list(self._pending.values())
