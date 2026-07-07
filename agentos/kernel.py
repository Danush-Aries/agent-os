"""The Agent OS kernel: boot agents, submit tasks, run the dispatch loop.

The kernel owns shared memory, the agent registry, the scheduler, and the
cross-cutting subsystems — tracing, guardrails, human-in-the-loop approval,
tools, and an optional LLM provider. Its run loop repeatedly schedules every
runnable task (concurrently, up to ``max_workers``), applying retries with
exponential backoff + jitter and per-task timeouts, checkpointing state after
each transition so a crashed run can resume.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent import Agent, Context
from .guardrails import Guardrail
from .hitl import ApprovalGate
from .memory import Blackboard
from .registry import AgentRegistry
from .scheduler import Scheduler
from .task import Task, TaskStatus
from .tools import ToolRegistry
from .tracing import Tracer

log = logging.getLogger("agentos")


@dataclass
class RunReport:
    completed: int = 0
    failed: int = 0
    blocked: int = 0
    unhandled: int = 0
    retried: int = 0
    awaiting_approval: int = 0
    dlq: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.failed == 0 and self.blocked == 0
                and self.unhandled == 0 and not self.dlq)


def _jitter(seed: str) -> float:
    """Deterministic pseudo-jitter in [0, 1) derived from a string (no RNG)."""
    h = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(h[:4], "big") / 2**32


class Kernel:
    def __init__(
        self,
        memory_path: str = ":memory:",
        *,
        max_workers: int = 1,
        tracer: Tracer | None = None,
        guardrail: Guardrail | None = None,
        approvals: ApprovalGate | None = None,
        tools: ToolRegistry | None = None,
        llm: object | None = None,
        checkpoint_path: str | None = None,
        sleep: "callable" = time.sleep,
    ) -> None:
        self.memory = Blackboard(memory_path)
        self.registry = AgentRegistry()
        self.scheduler = Scheduler()
        self.tracer = tracer or Tracer()
        self.guardrail = guardrail
        self.approvals = approvals or ApprovalGate()
        self.tools = tools or ToolRegistry()
        self.llm = llm  # optional LLM provider exposed to agents via Context
        self.max_workers = max_workers
        self.checkpoint_path = checkpoint_path
        self._sleep = sleep      # injectable so tests don't wait on real backoff
        self._lock = threading.Lock()

    # --- setup ---------------------------------------------------------------

    def register(self, agent: Agent) -> "Kernel":
        self.registry.register(agent)
        return self

    def submit(self, task: Task) -> int:
        tid = self.scheduler.add(task)
        self.tracer.emit("task.submitted", tid, kind=task.kind, priority=int(task.priority))
        self._checkpoint()
        return tid

    # --- execution of a single task (runs in a worker thread) ----------------

    def _run_handler(self, agent: Agent, task: Task, ctx: Context):
        """Invoke agent.handle with an optional per-attempt timeout."""
        if task.timeout_s is None:
            return agent.handle(task, ctx)
        box: dict[str, object] = {}

        def target():
            try:
                box["ok"] = agent.handle(task, ctx)
            except Exception as exc:  # propagate to the attempt loop
                box["err"] = exc

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(task.timeout_s)
        if t.is_alive():
            raise TimeoutError(f"task {task.id} exceeded timeout {task.timeout_s}s")
        if "err" in box:
            raise box["err"]  # type: ignore[misc]
        return box.get("ok")

    def _execute(self, task: Task, report: RunReport) -> None:
        agent = self.registry.find(task)
        if agent is None:
            task.status = TaskStatus.FAILED
            task.error = f"no agent handles kind '{task.kind}'"
            report.unhandled += 1
            self.tracer.emit("task.unhandled", task.id, kind=task.kind)
            return

        # input guardrail on the payload
        if self.guardrail is not None:
            gr = self.guardrail.check(json.dumps(task.payload))
            if not gr.allowed:
                task.status = TaskStatus.FAILED
                task.error = f"guardrail blocked input: {gr.reason}"
                report.failed += 1
                self.tracer.emit("guardrail.blocked", task.id, stage="input", reason=gr.reason)
                return

        task.owner = agent.name
        span = self.tracer.start_span(f"task:{task.kind}", task.id, **{"gen_ai.operation.name": task.kind})
        ctx = Context(self, task)

        for attempt in range(task.max_retries + 1):
            task.attempts += 1
            task.status = TaskStatus.RUNNING
            self.tracer.emit("task.started", task.id, attempt=task.attempts, owner=agent.name)
            try:
                result = self._run_handler(agent, task, ctx)
                # output guardrail (may redact strings)
                if self.guardrail is not None and isinstance(result, str):
                    gr = self.guardrail.check(result)
                    if not gr.allowed:
                        raise ValueError(f"guardrail blocked output: {gr.reason}")
                    result = gr.text
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.memory.put(f"task:{task.id}:result", result)
                report.completed += 1
                self.tracer.emit("task.completed", task.id, attempts=task.attempts)
                self.tracer.incr("tasks.completed")
                break
            except Exception as exc:
                task.error = f"{type(exc).__name__}: {exc}"
                if attempt < task.max_retries:
                    report.retried += 1
                    self.tracer.emit("task.retried", task.id, attempt=task.attempts, error=task.error)
                    self.tracer.incr("tasks.retried")
                    delay = task.backoff_base * (2 ** attempt) * (1 + _jitter(f"{task.id}:{attempt}"))
                    self._sleep(delay)
                    continue
                task.status = TaskStatus.FAILED
                report.failed += 1
                report.dlq.append(task.id)
                self.memory.put(f"task:{task.id}:dlq", task.error)
                self.tracer.emit("task.failed", task.id, attempts=task.attempts, error=task.error)
                self.tracer.incr("tasks.failed")
        self.tracer.end_span(span, status=task.status.value)

    # --- HITL gate -----------------------------------------------------------

    def _gate(self, task: Task, report: RunReport) -> bool:
        """Return True if the task may run now; handle approval transitions."""
        if not task.requires_approval or task.approved is True:
            return True
        decision = self.approvals.request(task)
        if decision is True:
            task.approved = True
            self.tracer.emit("approval.granted", task.id)
            return True
        if decision is False:
            task.approved = False
            task.status = TaskStatus.FAILED
            task.error = "denied by approval policy"
            report.failed += 1
            self.tracer.emit("approval.denied", task.id)
            return False
        # deferred: wait for an out-of-band human decision
        task.status = TaskStatus.AWAITING_APPROVAL
        self.tracer.emit("approval.requested", task.id)
        return False

    def _poll_approvals(self, report: RunReport) -> None:
        for task in self.scheduler.all():
            if task.status != TaskStatus.AWAITING_APPROVAL:
                continue
            decision = self.approvals.take_decision(task.id)
            if decision is True:
                task.approved = True
                task.status = TaskStatus.PENDING
                self.tracer.emit("approval.granted", task.id)
            elif decision is False:
                task.status = TaskStatus.FAILED
                task.error = "denied"
                report.failed += 1
                self.tracer.emit("approval.denied", task.id)

    # --- run loop ------------------------------------------------------------

    def run(self, max_steps: int = 10_000) -> RunReport:
        report = RunReport()
        steps = 0
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            inflight: dict[cf.Future, Task] = {}
            while True:
                self._poll_approvals(report)
                batch = self.scheduler.ready_batch()
                # apply the HITL gate; only truly-runnable tasks proceed
                runnable = [t for t in batch if self._gate(t, report)]

                free = self.max_workers - len(inflight)
                for task in runnable[:max(free, 0)]:
                    if steps >= max_steps:
                        break
                    task.status = TaskStatus.RUNNING  # claim so it isn't re-picked
                    fut = pool.submit(self._execute, task, report)
                    inflight[fut] = task
                    steps += 1

                if not inflight:
                    if not self.scheduler.has_pending():
                        break
                    # only awaiting-approval / blocked remain and nothing is running
                    if not any(self._gate_pending()):
                        break
                    continue

                done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    inflight.pop(fut)
                    exc = fut.exception()
                    if exc:  # defensive: _execute shouldn't raise
                        log.exception("dispatch worker crashed", exc_info=exc)
                self._checkpoint()

        report.blocked = sum(1 for t in self.scheduler.all() if t.status == TaskStatus.BLOCKED)
        report.awaiting_approval = sum(
            1 for t in self.scheduler.all() if t.status == TaskStatus.AWAITING_APPROVAL)
        return report

    def _gate_pending(self) -> list[bool]:
        """True for each task still awaiting a human decision (loop-guard)."""
        return [t.status == TaskStatus.AWAITING_APPROVAL for t in self.scheduler.all()
                if t.status == TaskStatus.AWAITING_APPROVAL]

    # --- checkpoint / resume / rollback --------------------------------------

    def _checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        snapshot = {"tasks": [t.as_dict() for t in self.scheduler.all()]}
        Path(self.checkpoint_path).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    @classmethod
    def resume(cls, checkpoint_path: str, **kw) -> "Kernel":
        """Reconstruct a kernel's task table from a checkpoint file.

        Non-terminal tasks are reset to PENDING so the run continues; agents
        must be re-registered by the caller before ``run()``.
        """
        k = cls(checkpoint_path=checkpoint_path, **kw)
        data = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        for td in data.get("tasks", []):
            status = TaskStatus(td["status"])
            if status in (TaskStatus.RUNNING, TaskStatus.AWAITING_APPROVAL):
                status = TaskStatus.PENDING
            task = Task(
                kind=td["kind"], payload=td["payload"], priority=td["priority"],
                deps=td["deps"], requires_approval=td.get("requires_approval", False),
                id=td["id"], status=status, result=td.get("result"),
                error=td.get("error"), owner=td.get("owner"), attempts=td.get("attempts", 0),
                approved=td.get("approved"),
            )
            k.scheduler.add(task)
        return k

    # --- introspection -------------------------------------------------------

    def ps(self) -> list[Task]:
        return self.scheduler.all()

    def metrics(self) -> dict[str, float]:
        tasks = self.scheduler.all()
        m = {
            "tasks_total": len(tasks),
            "queue_depth": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "awaiting_approval": sum(1 for t in tasks if t.status == TaskStatus.AWAITING_APPROVAL),
            "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
        }
        m.update(self.tracer.snapshot())
        return m

    def trace(self, task_id: int):
        return self.tracer.trace(task_id)

    def close(self) -> None:
        self.memory.close()
