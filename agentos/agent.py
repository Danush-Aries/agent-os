"""Agents (the "processes" of Agent OS) and the Context handed to them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .task import Task
from .tools import ToolResult

if TYPE_CHECKING:  # avoid a runtime import cycle with kernel
    from .kernel import Kernel


class Context:
    """What an agent is given when it runs a task.

    It exposes shared memory, the tool registry, an optional LLM provider, and
    the ability to spawn / delegate follow-up tasks — so an agent can decompose
    work or call out to tools and models without touching kernel internals.
    """

    def __init__(self, kernel: "Kernel", task: Task) -> None:
        self._kernel = kernel
        self.task = task
        self.memory = kernel.memory          # shared blackboard
        self.tools = kernel.tools            # ToolRegistry
        self.llm = kernel.llm                # LLM provider or None
        self.tracer = kernel.tracer

    # --- spawning / delegation ----------------------------------------------

    def spawn(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int | None = None,
        deps: list[int] | None = None,
        requires_approval: bool = False,
        max_retries: int = 0,
        timeout_s: float | None = None,
    ) -> int:
        """Submit a new task to the OS and return its id."""
        from .task import Priority
        t = Task(
            kind=kind,
            payload=payload or {},
            priority=Priority.NORMAL if priority is None else priority,
            deps=deps or [],
            requires_approval=requires_approval,
            max_retries=max_retries,
            timeout_s=timeout_s,
        )
        return self._kernel.submit(t)

    def delegate(self, kind: str, payload: dict[str, Any] | None = None, **kw) -> int:
        """Hand a subtask to whichever agent handles ``kind`` (alias of spawn)."""
        self.tracer.emit("agent.delegate", self.task.id, to_kind=kind)
        return self.spawn(kind, payload, **kw)

    def result_of(self, task_id: int) -> Any:
        return self.memory.get(f"task:{task_id}:result")

    # --- tools ---------------------------------------------------------------

    def call_tool(self, name: str, args: dict) -> ToolResult:
        """Invoke a registered tool (validated + rate-limited by the registry)."""
        self.tracer.emit("tool.call", self.task.id, tool=name)
        result = self._kernel.tools.call(name, args)
        self.tracer.emit("tool.result", self.task.id, tool=name, is_error=result.is_error)
        return result


class Agent:
    """Base class. Subclass and set ``name`` + ``handles``, override ``handle``.

    ``handles`` is the set of task kinds this agent accepts. The registry uses
    it to route tasks; override ``can_handle`` for finer control.
    """

    name: str = "agent"
    handles: set[str] = set()

    def can_handle(self, task: Task) -> bool:
        return task.kind in self.handles

    def handle(self, task: Task, ctx: Context) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError(f"{self.name} does not implement handle()")
