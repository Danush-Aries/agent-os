"""Agents (the "processes" of Agent OS) and the Context handed to them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .task import Task
from .tools import ToolResult

if TYPE_CHECKING:  # avoid a runtime import cycle with kernel
    from .kernel import Kernel


class _TracingLLM:
    """Transparent LLM wrapper that records each call for observability.

    Emits an ``llm.call`` event carrying OpenTelemetry GenAI semantic-convention
    attributes (``gen_ai.system``, ``gen_ai.request.model``,
    ``gen_ai.usage.input_tokens`` / ``output_tokens``) and rolls token + cost
    totals into the tracer's metrics — so ``kernel.metrics()`` and
    ``kernel.trace(id)`` reflect real model usage without any agent-side code.
    """

    def __init__(self, provider, tracer, task_id: int) -> None:
        self._p = provider
        self._t = tracer
        self._tid = task_id

    def complete(self, messages, tools=None, response_schema=None):
        r = self._p.complete(messages, tools=tools, response_schema=response_schema)
        in_tok = getattr(r, "input_tokens", 0) or 0
        out_tok = getattr(r, "output_tokens", 0) or 0
        cost = getattr(r, "cost_usd", 0.0) or 0.0
        self._t.emit("llm.call", self._tid, **{
            "gen_ai.system": type(self._p).__name__,
            "gen_ai.request.model": getattr(r, "model", ""),
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
            "gen_ai.usage.cost_usd": round(cost, 6),
        })
        self._t.incr("llm.calls")
        self._t.incr("llm.input_tokens", in_tok)
        self._t.incr("llm.output_tokens", out_tok)
        self._t.incr("llm.cost_usd", cost)
        return r

    def stream(self, messages, tools=None):
        return self._p.stream(messages, tools=tools)

    def __getattr__(self, name):  # passthrough for .model, etc.
        return getattr(self._p, name)


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
        self.tracer = kernel.tracer
        # Wrap the provider so every call is traced + metered (None stays None).
        self.llm = (
            _TracingLLM(kernel.llm, kernel.tracer, task.id)
            if kernel.llm is not None else None
        )

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
