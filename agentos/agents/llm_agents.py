"""LLM-powered agents for Agent OS.

Two agents that put the optional :class:`~agentos.llm.LLMProvider` to work:

* :class:`LLMAgent` — runs a tool-calling loop. It hands the registry's tool
  specs to the model, executes any tool calls the model asks for, feeds the
  results back, and repeats until the model stops calling tools (or a small
  iteration / token budget is hit).
* :class:`PlannerAgent` — asks the model for a small task DAG (JSON, via a
  response schema), then spawns each step via ``ctx.spawn``, wiring deps from
  the plan's ``after`` indices to the real spawned task ids.

Both are fully exercisable offline with :class:`agentos.llm.MockLLM` — no
network, no API key. The module imports only the standard library plus the
already-dependency-free agent/tool primitives.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent import Agent, Context
from ..task import Task
from ..tools import Tool


# --------------------------------------------------------------------------- #
# LLMAgent — a tool-calling loop
# --------------------------------------------------------------------------- #
class LLMAgent(Agent):
    """Drive an LLM through a bounded tool-calling loop.

    Payload keys (all optional except ``prompt``):

    * ``prompt``          — the user message (required; ``""`` if absent).
    * ``system``          — an optional system prompt prepended to the messages.
    * ``max_iterations``  — cap on model round-trips (default 5) so a model that
      keeps asking for tools can never loop forever.
    * ``max_tokens``      — a soft budget: once cumulative input+output tokens
      exceed it, the loop stops after the current round.
    * ``use_tools``       — set ``False`` to run without offering tools.

    Returns ``{"text", "tokens", "input_tokens", "output_tokens", "cost_usd",
    "tool_calls", "iterations"}`` where ``tool_calls`` is how many tool
    invocations were made.

    Raises :class:`RuntimeError` if no LLM provider is configured on the kernel.
    """

    name = "llm"
    handles = {"llm"}

    def handle(self, task: Task, ctx: Context) -> Any:
        if ctx.llm is None:
            raise RuntimeError(
                "LLMAgent requires an LLM provider, but ctx.llm is None. "
                "Construct the Kernel with llm=MockLLM() (or another provider)."
            )

        payload = task.payload
        prompt = payload.get("prompt", "")
        system = payload.get("system")
        max_iterations = int(payload.get("max_iterations", 5))
        max_tokens = payload.get("max_tokens")
        use_tools = payload.get("use_tools", True)

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        specs = ctx.tools.specs() if (use_tools and ctx.tools) else None
        if not specs:
            specs = None

        total_in = total_out = 0
        total_cost = 0.0
        tool_call_count = 0
        iterations = 0
        text = ""

        for _ in range(max(1, max_iterations)):
            iterations += 1
            res = ctx.llm.complete(messages, tools=specs)
            total_in += res.input_tokens
            total_out += res.output_tokens
            total_cost += res.cost_usd
            text = res.text

            # No tools requested -> we have the final answer.
            if not res.tool_calls:
                break

            # Execute each requested tool and feed results back to the model.
            summaries: list[str] = []
            for call in res.tool_calls:
                tname = call.get("name", "")
                targs = call.get("arguments", {}) or {}
                result = ctx.call_tool(tname, targs)
                tool_call_count += 1
                if result.is_error:
                    summaries.append(f"{tname} -> ERROR: {result.error}")
                else:
                    summaries.append(f"{tname} -> {result.content}")

            # Record the turn. The tool summary goes in as a *user* message so
            # MockLLM's "last user message" trigger changes and the loop can
            # terminate deterministically instead of re-firing the same tool.
            messages.append({"role": "assistant", "content": text or ""})
            messages.append({"role": "user", "content": "tool results: " + "; ".join(summaries)})

            # Soft budget guard: stop once we have spent the token allowance.
            if max_tokens is not None and (total_in + total_out) >= int(max_tokens):
                break

        return {
            "text": text,
            "tokens": total_in + total_out,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": total_cost,
            "tool_calls": tool_call_count,
            "iterations": iterations,
        }


# --------------------------------------------------------------------------- #
# PlannerAgent — produce and spawn a small task DAG
# --------------------------------------------------------------------------- #
# JSON schema describing the plan we ask the model for. Every field is required
# so MockLLM's schema-stub path fills them, making the agent testable offline.
PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "payload": {"type": "object"},
                    "after": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["kind", "payload", "after"],
            },
        }
    },
    "required": ["steps"],
}


class PlannerAgent(Agent):
    """Turn ``payload["goal"]`` into a spawned task DAG.

    Asks the LLM for structured JSON matching :data:`PLAN_SCHEMA`, then spawns
    one task per step. Each step's ``after`` list holds *plan indices* of
    earlier steps; those are translated into ``deps`` on the real spawned task
    ids. Only backward references (an index strictly less than the current
    step) are honoured, so a self- or forward-reference in the plan can never
    deadlock the DAG.

    If the model output can't be parsed into at least one step, a deterministic
    fallback spawns a single ``echo`` task so the planner always makes progress.

    Returns the list of spawned task ids (in plan order).
    """

    name = "planner"
    handles = {"plan"}

    def handle(self, task: Task, ctx: Context) -> Any:
        if ctx.llm is None:
            raise RuntimeError(
                "PlannerAgent requires an LLM provider, but ctx.llm is None."
            )

        goal = task.payload.get("goal", "")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task planner. Break the goal into a small DAG of "
                    "steps. Each step has a 'kind', a 'payload' object, and an "
                    "'after' list of earlier step indices it depends on."
                ),
            },
            {"role": "user", "content": goal},
        ]

        steps = self._plan_steps(ctx, messages)
        if not steps:
            # Deterministic fallback: always make progress.
            return [ctx.spawn("echo", {"message": goal or "plan"})]

        spawned: list[int] = []
        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            kind = step.get("kind") or "echo"
            payload = step.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            after = step.get("after") or []
            deps = [
                spawned[j]
                for j in after
                if isinstance(j, int) and 0 <= j < idx and j < len(spawned)
            ]
            spawned.append(ctx.spawn(kind, payload, deps=deps))

        return spawned

    @staticmethod
    def _plan_steps(ctx: Context, messages: list[dict]) -> list[dict]:
        """Return the parsed step list, or ``[]`` if the model output is unusable."""
        try:
            res = ctx.llm.complete(messages, response_schema=PLAN_SCHEMA)
            data = json.loads(res.text)
            steps = data.get("steps")
            if isinstance(steps, list) and steps:
                return steps
        except Exception:  # noqa: BLE001 - any parse failure -> fallback
            pass
        return []


# --------------------------------------------------------------------------- #
# agent_as_tool — expose an agent kind to an LLM as a callable tool
# --------------------------------------------------------------------------- #
# Marker key used to tag a tool's return value as "please route this as a task".
AGENT_TASK_KEY = "_agentos_task"


def agent_as_tool(
    agent_kind: str, name: str, description: str, input_schema: dict
) -> Tool:
    """Wrap an agent ``kind`` as a :class:`~agentos.tools.Tool`.

    The returned tool's ``func`` does not run the agent itself — tools are
    synchronous and agents run as scheduled tasks. Instead, calling it packages
    the supplied arguments into a task descriptor::

        {"_agentos_task": {"kind": agent_kind, "payload": {<args>}}}

    A kernel/router (or an agent's tool loop) can detect the
    :data:`AGENT_TASK_KEY` marker in a :class:`ToolResult`'s content and
    ``ctx.spawn(kind, payload)`` accordingly — turning "the model asked to use
    tool X" into "spawn a task handled by agent X". This is the simplest correct
    wiring: it keeps the tool pure and side-effect-free while giving the caller
    everything needed to route the work.
    """

    def _func(**kwargs: Any) -> dict:
        return {AGENT_TASK_KEY: {"kind": agent_kind, "payload": dict(kwargs)}}

    return Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        func=_func,
    )


LLM_AGENTS = [LLMAgent, PlannerAgent]

__all__ = [
    "LLMAgent",
    "PlannerAgent",
    "PLAN_SCHEMA",
    "agent_as_tool",
    "AGENT_TASK_KEY",
    "LLM_AGENTS",
]
