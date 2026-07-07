"""Offline tests for the LLM-powered agents (LLMAgent, PlannerAgent).

Everything runs against :class:`agentos.llm.MockLLM` — no network, no API key.
"""

from __future__ import annotations

from agentos import Kernel, Task, TaskStatus
from agentos.agents.builtins import CalcAgent, EchoAgent
from agentos.agents.llm_agents import (
    AGENT_TASK_KEY,
    LLM_AGENTS,
    LLMAgent,
    PlannerAgent,
    agent_as_tool,
)
from agentos.llm import MockLLM
from agentos.tools import Tool, ToolRegistry


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _calc_tool() -> Tool:
    """A calculator-style tool MockLLM recognises for arithmetic prompts."""
    def _calc(a, op, b):
        ops = {"+": a + b, "-": a - b, "*": a * b, "/": (a / b if b else None)}
        return ops[op]

    return Tool(
        name="calc",
        description="Evaluate a simple a <op> b arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
                "b": {"type": "number"},
            },
            "required": ["a", "op", "b"],
        },
        func=_calc,
    )


def _boot(*, with_calc_tool: bool = True) -> Kernel:
    tools = ToolRegistry()
    if with_calc_tool:
        tools.register(_calc_tool())
    k = Kernel(llm=MockLLM(), tools=tools)
    for agent_cls in LLM_AGENTS:
        k.register(agent_cls())
    k.register(EchoAgent())
    k.register(CalcAgent())
    return k


# --------------------------------------------------------------------------- #
# (1) LLMAgent drives a tool call
# --------------------------------------------------------------------------- #
def test_llm_agent_drives_tool_call():
    k = _boot()
    tid = k.submit(Task(kind="llm", payload={"prompt": "2 + 3"}))
    k.run()

    result = k.scheduler.get(tid).result
    assert k.scheduler.get(tid).status == TaskStatus.COMPLETED
    assert result["tool_calls"] >= 1          # at least one tool was invoked
    assert isinstance(result["text"], str) and result["text"]  # final text present
    assert result["tokens"] > 0


def test_llm_agent_use_tool_directive_also_fires():
    """The explicit 'use tool <name>' trigger works too (args may be empty)."""
    k = _boot()
    tid = k.submit(Task(kind="llm", payload={"prompt": "please use tool calc now"}))
    k.run()
    result = k.scheduler.get(tid).result
    assert result["tool_calls"] >= 1


# --------------------------------------------------------------------------- #
# (2) LLMAgent with a plain prompt echoes and accounts tokens
# --------------------------------------------------------------------------- #
def test_llm_agent_plain_prompt_echoes_with_tokens():
    k = _boot()
    tid = k.submit(Task(kind="llm", payload={"prompt": "hello world"}))
    k.run()

    result = k.scheduler.get(tid).result
    assert k.scheduler.get(tid).status == TaskStatus.COMPLETED
    assert result["text"] == "echo: hello world"
    assert result["tool_calls"] == 0
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["tokens"] == result["input_tokens"] + result["output_tokens"]


def test_llm_agent_requires_provider():
    """With no LLM configured the agent raises a clear error (task FAILS)."""
    k = Kernel()  # llm defaults to None
    k.register(LLMAgent())
    tid = k.submit(Task(kind="llm", payload={"prompt": "hi"}))
    k.run()
    task = k.scheduler.get(tid)
    assert task.status == TaskStatus.FAILED
    assert "ctx.llm is None" in task.error


# --------------------------------------------------------------------------- #
# (3) PlannerAgent spawns children through a real Kernel
# --------------------------------------------------------------------------- #
def test_planner_spawns_children_and_returns_ids():
    k = _boot()
    tid = k.submit(Task(kind="plan", payload={"goal": "do the thing"}))
    k.run()

    plan_task = k.scheduler.get(tid)
    assert plan_task.status == TaskStatus.COMPLETED

    child_ids = plan_task.result
    assert isinstance(child_ids, list)
    assert len(child_ids) >= 1

    existing_ids = {t.id for t in k.ps()}
    for cid in child_ids:
        assert cid in existing_ids
        assert cid != tid  # children are distinct from the plan task itself


def test_planner_fallback_when_no_schema_output(monkeypatch):
    """If the model yields no usable plan, a single echo child is spawned."""
    k = _boot()

    # Force the planner's structured call to return an unparseable body.
    class _EmptyLLM(MockLLM):
        def complete(self, messages, tools=None, response_schema=None):
            res = super().complete(messages, tools=None, response_schema=None)
            res.text = "not json at all"
            return res

    k.llm = _EmptyLLM()
    tid = k.submit(Task(kind="plan", payload={"goal": "fallback goal"}))
    k.run()

    child_ids = k.scheduler.get(tid).result
    assert len(child_ids) == 1
    child = k.scheduler.get(child_ids[0])
    assert child.kind == "echo"
    assert child.status == TaskStatus.COMPLETED
    assert child.result == "fallback goal"


# --------------------------------------------------------------------------- #
# agent_as_tool wiring
# --------------------------------------------------------------------------- #
def test_agent_as_tool_packages_task_descriptor():
    tool = agent_as_tool(
        "llm",
        name="ask_llm",
        description="Ask the LLM agent a question.",
        input_schema={
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
        },
    )
    reg = ToolRegistry()
    reg.register(tool)

    out = reg.call("ask_llm", {"prompt": "hi there"})
    assert not out.is_error
    assert out.content[AGENT_TASK_KEY] == {"kind": "llm", "payload": {"prompt": "hi there"}}


def test_llm_agents_export_list():
    assert LLM_AGENTS == [LLMAgent, PlannerAgent]
