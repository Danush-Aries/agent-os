"""End-to-end integration: the assembled OS, not individual subsystems.

Proves the merge holds together — default_kernel wires agents + tools + LLM +
guardrails + HITL, an LLMAgent drives a real tool, and approval-gated tools are
governed by the ApprovalGate. Everything runs offline (MockLLM + fakes).
"""

from __future__ import annotations

from agentos import Task, default_kernel
from agentos.agents import ALL_AGENTS
from agentos.hitl import ApprovalGate
from agentos.llm import MockLLM
from agentos.tools import Tool


def test_default_kernel_boots_everything():
    k = default_kernel()
    names = {a.name for a in k.registry.agents()}
    assert {"echo", "calc", "pipeline", "llm", "planner"} <= names
    # integration tools are registered on the kernel's tool registry
    assert {"web_search", "gh_search_repos", "fs_read", "shell_run"} <= set(k.tools.names())
    k.close()


def test_llm_agent_calls_a_registered_tool():
    k = default_kernel(llm=MockLLM())
    # register a deterministic calc tool the MockLLM knows how to trigger
    k.tools.register(Tool(
        name="calc",
        description="add two numbers",
        input_schema={"type": "object",
                      "properties": {"a": {"type": "number"}, "op": {"type": "string"},
                                     "b": {"type": "number"}},
                      "required": ["a", "op", "b"]},
        func=lambda a, op, b: a + b if op == "+" else None,
    ))
    tid = k.submit(Task(kind="llm", payload={"prompt": "please compute 2 + 3"}))
    k.run()
    task = k.scheduler.get(tid)
    assert task.status.value == "completed"
    assert task.result["tool_calls"] >= 1
    assert task.result["cost_usd"] >= 0
    k.close()


def test_pipeline_still_works_through_default_kernel():
    k = default_kernel()
    tid = k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    k.run()
    reduce_tasks = [t for t in k.ps() if t.kind == "_sum_reduce"]
    assert reduce_tasks and reduce_tasks[0].result == 10
    k.close()


def test_approval_gated_tool_defers_without_human():
    # A deferring policy leaves approval-required work parked, not auto-run.
    gate = ApprovalGate(policy=lambda _: None)
    k = default_kernel(approval_policy=lambda _: None)
    # fs_write is approval-gated at the tool level; here we assert the task-level
    # HITL gate parks a requires_approval task until a decision arrives.
    tid = k.submit(Task(kind="echo", payload={"message": "sensitive"}, requires_approval=True))
    report = k.run()
    assert k.scheduler.get(tid).status.value == "awaiting_approval"
    assert report.awaiting_approval == 1
    # now approve out-of-band and resume
    k.approvals.resolve(tid, True)
    k.run()
    assert k.scheduler.get(tid).status.value == "completed"
    k.close()


def test_guardrail_blocks_secret_in_payload():
    k = default_kernel()
    tid = k.submit(Task(kind="echo", payload={"message": "AKIAIOSFODNN7EXAMPLE"}))
    k.run()
    assert k.scheduler.get(tid).status.value == "failed"
    assert "guardrail" in (k.scheduler.get(tid).error or "")
    k.close()
