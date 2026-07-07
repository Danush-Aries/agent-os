"""Convenience factory: a fully-loaded Agent OS kernel in one call.

``default_kernel()`` wires everything the OS ships with — the built-in and
LLM/planner agents, the real-world tool integrations (web search, GitHub, HTTP,
filesystem, shell), an LLM provider (MockLLM by default, offline), guardrails,
and the human-in-the-loop gate — so a caller can go from import to running
agents without assembling the pieces by hand.
"""

from __future__ import annotations

from .agents import ALL_AGENTS
from .guardrails import Guardrail
from .hitl import ApprovalGate, auto_approve
from .kernel import Kernel
from .llm import get_provider
from .tools import ToolRegistry


def default_kernel(
    *,
    memory_path: str = ":memory:",
    max_workers: int = 1,
    llm: object | None = None,
    with_tools: bool = True,
    with_guardrails: bool = True,
    approval_policy=auto_approve,
    fs_root: str | None = None,
    checkpoint_path: str | None = None,
) -> Kernel:
    """Return a Kernel pre-loaded with all agents, tools, and subsystems.

    - ``llm`` defaults to ``get_provider()`` (MockLLM unless ``AGENTOS_LLM`` is set).
    - ``with_tools`` registers the integration toolpacks; write/create/shell
      tools among them are approval-gated, governed by ``approval_policy``.
    - ``approval_policy`` defaults to auto-approve for convenience; pass a
      deferring policy (``lambda _: None``) to force real human approval.
    """
    tools = ToolRegistry()
    if with_tools:
        # Imported lazily: the integration tools only need httpx at call time.
        from .toolpacks import register_all
        register_all(tools, fs_root=fs_root)

    k = Kernel(
        memory_path=memory_path,
        max_workers=max_workers,
        tools=tools,
        llm=llm if llm is not None else get_provider(),
        guardrail=Guardrail() if with_guardrails else None,
        approvals=ApprovalGate(policy=approval_policy),
        checkpoint_path=checkpoint_path,
    )
    for agent_cls in ALL_AGENTS:
        k.register(agent_cls())
    return k
