from .builtins import BUILTIN_AGENTS, CalcAgent, EchoAgent, PipelineAgent
from .llm_agents import (
    AGENT_TASK_KEY,
    LLM_AGENTS,
    PLAN_SCHEMA,
    LLMAgent,
    PlannerAgent,
    agent_as_tool,
)

# Every agent that ships with Agent OS.
ALL_AGENTS = [*BUILTIN_AGENTS, *LLM_AGENTS]

__all__ = [
    "BUILTIN_AGENTS", "CalcAgent", "EchoAgent", "PipelineAgent",
    "LLM_AGENTS", "LLMAgent", "PlannerAgent", "agent_as_tool",
    "AGENT_TASK_KEY", "PLAN_SCHEMA", "ALL_AGENTS",
]
