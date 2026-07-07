"""Agent registry: maps a task to the agent that should run it."""

from __future__ import annotations

from .agent import Agent
from .task import Task


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: list[Agent] = []

    def register(self, agent: Agent) -> None:
        if any(a.name == agent.name for a in self._agents):
            raise ValueError(f"an agent named '{agent.name}' is already registered")
        self._agents.append(agent)

    def agents(self) -> list[Agent]:
        return list(self._agents)

    def find(self, task: Task) -> Agent | None:
        """First registered agent that accepts this task (registration order)."""
        for agent in self._agents:
            if agent.can_handle(task):
                return agent
        return None
