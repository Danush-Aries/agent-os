"""Agent OS — a tiny operating system for cooperating agents.

Register agents, submit tasks (with priorities and dependencies), and let the
kernel dispatch them against a shared-memory blackboard.

    from agentos import Kernel, Task
    from agentos.agents import BUILTIN_AGENTS

    k = Kernel()
    for A in BUILTIN_AGENTS:
        k.register(A())
    k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    report = k.run()
"""

from .agent import Agent, Context
from .boot import default_kernel
from .kernel import Kernel, RunReport
from .task import Priority, Task, TaskStatus

__version__ = "0.2.0"
__all__ = [
    "Agent", "Context", "Kernel", "RunReport", "Task", "TaskStatus", "Priority",
    "default_kernel",
]
