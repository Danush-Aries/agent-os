"""02 - Priorities and dependencies.

Two scheduling guarantees, shown live:

* Among runnable tasks, higher ``priority`` runs first (ties break FIFO).
* A task with ``deps`` only becomes runnable once every dependency COMPLETED.

    uv run python examples/02_priorities_and_deps.py
"""

from __future__ import annotations

from agentos import Agent, Context, Kernel, Task
from agentos.agents import BUILTIN_AGENTS


class Recorder(Agent):
    """Records the order it is dispatched in so we can observe scheduling."""

    name = "recorder"
    handles = {"record"}

    def __init__(self) -> None:
        self.order: list[str] = []

    def handle(self, task: Task, ctx: Context):
        self.order.append(task.payload["label"])
        return task.payload["label"]


def main() -> int:
    k = Kernel()
    recorder = Recorder()
    k.register(recorder)
    for agent_cls in BUILTIN_AGENTS:
        k.register(agent_cls())

    # --- priority ordering (single worker -> strictly deterministic) ---------
    k.submit(Task(kind="record", payload={"label": "low"}, priority=0))
    k.submit(Task(kind="record", payload={"label": "critical"}, priority=30))
    k.submit(Task(kind="record", payload={"label": "normal"}, priority=10))

    # --- a dependency chain: c waits for b waits for a -----------------------
    a = k.submit(Task(kind="calc", payload={"op": "+", "a": 1, "b": 1}))
    b = k.submit(Task(kind="calc", payload={"op": "*", "a": 10, "b": 10}, deps=[a]))
    c = k.submit(Task(kind="calc", payload={"op": "-", "a": 5, "b": 2}, deps=[b]))

    k.run()

    print("dispatch order (by priority):", recorder.order)
    print(f"chain results: a={k.scheduler.get(a).result}, "
          f"b={k.scheduler.get(b).result}, c={k.scheduler.get(c).result}")
    print("chain completed in dependency order:",
          all(k.scheduler.get(t).status.value == "completed" for t in (a, b, c)))

    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
