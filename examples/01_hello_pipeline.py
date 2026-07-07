"""01 - Hello, pipeline.

The smallest useful Agent OS program: boot the built-in agents, submit a
fan-out/reduce ``sum_pipeline`` plus a leaf ``echo`` task, run the kernel, and
read the results back off the process table.

    uv run python examples/01_hello_pipeline.py
"""

from __future__ import annotations

from agentos import Kernel, Task
from agentos.agents import BUILTIN_AGENTS


def main() -> int:
    k = Kernel()
    for agent_cls in BUILTIN_AGENTS:
        k.register(agent_cls())

    pipeline = k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    hello = k.submit(Task(kind="echo", payload={"message": "Agent OS is booted."}))

    report = k.run()

    print(f"run: {report.completed} completed, {report.failed} failed")
    print(f"echo[{hello}]      -> {k.scheduler.get(hello).result!r}")
    # sum_pipeline spawns one child per number, then a _sum_reduce that sums them.
    reduce = next(t for t in k.ps() if t.kind == "_sum_reduce")
    print(f"pipeline[{pipeline}] spawned {len(k.ps()) - 2} tasks")
    print(f"reduce[{reduce.id}]    -> {reduce.result}   (expected 10)")

    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
