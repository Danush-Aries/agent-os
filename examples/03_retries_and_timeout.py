"""03 - Retries and timeouts.

Reliability knobs on a Task:

* ``max_retries`` — the kernel re-invokes a failing handler with exponential
  backoff + jitter. A handler that fails once then succeeds recovers on its own.
* ``timeout_s`` — a handler that runs too long is abandoned and (with no
  retries left) fails with a TimeoutError, without hanging the whole run.

    uv run python examples/03_retries_and_timeout.py
"""

from __future__ import annotations

import time

from agentos import Agent, Context, Kernel, Task


class FlakyAgent(Agent):
    """Fails on its first attempt, succeeds on the second."""

    name = "flaky"
    handles = {"flaky"}

    def __init__(self) -> None:
        self.calls = 0

    def handle(self, task: Task, ctx: Context):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient failure (attempt 1)")
        return f"succeeded on attempt {self.calls}"


class SlowAgent(Agent):
    """Sleeps far longer than the task's timeout allows."""

    name = "slow"
    handles = {"slow"}

    def handle(self, task: Task, ctx: Context):
        time.sleep(2.0)
        return "should never be returned"


def main() -> int:
    k = Kernel()
    k.register(FlakyAgent())
    k.register(SlowAgent())

    flaky = k.submit(Task(kind="flaky", max_retries=2))
    timed = k.submit(Task(kind="slow", timeout_s=0.1))

    k.run()

    ft, tt = k.scheduler.get(flaky), k.scheduler.get(timed)
    print(f"flaky[{flaky}] status={ft.status.value} attempts={ft.attempts} result={ft.result!r}")
    print(f"slow[{timed}]  status={tt.status.value} attempts={tt.attempts} error={tt.error!r}")

    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
