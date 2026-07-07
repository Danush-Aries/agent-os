"""Regression tests for concurrent execution and LLM observability.

These guard two fixes: (1) the scheduler is thread-safe, so worker threads
spawning tasks while the main loop scans for work never raise
"dictionary changed size during iteration" and report counts stay accurate;
(2) LLM calls are recorded with OpenTelemetry GenAI attributes + metrics.
"""

from __future__ import annotations

import time

from agentos import Agent, Kernel, Task, default_kernel
from agentos.llm import MockLLM
from agentos.tools import Tool


class _SlowSpawner(Agent):
    name = "slowspawner"
    handles = {"slow"}

    def handle(self, task, ctx):
        # Spawn while the main loop is actively scanning — the race window.
        for i in range(20):
            ctx.spawn("noop", {"i": i})
            time.sleep(0.0005)
        return "done"


class _Noop(Agent):
    name = "noop"
    handles = {"noop"}

    def handle(self, task, ctx):
        return task.payload.get("i")


def test_concurrent_spawning_is_thread_safe():
    k = Kernel(max_workers=16)
    k.register(_SlowSpawner()).register(_Noop())
    for _ in range(6):
        k.submit(Task(kind="slow"))
    report = k.run()  # must not raise
    assert report.failed == 0
    completed = sum(1 for t in k.ps() if t.status.value == "completed")
    # 6 spawners + 6*20 noops = 126, and report.completed must match exactly
    assert completed == 126
    assert report.completed == 126
    k.close()


def test_concurrent_fanout_results_correct():
    k = default_kernel(max_workers=12)
    for i in range(30):
        k.submit(Task(kind="sum_pipeline", payload={"numbers": list(range(i, i + 5))}))
    report = k.run()
    reduces = [t for t in k.ps() if t.kind == "_sum_reduce"]
    assert len(reduces) == 30
    assert all(t.status.value == "completed" for t in reduces)
    # each reduce sums its 5 consecutive numbers
    for i, t in enumerate(sorted(reduces, key=lambda x: x.id)):
        assert isinstance(t.result, int)
    assert report.ok
    k.close()


def test_llm_calls_are_traced_and_metered():
    k = default_kernel(llm=MockLLM())
    k.tools.register(Tool(
        name="calc", description="add",
        input_schema={"type": "object",
                      "properties": {"a": {"type": "number"}, "op": {"type": "string"},
                                     "b": {"type": "number"}},
                      "required": ["a", "op", "b"]},
        func=lambda a, op, b: a + b,
    ))
    tid = k.submit(Task(kind="llm", payload={"prompt": "compute 2 + 3"}))
    k.run()
    metrics = k.metrics()
    assert metrics.get("llm.calls", 0) >= 1
    assert metrics.get("llm.input_tokens", 0) > 0
    events = [e for e in k.trace(tid)["events"] if e["kind"] == "llm.call"]
    assert events
    attrs = events[0]["attrs"]
    assert "gen_ai.request.model" in attrs
    assert "gen_ai.usage.input_tokens" in attrs
    assert "gen_ai.system" in attrs
    k.close()
