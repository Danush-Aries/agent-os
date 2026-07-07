"""Tests for the Agent OS kernel, scheduler, memory, and built-in agents."""

from __future__ import annotations

import pytest

from agentos import Agent, Context, Kernel, Task, TaskStatus
from agentos.agents import BUILTIN_AGENTS
from agentos.cli import main


def boot() -> Kernel:
    k = Kernel()
    for agent_cls in BUILTIN_AGENTS:
        k.register(agent_cls())
    return k


# --- basic dispatch ----------------------------------------------------------


def test_echo_runs_and_stores_result():
    k = boot()
    tid = k.submit(Task(kind="echo", payload={"message": "hi"}))
    report = k.run()
    assert report.ok
    assert k.scheduler.get(tid).status == TaskStatus.COMPLETED
    assert k.scheduler.get(tid).result == "hi"
    assert k.memory.get(f"task:{tid}:result") == "hi"


def test_calc_computes():
    k = boot()
    tid = k.submit(Task(kind="calc", payload={"op": "*", "a": 6, "b": 7}))
    k.run()
    assert k.scheduler.get(tid).result == 42


def test_unhandled_kind_fails_gracefully():
    k = boot()
    tid = k.submit(Task(kind="does-not-exist"))
    report = k.run()
    assert not report.ok
    assert report.unhandled == 1
    assert k.scheduler.get(tid).status == TaskStatus.FAILED
    assert "no agent handles" in k.scheduler.get(tid).error


def test_handler_exception_is_isolated():
    k = boot()
    bad = k.submit(Task(kind="calc", payload={"op": "??", "a": 1, "b": 2}))
    good = k.submit(Task(kind="echo", payload={"message": "still runs"}))
    report = k.run()
    assert k.scheduler.get(bad).status == TaskStatus.FAILED
    assert k.scheduler.get(good).status == TaskStatus.COMPLETED  # one failure != OS crash
    assert report.failed == 1 and report.completed == 1


# --- scheduling: priority + dependencies -------------------------------------


def test_priority_orders_runnable_tasks():
    order: list[str] = []

    class Recorder(Agent):
        name = "rec"
        handles = {"rec"}

        def handle(self, task: Task, ctx: Context):
            order.append(task.payload["label"])
            return task.payload["label"]

    k = Kernel()
    k.register(Recorder())
    k.submit(Task(kind="rec", payload={"label": "low"}, priority=0))
    k.submit(Task(kind="rec", payload={"label": "high"}, priority=10))
    k.submit(Task(kind="rec", payload={"label": "mid"}, priority=5))
    k.run()
    assert order == ["high", "mid", "low"]


def test_dependencies_run_after_their_deps():
    k = boot()
    a = k.submit(Task(kind="calc", payload={"op": "+", "a": 1, "b": 1}))
    b = k.submit(Task(kind="calc", payload={"op": "+", "a": 10, "b": 10}, deps=[a]))
    k.run()
    # both complete, and b only becomes runnable after a
    assert k.scheduler.get(a).status == TaskStatus.COMPLETED
    assert k.scheduler.get(b).status == TaskStatus.COMPLETED


def test_dependent_task_blocked_when_dep_fails():
    k = boot()
    bad = k.submit(Task(kind="calc", payload={"op": "??", "a": 1, "b": 1}))
    dependent = k.submit(Task(kind="echo", payload={"message": "x"}, deps=[bad]))
    report = k.run()
    assert k.scheduler.get(bad).status == TaskStatus.FAILED
    assert k.scheduler.get(dependent).status == TaskStatus.BLOCKED
    assert report.blocked == 1


# --- spawning: an agent decomposes work at runtime ---------------------------


def test_pipeline_spawns_children_and_reduces():
    k = boot()
    k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    report = k.run()
    assert report.ok
    reduce_tasks = [t for t in k.ps() if t.kind == "_sum_reduce"]
    assert len(reduce_tasks) == 1
    assert reduce_tasks[0].result == 10  # 1+2+3+4


# --- memory ------------------------------------------------------------------


def test_blackboard_is_shared_across_agents():
    class Writer(Agent):
        name = "writer"
        handles = {"write"}

        def handle(self, task, ctx):
            ctx.memory.put("shared", task.payload["v"])
            return "ok"

    class Reader(Agent):
        name = "reader"
        handles = {"read"}

        def handle(self, task, ctx):
            return ctx.memory.get("shared")

    k = Kernel()
    k.register(Writer()).register(Reader())
    w = k.submit(Task(kind="write", payload={"v": 99}))
    r = k.submit(Task(kind="read", deps=[w]))
    k.run()
    assert k.scheduler.get(r).result == 99


def test_duplicate_agent_name_rejected():
    k = Kernel()
    k.register(BUILTIN_AGENTS[0]())
    with pytest.raises(ValueError):
        k.register(BUILTIN_AGENTS[0]())


# --- CLI ---------------------------------------------------------------------


def test_cli_demo_exits_zero(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "completed" in out


def test_cli_run_from_file(tmp_path, capsys):
    f = tmp_path / "tasks.json"
    f.write_text(
        '[{"kind": "calc", "payload": {"op": "+", "a": 2, "b": 3}},'
        ' {"kind": "echo", "payload": {"message": "done"}, "deps": [0]}]',
        encoding="utf-8",
    )
    assert main(["run", str(f)]) == 0
