"""Command line for Agent OS.

    agentos demo                 # run the built-in pipeline demo
    agentos run tasks.json       # boot built-in agents, run tasks from a file
    agentos agents               # list registered built-in agents
    agentos ps                   # run the demo, print the process table
    agentos metrics              # run the demo, print kernel metrics
    agentos trace <task_id>      # run the demo, pretty-print a task's trace
    agentos serve                # start the REST API (needs the [api] extra)

A tasks file is JSON: a list of {"kind","payload","priority","deps"} objects,
where deps reference other tasks by their 0-based index in the list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agents import BUILTIN_AGENTS
from .kernel import Kernel
from .task import Task, TaskStatus


def _boot() -> Kernel:
    k = Kernel()
    for agent_cls in BUILTIN_AGENTS:
        k.register(agent_cls())
    return k


def _print_ps(kernel: Kernel) -> None:
    print(f"{'ID':>3}  {'KIND':<14} {'STATUS':<10} {'OWNER':<10} RESULT")
    for t in kernel.ps():
        result = "" if t.result is None else json.dumps(t.result)
        detail = t.error or result
        print(f"{t.id:>3}  {t.kind:<14} {t.status.value:<10} {(t.owner or ''):<10} {detail}")


def _run_demo(k: Kernel) -> None:
    """Submit the standard demo workload and run it to completion."""
    k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    k.submit(Task(kind="echo", payload={"message": "Agent OS is booted."}))
    k.run()


def _load_tasks(path: Path) -> list[Task]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, list):
        raise ValueError("tasks file must be a JSON list")
    tasks = [
        Task(
            kind=item["kind"],
            payload=item.get("payload", {}),
            priority=item.get("priority", 0),
        )
        for item in spec
    ]
    # Resolve index-based deps to real task ids now that all tasks have ids.
    for task, item in zip(tasks, spec):
        for dep_index in item.get("deps", []):
            task.deps.append(tasks[dep_index].id)
    return tasks


def cmd_demo(_: argparse.Namespace) -> int:
    k = _boot()
    k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
    k.submit(Task(kind="echo", payload={"message": "Agent OS is booted."}))
    report = k.run()
    _print_ps(k)
    print(f"\n{report.completed} completed, {report.failed} failed, {report.blocked} blocked")
    k.close()
    return 0 if report.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.tasks)
    if not path.exists():
        print(f"tasks file not found: {path}", file=sys.stderr)
        return 2
    k = _boot()
    for task in _load_tasks(path):
        k.submit(task)
    report = k.run()
    _print_ps(k)
    print(f"\n{report.completed} completed, {report.failed} failed, {report.blocked} blocked")
    k.close()
    return 0 if report.ok else 1


def cmd_agents(_: argparse.Namespace) -> int:
    for agent_cls in BUILTIN_AGENTS:
        a = agent_cls()
        print(f"{a.name:<12} handles: {', '.join(sorted(a.handles))}")
    return 0


def cmd_ps(_: argparse.Namespace) -> int:
    k = _boot()
    _run_demo(k)
    _print_ps(k)
    k.close()
    return 0


def cmd_metrics(_: argparse.Namespace) -> int:
    k = _boot()
    _run_demo(k)
    metrics = k.metrics()
    width = max((len(key) for key in metrics), default=0)
    for key, value in metrics.items():
        # print integers cleanly, keep floats readable
        shown = int(value) if isinstance(value, float) and value.is_integer() else value
        print(f"{key:<{width}}  {shown}")
    k.close()
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    k = _boot()
    _run_demo(k)
    trace = k.trace(args.task_id)
    if trace is None or (trace.get("span") is None and not trace.get("events")):
        print(f"no trace for task {args.task_id}", file=sys.stderr)
        k.close()
        return 2
    print(json.dumps(trace, indent=2, default=str))
    k.close()
    return 0


def cmd_serve(_: argparse.Namespace) -> int:
    try:
        from .server import run
    except ImportError:
        print("install the [api] extra: uv tool install agent-os[api]", file=sys.stderr)
        return 1
    run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentos", description="A tiny OS for cooperating agents.")
    p.add_argument("--version", action="version", version=f"agentos {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the built-in pipeline demo").set_defaults(func=cmd_demo)
    runp = sub.add_parser("run", help="run tasks from a JSON file")
    runp.add_argument("tasks")
    runp.set_defaults(func=cmd_run)
    sub.add_parser("agents", help="list built-in agents").set_defaults(func=cmd_agents)
    sub.add_parser("ps", help="run the demo and print the process table").set_defaults(func=cmd_ps)
    sub.add_parser("metrics", help="run the demo and print kernel metrics").set_defaults(func=cmd_metrics)
    tracep = sub.add_parser("trace", help="run the demo and print a task's trace")
    tracep.add_argument("task_id", type=int, help="id of the task to trace")
    tracep.set_defaults(func=cmd_trace)
    sub.add_parser("serve", help="start the REST API (needs the [api] extra)").set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
