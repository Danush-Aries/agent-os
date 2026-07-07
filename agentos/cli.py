"""Command line for Agent OS.

    agent-os                     # boot everything + open the live dashboard
    agent-os up                  # same as bare invocation
    agent-os demo                # run the built-in pipeline demo
    agent-os run tasks.json      # boot built-in agents, run tasks from a file
    agent-os agents              # list registered built-in agents
    agent-os ps                  # run the demo, print the process table
    agent-os metrics             # run the demo, print kernel metrics
    agent-os trace <task_id>     # run the demo, pretty-print a task's trace
    agent-os serve               # start the REST API without opening a browser
    agent-os config              # show / set config (e.g. LLM provider)

A tasks file is JSON: a list of {"kind","payload","priority","deps"} objects,
where deps reference other tasks by their 0-based index in the list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .agents import BUILTIN_AGENTS
from .kernel import Kernel
from .task import Task, TaskStatus

CONFIG_PATH = Path(os.environ.get("AGENTOS_CONFIG",
                                  str(Path.home() / ".agentos" / "config.json")))


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _apply_config_env() -> None:
    """Load persisted config into the environment (env still wins if set)."""
    for key, val in _load_config().items():
        os.environ.setdefault(key, str(val))


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


def _serve(open_browser: bool) -> int:
    try:
        from .server import run
    except ImportError:
        print("The dashboard needs the [api] extra. Install with:\n"
              "    uv tool install \"agent-os[api]\"", file=sys.stderr)
        return 1
    from .llm import auto_provider
    provider = type(auto_provider()).__name__
    print(f"Agent OS — booting all agents + tools + LLM ({provider})")
    run(open_browser=open_browser)
    return 0


def cmd_up(_: argparse.Namespace) -> int:
    """Boot everything and open the dashboard — the default action."""
    return _serve(open_browser=True)


def cmd_serve(_: argparse.Namespace) -> int:
    return _serve(open_browser=False)


def cmd_config(args: argparse.Namespace) -> int:
    cfg = _load_config()
    if args.set:
        for pair in args.set:
            if "=" not in pair:
                print(f"bad --set '{pair}', expected KEY=VALUE", file=sys.stderr)
                return 2
            key, _, val = pair.partition("=")
            cfg[key.strip()] = val.strip()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"wrote {CONFIG_PATH}")
    if not cfg:
        print(f"no config yet ({CONFIG_PATH}).")
        print("example: agent-os config --set AGENTOS_LLM=claude-cli")
    else:
        print(f"# {CONFIG_PATH}")
        for k, v in cfg.items():
            print(f"{k}={v}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-os",
        description="An operating system for cooperating agents. "
                    "Run with no arguments to boot everything and open the dashboard.",
    )
    p.add_argument("--version", action="version", version=f"agent-os {__version__}")
    sub = p.add_subparsers(dest="command")  # optional: bare invocation => up
    sub.add_parser("up", help="boot everything and open the dashboard (default)").set_defaults(func=cmd_up)
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
    sub.add_parser("serve", help="start the REST API without opening a browser").set_defaults(func=cmd_serve)
    cfgp = sub.add_parser("config", help="show or set persisted config")
    cfgp.add_argument("--set", action="append", metavar="KEY=VALUE",
                      help="persist a config value (repeatable)")
    cfgp.set_defaults(func=cmd_config)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "func", None) is None:
        return cmd_up(args)  # bare `agent-os` boots the dashboard
    return args.func(args)


def console_main() -> int:
    """Real CLI entry point. Loads persisted config into the env first.

    Kept separate from ``main`` so the test suite (which calls ``main`` directly)
    is never affected by a user's ~/.agentos/config.json.
    """
    _apply_config_env()  # persisted config feeds env (real env still wins)
    return main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(console_main())
