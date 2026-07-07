"""A few working agents that ship with Agent OS — usable and illustrative.

They demonstrate the three things the OS is built to do: run a leaf task
(EchoAgent), compute a result other tasks depend on (CalcAgent), and decompose
a job into a dependent sub-pipeline (PipelineAgent).
"""

from __future__ import annotations

import operator
from typing import Any

from ..agent import Agent, Context
from ..task import Task

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}


class EchoAgent(Agent):
    name = "echo"
    handles = {"echo"}

    def handle(self, task: Task, ctx: Context) -> Any:
        return task.payload.get("message", "")


class CalcAgent(Agent):
    """Handles ``{"kind": "calc", "payload": {"op": "+", "a": 2, "b": 3}}``."""

    name = "calc"
    handles = {"calc"}

    def handle(self, task: Task, ctx: Context) -> Any:
        op = task.payload["op"]
        if op not in _OPS:
            raise ValueError(f"unknown op '{op}'")
        return _OPS[op](task.payload["a"], task.payload["b"])


class PipelineAgent(Agent):
    """Fan-out then reduce.

    Given ``{"kind": "sum_pipeline", "payload": {"numbers": [1,2,3,4]}}`` it
    spawns one ``calc`` task per adjacent pair-add... but to keep it simple and
    deterministic it spawns per-number 'echo' children and sums their results
    once they complete. Shows spawn + deps + reading child results.
    """

    name = "pipeline"
    handles = {"sum_pipeline", "_sum_reduce"}

    def handle(self, task: Task, ctx: Context) -> Any:
        if task.kind == "sum_pipeline":
            numbers = task.payload.get("numbers", [])
            child_ids = [
                ctx.spawn("calc", {"op": "+", "a": n, "b": 0}, priority=1)
                for n in numbers
            ]
            # Reduce step depends on all children; runs after they complete.
            ctx.spawn("_sum_reduce", {"children": child_ids}, deps=child_ids)
            return {"spawned": child_ids}
        # _sum_reduce
        child_ids = task.payload["children"]
        return sum(ctx.result_of(cid) for cid in child_ids)


BUILTIN_AGENTS = [EchoAgent, CalcAgent, PipelineAgent]
