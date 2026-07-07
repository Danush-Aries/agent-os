"""05 - Write your own agent, end to end.

An agent is a class with a ``name``, the task ``kinds`` it ``handles``, and a
``handle(task, ctx)`` method. Register it, submit tasks of its kind, run.
This one also uses the shared blackboard and spawns a follow-up task.

    uv run python examples/05_custom_agent.py
"""

from __future__ import annotations

from agentos import Agent, Context, Kernel, Task


class ReverseAgent(Agent):
    name = "reverser"
    handles = {"reverse"}

    def handle(self, task: Task, ctx: Context):
        text = task.payload["text"]
        reversed_text = text[::-1]
        ctx.memory.put("last_reversed", reversed_text)   # shared memory
        # spawn a follow-up leaf task that shouts the result
        ctx.spawn("shout", {"text": reversed_text})
        return reversed_text


class ShoutAgent(Agent):
    name = "shouter"
    handles = {"shout"}

    def handle(self, task: Task, ctx: Context):
        return task.payload["text"].upper() + "!"


def main() -> int:
    k = Kernel()
    k.register(ReverseAgent())
    k.register(ShoutAgent())

    tid = k.submit(Task(kind="reverse", payload={"text": "agent os"}))
    k.run()

    print("reversed:", k.scheduler.get(tid).result)
    print("blackboard[last_reversed]:", k.memory.get("last_reversed"))
    shout = next(t for t in k.ps() if t.kind == "shout")
    print("spawned shout ->", shout.result)

    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
