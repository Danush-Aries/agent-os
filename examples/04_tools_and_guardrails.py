"""04 - Tools and guardrails.

* Tools are typed, schema-validated actions an agent invokes via
  ``ctx.call_tool(name, args)``. The registry validates arguments before the
  function runs and wraps failures so a bad tool never crashes the agent.
* A Guardrail scans task payloads (input) and string results (output): it
  *blocks* secrets / prompt-injection and *redacts* PII.

    uv run python examples/04_tools_and_guardrails.py
"""

from __future__ import annotations

from agentos import Agent, Context, Kernel, Task
from agentos.guardrails import Guardrail
from agentos.tools import Tool, ToolRegistry


class WordCountAgent(Agent):
    """Calls the ``wordcount`` tool instead of doing the work itself."""

    name = "counter"
    handles = {"count_words"}

    def handle(self, task: Task, ctx: Context):
        result = ctx.call_tool("wordcount", {"text": task.payload["text"]})
        if result.is_error:
            raise RuntimeError(result.error)
        return result.content


def main() -> int:
    # --- a registry with one schema-validated tool ---------------------------
    tools = ToolRegistry()
    tools.register(Tool(
        name="wordcount",
        description="Count the words in a string.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        func=lambda text: len(text.split()),
    ))

    k = Kernel(tools=tools, guardrail=Guardrail())
    from agentos.agents import EchoAgent
    k.register(WordCountAgent())
    k.register(EchoAgent())

    # 1) a tool call through the agent
    count = k.submit(Task(kind="count_words", payload={"text": "agent os counts these words"}))

    # 2) output redaction: an email in a returned string is masked by the guardrail
    redacted = k.submit(Task(kind="echo", payload={"message": "reach me at alice@example.com"}))

    # 3) input block: a secret in the payload never reaches the agent
    blocked = k.submit(Task(kind="echo", payload={"message": "key AKIAIOSFODNN7EXAMPLE leaked"}))

    k.run()

    ct, rt, bt = (k.scheduler.get(t) for t in (count, redacted, blocked))
    print(f"tool call    -> {ct.result} words (status={ct.status.value})")
    print(f"redacted out -> {rt.result!r}")
    print(f"blocked in   -> status={bt.status.value} error={bt.error!r}")

    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
