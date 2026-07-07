"""Run an LLM agent backed by the real Claude API.

Set ANTHROPIC_API_KEY (and optionally AGENTOS_ANTHROPIC_MODEL) and run:

    ANTHROPIC_API_KEY=sk-ant-... uv run --extra llm python examples/06_claude_agent.py

Without a key this prints how to enable it and exits 0 (so it stays CI-safe).
The same LLMAgent runs against MockLLM/Ollama/Claude — only the provider swaps.
"""

from __future__ import annotations

import os

from agentos import Task, default_kernel
from agentos.llm import AnthropicProvider, MockLLM
from agentos.tools import Tool


def build_calc_tool() -> Tool:
    return Tool(
        name="calc",
        description="Add, subtract, or multiply two numbers.",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "op": {"type": "string", "enum": ["+", "-", "*"]},
                "b": {"type": "number"},
            },
            "required": ["a", "op", "b"],
        },
        func=lambda a, op, b: {"+" : a + b, "-": a - b, "*": a * b}[op],
    )


def main() -> int:
    if os.environ.get("ANTHROPIC_API_KEY"):
        provider = AnthropicProvider()  # real Claude (model from AGENTOS_ANTHROPIC_MODEL)
        print(f"Using Claude: {provider.model}")
    else:
        provider = MockLLM()
        print("ANTHROPIC_API_KEY not set — using MockLLM (offline).")
        print("Enable real Claude: export ANTHROPIC_API_KEY=sk-ant-... "
              "and install the [llm] extra.")

    k = default_kernel(llm=provider)
    k.tools.register(build_calc_tool())

    tid = k.submit(Task(kind="llm", payload={
        "prompt": "What is 21 * 2? Use the calc tool, then state the answer.",
        "system": "You are a precise calculator assistant.",
    }))
    k.run()
    result = k.scheduler.get(tid).result
    print("\n--- agent result ---")
    print("text:", result.get("text"))
    print("tool calls:", result.get("tool_calls"))
    print("cost USD:", result.get("cost_usd"))
    k.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
