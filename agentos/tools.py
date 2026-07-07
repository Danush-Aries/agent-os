"""Tools: typed, validated actions agents can invoke.

Each tool carries a JSON-Schema for its arguments, validated (with a small
stdlib validator — no external dependency) before the function runs. Tools can
be flagged ``requires_approval`` (routed through the HITL gate by the kernel)
and given a ``rate_limit`` (max calls per window). Results are wrapped so a
failing tool reports ``is_error`` rather than crashing the caller.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque


@dataclass
class ToolResult:
    content: Any
    is_error: bool = False
    error: str | None = None


class ToolError(Exception):
    pass


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-Schema validator: type, required, properties, enum, items.

    Returns a list of human-readable error strings (empty == valid). Supports
    the subset used for tool arguments; enough to catch wrong/missing fields.
    """
    errors: list[str] = []
    t = schema.get("type")
    types = {
        "object": dict, "array": list, "string": str,
        "number": (int, float), "integer": int, "boolean": bool, "null": type(None),
    }
    if t in types:
        # bool is a subclass of int in Python; don't let True satisfy integer/number.
        bool_as_number = isinstance(value, bool) and t in ("integer", "number")
        if bool_as_number or not isinstance(value, types[t]):
            got = "boolean" if isinstance(value, bool) else type(value).__name__
            errors.append(f"{path}: expected {t}, got {got}")
            return errors
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    if t == "object":
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property '{req}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                errors.extend(validate_schema(value[key], subschema, f"{path}.{key}"))
    if t == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[..., Any]
    requires_approval: bool = False
    rate_limit: int | None = None          # max calls per window
    rate_window_s: float = 60.0
    _calls: Deque[float] = field(default_factory=deque, repr=False)

    def to_spec(self) -> dict:
        """Anthropic/OpenAI-style tool spec for handing to an LLM."""
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._clock = clock or time.time

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, input_schema: dict, **kw):
        """Decorator form: @registry.tool('add', '...', schema)."""
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name=name, description=description,
                               input_schema=input_schema, func=fn, **kw))
            return fn
        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict]:
        return [t.to_spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def _rate_ok(self, tool: Tool) -> bool:
        if not tool.rate_limit:
            return True
        now = self._clock()
        while tool._calls and now - tool._calls[0] > tool.rate_window_s:
            tool._calls.popleft()
        if len(tool._calls) >= tool.rate_limit:
            return False
        tool._calls.append(now)
        return True

    def call(self, name: str, args: dict) -> ToolResult:
        """Validate args, enforce rate limit, run the tool, wrap errors."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(content=None, is_error=True, error=f"unknown tool '{name}'")
        errs = validate_schema(args, tool.input_schema)
        if errs:
            return ToolResult(content=None, is_error=True, error="; ".join(errs))
        if not self._rate_ok(tool):
            return ToolResult(content=None, is_error=True,
                              error=f"rate limit exceeded for '{name}'")
        try:
            return ToolResult(content=tool.func(**args))
        except Exception as exc:  # tools must never crash the agent
            return ToolResult(content=None, is_error=True,
                              error=f"{type(exc).__name__}: {exc}")
