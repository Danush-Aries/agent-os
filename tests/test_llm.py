"""Offline tests for agentos.llm — MockLLM and FallbackLLM only.

No network, no optional extras. Real providers (Ollama/Anthropic) are only
smoke-checked for lazy-import safety (constructing them must not import their
transport).
"""

from __future__ import annotations

import json

import pytest

from agentos.llm import (
    FallbackLLM,
    LLMProvider,
    LLMResult,
    MockLLM,
    cost_of,
    get_provider,
)


CALC_TOOL = {
    "name": "calculator",
    "description": "Evaluate a simple binary arithmetic expression.",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "op": {"type": "string"},
            "b": {"type": "number"},
        },
        "required": ["a", "op", "b"],
    },
}

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string"},
        "active": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "role": {"type": "string", "enum": ["admin", "user"]},
    },
    "required": ["name", "age", "active", "tags", "role"],
}


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
def test_structured_output_is_schema_valid_json():
    llm = MockLLM()
    res = llm.complete(_user("give me a person"), response_schema=PERSON_SCHEMA)
    obj = json.loads(res.text)  # must be valid JSON

    # every required field present + type-appropriate
    for req in PERSON_SCHEMA["required"]:
        assert req in obj
    assert isinstance(obj["name"], str)
    assert isinstance(obj["age"], int)
    assert isinstance(obj["active"], bool)
    assert isinstance(obj["tags"], list)
    assert obj["role"] in ("admin", "user")  # enum honored
    assert res.tool_calls == []


def test_structured_output_is_deterministic():
    llm = MockLLM()
    a = llm.complete(_user("x"), response_schema=PERSON_SCHEMA).text
    b = llm.complete(_user("y"), response_schema=PERSON_SCHEMA).text
    assert a == b


# --------------------------------------------------------------------------- #
# Tool calling
# --------------------------------------------------------------------------- #
def test_tool_call_on_explicit_trigger():
    llm = MockLLM()
    res = llm.complete(_user("please use tool calculator now"), tools=[CALC_TOOL])
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0]["name"] == "calculator"
    assert isinstance(res.tool_calls[0]["arguments"], dict)


def test_tool_call_on_arithmetic_expression():
    llm = MockLLM()
    res = llm.complete(_user("what is 2 + 3?"), tools=[CALC_TOOL])
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert call["name"] == "calculator"
    assert call["arguments"] == {"a": 2, "op": "+", "b": 3}


def test_no_tool_call_without_trigger():
    llm = MockLLM()
    res = llm.complete(_user("hello there"), tools=[CALC_TOOL])
    assert res.tool_calls == []
    assert res.text.startswith("echo:")


def test_tool_calling_loop_simulation():
    """A tiny agent loop: model asks for a tool, we run it, model echoes."""
    llm = MockLLM()
    messages = _user("compute 4 * 5")
    res = llm.complete(messages, tools=[CALC_TOOL])
    assert res.tool_calls, "expected the model to request a tool"

    call = res.tool_calls[0]
    a, op, b = call["arguments"]["a"], call["arguments"]["op"], call["arguments"]["b"]
    tool_result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0}[op]
    assert tool_result == 20

    messages.append({"role": "assistant", "content": ""})
    messages.append({"role": "user", "content": f"result is {tool_result}"})
    final = llm.complete(messages)  # no tools -> plain echo
    assert final.tool_calls == []
    assert "20" in final.text


# --------------------------------------------------------------------------- #
# Plain echo
# --------------------------------------------------------------------------- #
def test_plain_echo():
    llm = MockLLM()
    res = llm.complete(_user("hello world"))
    assert res.text == "echo: hello world"
    assert res.tool_calls == []


# --------------------------------------------------------------------------- #
# Tokens / cost
# --------------------------------------------------------------------------- #
def test_tokens_and_cost_positive():
    llm = MockLLM(model="claude-fable-5")  # a priced model
    res = llm.complete(_user("one two three four five"))
    assert res.input_tokens > 0
    assert res.output_tokens > 0
    assert res.cost_usd > 0


def test_mock_model_is_free_but_tokens_counted():
    llm = MockLLM()  # default "mock" model -> free
    res = llm.complete(_user("a b c"))
    assert res.model == "mock"
    assert res.input_tokens > 0
    assert res.cost_usd == 0.0


def test_cost_of_helper():
    assert cost_of("claude-fable-5", 1_000_000, 0) == pytest.approx(3.0)
    assert cost_of("claude-fable-5", 0, 1_000_000) == pytest.approx(15.0)
    # unknown model falls back to default pricing row
    assert cost_of("totally-unknown", 1_000_000, 0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def test_stream_joins_to_full_text():
    llm = MockLLM()
    messages = _user("stream this sentence back please")
    full = llm.complete(messages).text
    streamed = "".join(llm.stream(messages))
    assert streamed == full
    assert full == "echo: stream this sentence back please"


def test_stream_yields_multiple_deltas():
    llm = MockLLM()
    deltas = list(llm.stream(_user("many words here now")))
    assert len(deltas) > 1


# --------------------------------------------------------------------------- #
# FallbackLLM
# --------------------------------------------------------------------------- #
class _Raising(LLMProvider):
    """A stub primary that always raises."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def complete(self, messages, tools=None, response_schema=None):
        raise self.exc

    def stream(self, messages, tools=None):
        raise self.exc
        yield  # pragma: no cover


class _RateLimitedResult(LLMProvider):
    """A stub primary that returns a rate-limited LLMResult (no raise)."""

    def complete(self, messages, tools=None, response_schema=None):
        return LLMResult(text="", model="primary", cost_usd=0.01,
                         raw={"rate_limited": True})

    def stream(self, messages, tools=None):
        yield "x"


def test_fallback_on_exception():
    primary = _Raising(RuntimeError("boom"))
    fb = FallbackLLM(primary, MockLLM())
    res = fb.complete(_user("hi"))
    assert res.text == "echo: hi"


def test_fallback_on_rate_limit_exception():
    primary = _Raising(Exception("HTTP 429 Too Many Requests"))
    fb = FallbackLLM(primary, MockLLM(model="claude-fable-5"))
    res = fb.complete(_user("one two three"))
    assert res.text.startswith("echo:")
    assert res.cost_usd > 0


def test_fallback_aggregates_cost_on_rate_limited_result():
    primary = _RateLimitedResult()
    fb = FallbackLLM(primary, MockLLM(model="claude-fable-5"))
    res = fb.complete(_user("one two three four"))
    # fallback cost + primary's wasted 0.01
    assert res.cost_usd > 0.01
    assert res.text.startswith("echo:")


def test_fallback_stream_falls_back():
    primary = _Raising(RuntimeError("nope"))
    fb = FallbackLLM(primary, MockLLM())
    out = "".join(fb.stream(_user("hello")))
    assert out == "echo: hello"


def test_fallback_uses_primary_when_healthy():
    fb = FallbackLLM(MockLLM(), _Raising(RuntimeError("should not be called")))
    res = fb.complete(_user("healthy"))
    assert res.text == "echo: healthy"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_get_provider_default_is_mock():
    assert isinstance(get_provider(), MockLLM)


def test_get_provider_by_name():
    assert isinstance(get_provider("mock"), MockLLM)


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("does-not-exist")


def test_real_providers_construct_without_transport_import():
    """Constructing Ollama/Anthropic must not require the [llm] extra."""
    from agentos.llm import AnthropicProvider, OllamaProvider

    OllamaProvider()  # no httpx import at construction
    AnthropicProvider(api_key="dummy")  # no anthropic import at construction
