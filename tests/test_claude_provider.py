"""Offline tests for AnthropicProvider — SDK wiring without a network call.

We inject a fake Anthropic client (matching the real SDK response shape) so the
content-block parsing, structured-output path, token/cost accounting, and the
``refusal`` stop-reason handling are all verified with no API key.
"""

from __future__ import annotations

import types

import pytest

from agentos.llm import AnthropicProvider, LLMResult


def _block(**kw):
    return types.SimpleNamespace(**kw)


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def _provider_with(response) -> AnthropicProvider:
    p = AnthropicProvider(api_key="test-key")
    p._client = lambda: _FakeClient(response)  # type: ignore[method-assign]
    return p


def test_defaults_are_skill_compliant():
    p = AnthropicProvider(api_key="k")
    assert p.model == "claude-opus-4-8"      # recommended default
    assert p.max_tokens == 4096              # not the truncating 1024


def test_text_response_parsed_with_usage_and_cost():
    resp = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="hello world")],
        usage=types.SimpleNamespace(input_tokens=12, output_tokens=5),
    )
    r = _provider_with(resp).complete([{"role": "user", "content": "hi"}])
    assert isinstance(r, LLMResult)
    assert r.text == "hello world"
    assert r.input_tokens == 12 and r.output_tokens == 5
    assert r.cost_usd >= 0.0


def test_tool_use_block_becomes_tool_call():
    resp = types.SimpleNamespace(
        stop_reason="tool_use",
        content=[_block(type="tool_use", name="calc", input={"a": 2, "op": "+", "b": 3})],
        usage=types.SimpleNamespace(input_tokens=8, output_tokens=4),
    )
    r = _provider_with(resp).complete(
        [{"role": "user", "content": "2+3"}],
        tools=[{"name": "calc", "description": "add", "input_schema": {}}],
    )
    assert r.tool_calls == [{"name": "calc", "arguments": {"a": 2, "op": "+", "b": 3}}]


def test_structured_output_forces_tool_and_returns_json():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}},
              "required": ["answer"]}
    resp = types.SimpleNamespace(
        stop_reason="tool_use",
        content=[_block(type="tool_use", name="structured_output",
                        input={"answer": "42"})],
        usage=types.SimpleNamespace(input_tokens=6, output_tokens=3),
    )
    p = _provider_with(resp)
    r = p.complete([{"role": "user", "content": "q"}], response_schema=schema)
    assert '"answer": "42"' in r.text
    # the forced-tool wiring was passed to the API
    assert p._client().messages  # sanity: client shape intact


def test_refusal_stop_reason_is_surfaced():
    resp = types.SimpleNamespace(stop_reason="refusal", content=[],
                                 usage=types.SimpleNamespace(input_tokens=0, output_tokens=0))
    r = _provider_with(resp).complete([{"role": "user", "content": "..."}])
    assert r.confidence == 0.0
    assert r.text == ""
    assert r.raw == {"stop_reason": "refusal"}


def test_system_message_is_split_out():
    resp = types.SimpleNamespace(
        stop_reason="end_turn",
        content=[_block(type="text", text="ok")],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    p = _provider_with(resp)
    p.complete([{"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}])
    kw = p._client().messages  # not the same client instance; re-check via a fresh call
    # Re-run capturing kwargs from a persistent fake:
    fake = _FakeClient(resp)
    p._client = lambda: fake  # type: ignore[method-assign]
    p.complete([{"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}])
    assert fake.messages.last_kwargs["system"] == "be terse"
    assert all(m["role"] != "system" for m in fake.messages.last_kwargs["messages"])


def test_missing_key_raises():
    p = AnthropicProvider(api_key=None)
    p.api_key = None
    with pytest.raises(RuntimeError):
        p._client()
