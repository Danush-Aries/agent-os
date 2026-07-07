"""LLM provider abstraction for Agent OS.

The core package is dependency-free. This module keeps that promise: only the
standard library is imported at module top. Real providers (Ollama, Anthropic)
import their transport (``httpx`` / ``anthropic``) *lazily* inside methods, so
importing :mod:`agentos.llm` never requires the optional ``[llm]`` extra.

The default provider is :class:`MockLLM` — deterministic, offline, and useful
for exercising an agent's tool-calling loop in tests.

    from agentos.llm import get_provider

    llm = get_provider()            # MockLLM unless $AGENTOS_LLM says otherwise
    res = llm.complete([{"role": "user", "content": "hello"}])
    print(res.text, res.cost_usd)
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class LLMResult:
    """The outcome of a single completion.

    ``tool_calls`` is a list of ``{"name": str, "arguments": dict}`` entries.
    Token counts and ``cost_usd`` let a caller do budgeting/accounting.
    """

    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    confidence: float = 1.0
    raw: dict | None = None


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
# USD per 1,000,000 tokens. Keep this small and explicit; unknown models fall
# back to the "default" entry via cost_of().
PRICING: dict[str, dict] = {
    "claude-fable-5": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
    "mock": {"input": 0.0, "output": 0.0},
    "default": {"input": 1.00, "output": 3.00},
}


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    """Return the USD cost of ``in_tok`` input + ``out_tok`` output tokens.

    Falls back to the ``"default"`` pricing row for unknown models.
    """
    row = PRICING.get(model) or PRICING["default"]
    return (in_tok * row["input"] + out_tok * row["output"]) / 1_000_000


# --------------------------------------------------------------------------- #
# Provider interface
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    """Abstract interface every provider implements.

    ``complete`` returns a fully-formed :class:`LLMResult`. ``stream`` yields
    incremental text deltas (strings). Providers that cannot stream may fall
    back to yielding the single completed text.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        ...

    @abstractmethod
    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[str]:
        ...


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _last_user_message(messages: list[dict]) -> str:
    """Text content of the most recent user-role message ("" if none)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # Anthropic-style block content
            if isinstance(content, list):
                parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                return " ".join(parts)
    return ""


def _word_count(text: str) -> int:
    return max(1, len(text.split()))


def _stub_for_schema(schema: dict) -> object:
    """Build a schema-valid stub value from a (subset of) JSON Schema.

    Handles ``type`` of object/array/string/integer/number/boolean/null, the
    ``required``/``properties``/``items`` keywords, and ``enum``. Optional
    properties are omitted; required ones are always filled.
    """
    if not isinstance(schema, dict):
        return None

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    stype = schema.get("type")
    # ``type`` may be a list; pick the first non-null option.
    if isinstance(stype, list):
        stype = next((t for t in stype if t != "null"), stype[0] if stype else None)

    if stype == "object" or ("properties" in schema and stype is None):
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        out: dict = {}
        for name in required:
            out[name] = _stub_for_schema(props.get(name, {}))
        return out
    if stype == "array":
        item_schema = schema.get("items", {})
        return [_stub_for_schema(item_schema)] if item_schema else []
    if stype == "string":
        return schema.get("default", "stub")
    if stype == "integer":
        return schema.get("default", 0)
    if stype == "number":
        return schema.get("default", 0.0)
    if stype == "boolean":
        return schema.get("default", False)
    if stype == "null":
        return None
    # Unknown/unspecified — a string is the safest schema-agnostic stub.
    return schema.get("default", "stub")


_ARITH_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)")


def _match_tool_trigger(
    text: str, tools: list[dict]
) -> tuple[str, dict] | None:
    """Decide whether ``text`` should trigger one of ``tools``.

    Two triggers are supported (deterministically):

    1. An explicit ``use tool <name>`` directive.
    2. An arithmetic expression (``2 + 3``) when a calculator-like tool
       (name containing calc/math/add/arith) is available — arguments are
       parsed from the expression.
    """
    names = [t.get("name") or t.get("function", {}).get("name") for t in tools]
    names = [n for n in names if n]
    if not names:
        return None

    lowered = text.lower()

    # 1. Explicit "use tool <name>"
    m = re.search(r"use\s+tool\s+([A-Za-z0-9_\-]+)", lowered)
    if m:
        wanted = m.group(1)
        for n in names:
            if n.lower() == wanted:
                return n, {}
        # Directive present but name unknown — trigger the first tool anyway.
        return names[0], {}

    # 2. Arithmetic expression -> calculator-style tool
    arith = _ARITH_RE.search(text)
    if arith:
        calc = next(
            (
                n
                for n in names
                if any(k in n.lower() for k in ("calc", "math", "add", "arith"))
            ),
            None,
        )
        if calc:
            a, op, b = arith.groups()
            num = lambda s: int(s) if re.fullmatch(r"-?\d+", s) else float(s)
            return calc, {"a": num(a), "op": op, "b": num(b)}

    return None


# --------------------------------------------------------------------------- #
# MockLLM — deterministic, offline, the default
# --------------------------------------------------------------------------- #
class MockLLM(LLMProvider):
    """A deterministic, network-free provider for tests and local runs.

    Decision order in :meth:`complete`:

    1. ``response_schema`` present -> ``text`` is a schema-valid JSON string.
    2. ``tools`` present and the last user message triggers one -> emit a
       ``tool_calls`` entry (see :func:`_match_tool_trigger`).
    3. Otherwise -> echo a deterministic reply derived from the last user
       message.

    Token counts are word counts; cost is computed against the ``"mock"``
    pricing row (which is free, so ``cost_usd`` is 0 unless a caller overrides
    the model). Set ``model`` at construction to bill against another row.
    """

    def __init__(self, model: str = "mock") -> None:
        self.model = model

    # -- internal: compute the reply + tool_calls without token bookkeeping --
    def _decide(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        response_schema: dict | None,
    ) -> tuple[str, list[dict]]:
        user = _last_user_message(messages)

        if response_schema is not None:
            stub = _stub_for_schema(response_schema)
            return json.dumps(stub, sort_keys=True), []

        if tools:
            hit = _match_tool_trigger(user, tools)
            if hit is not None:
                name, args = hit
                return "", [{"name": name, "arguments": args}]

        return f"echo: {user}", []

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        text, tool_calls = self._decide(messages, tools, response_schema)

        in_tok = sum(
            _word_count(m.get("content", ""))
            for m in messages
            if isinstance(m.get("content"), str)
        )
        in_tok = max(1, in_tok)
        # Output tokens count the text plus a nominal token per tool call.
        out_tok = (_word_count(text) if text else 0) + len(tool_calls)
        out_tok = max(1, out_tok)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_tokens=0,
            model=self.model,
            cost_usd=cost_of(self.model, in_tok, out_tok),
            confidence=1.0,
            raw={"provider": "mock"},
        )

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[str]:
        text, _ = self._decide(messages, tools, None)
        words = text.split(" ")
        for i, w in enumerate(words):
            # Preserve spacing so "".join(stream) == text.
            yield w if i == 0 else " " + w


# --------------------------------------------------------------------------- #
# OllamaProvider — lazy httpx
# --------------------------------------------------------------------------- #
class OllamaProvider(LLMProvider):
    """Talk to a local Ollama server (`/api/chat`).

    ``httpx`` is imported lazily inside the methods, so importing this class
    never pulls in the optional extra. Configure via env:

    * ``OLLAMA_HOST``          (default ``http://localhost:11434``)
    * ``AGENTOS_OLLAMA_MODEL`` (default ``llama3``)
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        self.model = model or os.environ.get("AGENTOS_OLLAMA_MODEL", "llama3")
        self.host = (
            host
            or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        import httpx  # lazy

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if response_schema is not None:
            # Ollama accepts a JSON schema as `format` for structured output.
            payload["format"] = response_schema

        resp = httpx.post(
            f"{self.host}/api/chat", json=payload, timeout=120.0
        )
        resp.raise_for_status()
        data = resp.json()

        msg = data.get("message", {}) or {}
        text = msg.get("content", "") or ""
        tool_calls: list[dict] = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            tool_calls.append({"name": fn.get("name", ""), "arguments": args})

        in_tok = int(data.get("prompt_eval_count", 0) or 0)
        out_tok = int(data.get("eval_count", 0) or 0)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
            cost_usd=cost_of(self.model, in_tok, out_tok),
            raw=data,
        )

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[str]:
        import httpx  # lazy

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        with httpx.stream(
            "POST", f"{self.host}/api/chat", json=payload, timeout=120.0
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                delta = (chunk.get("message", {}) or {}).get("content", "")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break


# --------------------------------------------------------------------------- #
# AnthropicProvider — lazy anthropic / httpx
# --------------------------------------------------------------------------- #
class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider.

    Uses the official ``anthropic`` SDK, imported lazily inside the methods.
    Configure via:

    * ``ANTHROPIC_API_KEY``       (required at call time)
    * ``AGENTOS_ANTHROPIC_MODEL`` (default ``claude-opus-4-8``)
    * ``AGENTOS_ANTHROPIC_MAX_TOKENS`` (default ``4096``)

    Structured output is implemented by forcing a single-tool call whose input
    schema is ``response_schema``; the tool input is returned as JSON ``text``.
    A ``refusal`` stop reason (safety decline, HTTP 200 with empty content) is
    surfaced as ``confidence=0.0`` rather than an empty result.
    """

    _STRUCT_TOOL = "structured_output"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        # Default to Opus 4.8 (recommended general-purpose model). Override via
        # AGENTOS_ANTHROPIC_MODEL — "claude-fable-5" for the most capable model,
        # "claude-haiku-4-5" for cheap/fast subagents.
        self.model = model or os.environ.get(
            "AGENTOS_ANTHROPIC_MODEL", "claude-opus-4-8"
        )
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        # 1024 truncates real agent turns; 4096 finishes a thought while staying
        # under the non-streaming SDK timeout.
        self.max_tokens = int(os.environ.get("AGENTOS_ANTHROPIC_MAX_TOKENS", "4096"))

    def _client(self):
        # Check config before importing the SDK so a missing key is a clean
        # error even when the [llm] extra isn't installed.
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        import anthropic  # lazy; needs the [llm] extra
        return anthropic.Anthropic(api_key=self.api_key)

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
        system = None
        convo: list[dict] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                convo.append(m)
        return system, convo

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        client = self._client()
        system, convo = self._split_system(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": convo,
        }
        if system:
            kwargs["system"] = system

        if response_schema is not None:
            kwargs["tools"] = [
                {
                    "name": self._STRUCT_TOOL,
                    "description": "Return the answer using this schema.",
                    "input_schema": response_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": self._STRUCT_TOOL}
        elif tools:
            kwargs["tools"] = tools

        resp = client.messages.create(**kwargs)

        # Safety classifiers can decline with HTTP 200 + stop_reason "refusal"
        # (empty content, not billed pre-output). Surface it instead of
        # returning a silently-empty result. Always check before reading content.
        if getattr(resp, "stop_reason", None) == "refusal":
            return LLMResult(
                text="", tool_calls=[], input_tokens=0, output_tokens=0,
                model=self.model, cost_usd=0.0, confidence=0.0,
                raw={"stop_reason": "refusal"},
            )

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        structured_text: str | None = None
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                name = getattr(block, "name", "")
                args = getattr(block, "input", {}) or {}
                if response_schema is not None and name == self._STRUCT_TOOL:
                    structured_text = json.dumps(args, sort_keys=True)
                else:
                    tool_calls.append({"name": name, "arguments": args})

        text = structured_text if structured_text is not None else "".join(text_parts)

        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cache_tok = int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ) + int(getattr(usage, "cache_creation_input_tokens", 0) or 0)

        raw = None
        if hasattr(resp, "model_dump"):
            try:
                raw = resp.model_dump()
            except Exception:  # pragma: no cover - defensive
                raw = None

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_tokens=cache_tok,
            model=self.model,
            cost_usd=cost_of(self.model, in_tok, out_tok),
            raw=raw,
        )

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[str]:
        client = self._client()
        system, convo = self._split_system(messages)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": convo,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        with client.messages.stream(**kwargs) as stream:
            for delta in stream.text_stream:
                if delta:
                    yield delta


# --------------------------------------------------------------------------- #
# FallbackLLM — resilience wrapper
# --------------------------------------------------------------------------- #
class RateLimitError(Exception):
    """Raised (or signalled) to indicate the primary is rate-limited."""


def _is_rate_limited(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text or "too many requests" in text


class FallbackLLM(LLMProvider):
    """Try ``primary`` first; on failure or rate-limit, use ``fallback``.

    A "rate-limit signal" is either a raised exception whose type/message looks
    like a rate limit, or a returned :class:`LLMResult` whose ``raw`` contains
    ``{"rate_limited": True}``. Costs from both attempts are aggregated onto
    the returned result's ``cost_usd``.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self.primary = primary
        self.fallback = fallback

    @staticmethod
    def _result_is_rate_limited(res: LLMResult) -> bool:
        return bool(res.raw and res.raw.get("rate_limited"))

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_schema: dict | None = None,
    ) -> LLMResult:
        primary_cost = 0.0
        try:
            res = self.primary.complete(messages, tools, response_schema)
            if not self._result_is_rate_limited(res):
                return res
            primary_cost = res.cost_usd  # count the wasted attempt
        except Exception:  # noqa: BLE001 - any failure triggers fallback
            pass

        res = self.fallback.complete(messages, tools, response_schema)
        res.cost_usd += primary_cost
        return res

    def stream(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[str]:
        try:
            yield from self.primary.stream(messages, tools)
        except Exception:  # noqa: BLE001 - fall back on any streaming failure
            yield from self.fallback.stream(messages, tools)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
class ClaudeCliProvider(LLMProvider):
    """Use a Claude **Max/Pro subscription** via the ``claude`` CLI (Claude Code).

    Unlike :class:`AnthropicProvider` (which bills a separate ``ANTHROPIC_API_KEY``),
    this shells out to the logged-in ``claude`` binary in headless mode
    (``claude -p ... --output-format json``), so it runs on your Claude
    subscription with no API key. ``total_cost_usd`` from the CLI is the
    equivalent API cost (covered by the subscription).

    Tool-calling and forced structured output aren't exposed through the CLI, so
    ``complete`` returns text only; ``response_schema`` is honored best-effort by
    instructing the model to emit matching JSON. The subprocess ``runner`` is
    injectable for offline tests.
    """

    def __init__(self, binary: str | None = None, model: str | None = None,
                 runner=None) -> None:
        self.binary = binary or os.environ.get("AGENTOS_CLAUDE_BIN", "claude")
        self.model = model or os.environ.get("AGENTOS_CLAUDE_MODEL")  # None => CLI default
        self._runner = runner  # (cmd: list[str], stdin: str) -> (stdout, returncode)

    @staticmethod
    def _flatten(messages: list[dict]) -> tuple[str, str]:
        system_parts, user_parts = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content)
            (system_parts if role == "system" else user_parts).append(content)
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    def _run(self, cmd: list[str], stdin: str) -> tuple[str, int]:
        if self._runner is not None:
            return self._runner(cmd, stdin)
        import subprocess  # lazy; stdlib
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
        return proc.stdout, proc.returncode

    def complete(self, messages, tools=None, response_schema=None) -> LLMResult:
        system, user = self._flatten(messages)
        if response_schema is not None:
            user += ("\n\nRespond with ONLY a JSON object matching this schema, "
                     f"no prose:\n{json.dumps(response_schema)}")
        cmd = [self.binary, "-p", "--output-format", "json"]
        if self.model:
            cmd += ["--model", self.model]
        if system:
            cmd += ["--append-system-prompt", system]
        stdout, rc = self._run(cmd, user)
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, TypeError):
            return LLMResult(text="", model="claude-cli", confidence=0.0,
                             raw={"returncode": rc, "stdout": (stdout or "")[:500]})
        usage = data.get("usage") or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cache_tok = int(usage.get("cache_read_input_tokens", 0) or 0)
        errored = bool(data.get("is_error")) or data.get("api_error_status")
        return LLMResult(
            text=data.get("result", "") or "",
            tool_calls=[],
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_tokens=cache_tok,
            model=data.get("model") or self.model or "claude-cli",
            cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            confidence=0.0 if errored else 1.0,
            raw=data,
        )

    def stream(self, messages, tools=None) -> Iterator[str]:
        # The CLI can stream, but for simplicity yield the completed text once.
        yield self.complete(messages, tools=tools).text


def auto_provider() -> LLMProvider:
    """Pick the best available provider for an *application* (not tests).

    Order: ``$AGENTOS_LLM`` if set; else the Claude CLI (your Max/Pro
    subscription) if the ``claude`` binary is on PATH; else ``ANTHROPIC_API_KEY``
    via the API; else the offline MockLLM. Library/test default stays MockLLM —
    only apps that opt in (the ``agent-os`` CLI, the server) call this.
    """
    import shutil

    if os.environ.get("AGENTOS_LLM"):
        return get_provider()
    if shutil.which(os.environ.get("AGENTOS_CLAUDE_BIN", "claude")):
        return ClaudeCliProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    return MockLLM()


def get_provider(name: str | None = None) -> LLMProvider:
    """Return a provider instance.

    Selection order: explicit ``name`` argument, else ``$AGENTOS_LLM``, else
    ``"mock"``. Recognised: ``mock`` | ``ollama`` | ``anthropic`` |
    ``claude-cli`` (aliases: ``claude-code``, ``max``, ``claude``).
    """
    choice = (name or os.environ.get("AGENTOS_LLM", "mock")).strip().lower()
    if choice == "mock":
        return MockLLM()
    if choice == "ollama":
        return OllamaProvider()
    if choice == "anthropic":
        return AnthropicProvider()
    if choice in ("claude-cli", "claude-code", "max", "claude"):
        return ClaudeCliProvider()
    raise ValueError(
        f"unknown provider {choice!r} (use mock|ollama|anthropic|claude-cli)"
    )


__all__ = [
    "LLMResult",
    "LLMProvider",
    "PRICING",
    "cost_of",
    "MockLLM",
    "OllamaProvider",
    "AnthropicProvider",
    "ClaudeCliProvider",
    "FallbackLLM",
    "RateLimitError",
    "get_provider",
]
