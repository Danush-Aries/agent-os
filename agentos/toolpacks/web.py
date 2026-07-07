"""Web tools: search and raw HTTP fetch.

Networking is done with :mod:`httpx`, imported *lazily* so the core package
stays stdlib-only. Every network entry point accepts an injectable transport
(a ``client`` object or a ``fetch`` callable) so tests can supply canned data
without touching the network. The response-shaping logic lives in the pure
``_parse_ddg`` function, which is unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Callable

from ..tools import Tool

DDG_ENDPOINT = "https://api.duckduckgo.com/"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested, no I/O)
# --------------------------------------------------------------------------- #
def _parse_ddg(data: dict, k: int = 5) -> list[dict]:
    """Shape a DuckDuckGo Instant-Answer JSON payload into result dicts.

    Returns a list of ``{"title", "url", "snippet"}``. Draws from the
    ``Abstract*`` fields first (the headline answer), then flattens
    ``RelatedTopics`` (which may contain nested ``Topics`` groups).
    """
    results: list[dict] = []

    heading = (data.get("Heading") or "").strip()
    abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract and abstract_url:
        results.append({
            "title": heading or abstract_url,
            "url": abstract_url,
            "snippet": abstract,
        })

    def _flatten(topics: list) -> None:
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            if "Topics" in topic and isinstance(topic["Topics"], list):
                _flatten(topic["Topics"])
                continue
            url = (topic.get("FirstURL") or "").strip()
            text = (topic.get("Text") or "").strip()
            if not url:
                continue
            title = text.split(" - ", 1)[0] if text else url
            results.append({"title": title, "url": url, "snippet": text})

    _flatten(data.get("RelatedTopics") or [])

    # de-dup by url, preserve order, cap at k
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in results:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        deduped.append(r)
    return deduped[: max(0, k)]


def _shape_fetch(status: int, text: str, headers: dict | None) -> dict:
    """Pure shaper for :func:`http_fetch` responses."""
    return {"status": int(status), "text": text, "headers": dict(headers or {})}


# --------------------------------------------------------------------------- #
# Network functions (lazy httpx, injectable transport)
# --------------------------------------------------------------------------- #
def web_search(
    query: str,
    k: int = 5,
    *,
    client: Any | None = None,
    fetch: Callable[..., dict] | None = None,
) -> list[dict]:
    """Search the web via the DuckDuckGo instant-answer JSON API.

    ``fetch`` (highest precedence) is a callable ``(url, params) -> dict`` used
    verbatim for tests. ``client`` is an httpx-like object exposing ``.get``.
    If neither is given a real ``httpx.Client`` is created lazily.
    """
    params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}

    if fetch is not None:
        data = fetch(DDG_ENDPOINT, params)
        return _parse_ddg(data, k)

    if client is None:
        import httpx  # lazy: only needed for real network calls

        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            resp = c.get(DDG_ENDPOINT, params=params)
            data = resp.json()
    else:
        resp = client.get(DDG_ENDPOINT, params=params)
        data = resp.json()
    return _parse_ddg(data, k)


def http_fetch(
    url: str,
    method: str = "GET",
    *,
    body: str | None = None,
    headers: dict | None = None,
    transport: Any | None = None,
) -> dict:
    """Fetch a URL and return ``{"status", "text", "headers"}``.

    ``transport`` is an injectable httpx-like object exposing ``.request`` (or
    ``.get``/``.post``). When absent a real ``httpx.Client`` is created lazily.
    """
    method = method.upper()

    def _do(obj: Any) -> dict:
        resp = obj.request(method, url, content=body, headers=headers)
        return _shape_fetch(resp.status_code, resp.text, dict(resp.headers))

    if transport is not None:
        return _do(transport)

    import httpx  # lazy

    with httpx.Client(timeout=15.0, follow_redirects=True) as c:
        return _do(c)


# --------------------------------------------------------------------------- #
# Tool factories
# --------------------------------------------------------------------------- #
def web_search_tool() -> Tool:
    return Tool(
        name="web_search",
        description="Search the web (DuckDuckGo) and return a list of "
        "{title, url, snippet} results.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
        func=web_search,
        rate_limit=30,
    )


def http_fetch_tool() -> Tool:
    return Tool(
        name="http_fetch",
        description="Fetch a URL over HTTP(S) and return {status, text, headers}.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string",
                           "enum": ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"]},
                "body": {"type": "string"},
                "headers": {"type": "object"},
            },
            "required": ["url"],
        },
        func=http_fetch,
        rate_limit=60,
    )
