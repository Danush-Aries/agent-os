"""GitHub tools built on the REST API v3.

``httpx`` is imported lazily; an optional ``GITHUB_TOKEN`` env var is used for
authentication when present. Every entry point accepts an injectable
``request`` callable ``(method, url, *, params, json) -> dict`` so tests can
feed canned JSON with no network. Response shaping lives in pure helpers that
are unit-tested directly.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ..tools import Tool

API_ROOT = "https://api.github.com"


# --------------------------------------------------------------------------- #
# Pure response shapers (unit-tested)
# --------------------------------------------------------------------------- #
def _shape_repo(item: dict) -> dict:
    """Trim a GitHub repo object to the fields agents care about."""
    return {
        "full_name": item.get("full_name"),
        "html_url": item.get("html_url"),
        "description": item.get("description"),
        "stars": item.get("stargazers_count", 0),
        "language": item.get("language"),
        "forks": item.get("forks_count", 0),
        "open_issues": item.get("open_issues_count", 0),
    }


def _shape_search_repos(data: dict, k: int = 5) -> list[dict]:
    items = data.get("items") or []
    return [_shape_repo(it) for it in items[: max(0, k)]]


def _shape_issue(item: dict) -> dict:
    return {
        "number": item.get("number"),
        "title": item.get("title"),
        "state": item.get("state"),
        "html_url": item.get("html_url"),
        "user": (item.get("user") or {}).get("login"),
        "labels": [lbl.get("name") for lbl in (item.get("labels") or [])
                   if isinstance(lbl, dict)],
    }


def _shape_issues(data: list, k: int | None = None) -> list[dict]:
    issues = [_shape_issue(it) for it in data if isinstance(it, dict)]
    return issues if k is None else issues[: max(0, k)]


# --------------------------------------------------------------------------- #
# Default transport (lazy httpx)
# --------------------------------------------------------------------------- #
def _default_request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
) -> Any:
    import httpx  # lazy

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with httpx.Client(timeout=15.0) as c:
        resp = c.request(method, url, params=params, json=json, headers=headers)
        return resp.json()


def _run(request: Callable | None, method: str, url: str, **kw) -> Any:
    return (request or _default_request)(method, url, **kw)


# --------------------------------------------------------------------------- #
# Tool functions
# --------------------------------------------------------------------------- #
def gh_search_repos(query: str, k: int = 5, *,
                    request: Callable | None = None) -> list[dict]:
    data = _run(request, "GET", f"{API_ROOT}/search/repositories",
                params={"q": query, "per_page": k})
    return _shape_search_repos(data, k)


def gh_get_repo(owner: str, repo: str, *,
                request: Callable | None = None) -> dict:
    data = _run(request, "GET", f"{API_ROOT}/repos/{owner}/{repo}")
    return _shape_repo(data)


def gh_list_issues(owner: str, repo: str, state: str = "open", *,
                   request: Callable | None = None) -> list[dict]:
    data = _run(request, "GET", f"{API_ROOT}/repos/{owner}/{repo}/issues",
                params={"state": state})
    return _shape_issues(data if isinstance(data, list) else [])


def gh_create_issue(owner: str, repo: str, title: str, body: str = "", *,
                    request: Callable | None = None) -> dict:
    data = _run(request, "POST", f"{API_ROOT}/repos/{owner}/{repo}/issues",
                json={"title": title, "body": body})
    return _shape_issue(data)


# --------------------------------------------------------------------------- #
# Tool factories
# --------------------------------------------------------------------------- #
def gh_search_repos_tool() -> Tool:
    return Tool(
        name="gh_search_repos",
        description="Search GitHub repositories; returns trimmed repo dicts.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
        func=gh_search_repos,
        rate_limit=30,
    )


def gh_get_repo_tool() -> Tool:
    return Tool(
        name="gh_get_repo",
        description="Fetch metadata for a single GitHub repository.",
        input_schema={
            "type": "object",
            "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}},
            "required": ["owner", "repo"],
        },
        func=gh_get_repo,
        rate_limit=60,
    )


def gh_list_issues_tool() -> Tool:
    return Tool(
        name="gh_list_issues",
        description="List issues for a GitHub repo (state: open|closed|all).",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"]},
            },
            "required": ["owner", "repo"],
        },
        func=gh_list_issues,
        rate_limit=60,
    )


def gh_create_issue_tool() -> Tool:
    return Tool(
        name="gh_create_issue",
        description="Create a new issue on a GitHub repo (requires approval).",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["owner", "repo", "title"],
        },
        func=gh_create_issue,
        requires_approval=True,
        rate_limit=10,
    )
