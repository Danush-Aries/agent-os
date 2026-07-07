"""Real-world tool-integrations pack for Agent OS.

Registers a batch of practical tools — web search, HTTP fetch, GitHub REST,
sandboxed filesystem, and shell execution — into a :class:`ToolRegistry`. The
core stays stdlib-only: network integrations import ``httpx`` lazily behind the
optional ``[integrations]`` extra, and every network entry point accepts an
injectable transport so the whole pack is offline-testable.

Usage::

    from agentos.tools import ToolRegistry
    from agentos.toolpacks import register_all

    reg = ToolRegistry()
    register_all(reg)                       # all tools
    register_all(reg, enable=["web_search"])  # subset by name

Writing / creating / executing tools (``fs_write``, ``gh_create_issue``,
``shell_run``) set ``requires_approval=True`` so the kernel HITL gate governs
them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..tools import Tool, ToolRegistry
from .fs import fs_list_tool, fs_read_tool, fs_write_tool
from .github import (
    gh_create_issue_tool,
    gh_get_repo_tool,
    gh_list_issues_tool,
    gh_search_repos_tool,
)
from .shell import shell_run_tool
from .web import http_fetch_tool, web_search_tool

__all__ = [
    "register_all",
    "all_tools",
    "web_search_tool",
    "http_fetch_tool",
    "gh_search_repos_tool",
    "gh_get_repo_tool",
    "gh_list_issues_tool",
    "gh_create_issue_tool",
    "fs_read_tool",
    "fs_write_tool",
    "fs_list_tool",
    "shell_run_tool",
]


def all_tools(fs_root: Path | str | None = None) -> list[Tool]:
    """Build (but do not register) every tool in the pack."""
    return [
        web_search_tool(),
        http_fetch_tool(),
        gh_search_repos_tool(),
        gh_get_repo_tool(),
        gh_list_issues_tool(),
        gh_create_issue_tool(),
        fs_read_tool(fs_root),
        fs_write_tool(fs_root),
        fs_list_tool(fs_root),
        shell_run_tool(),
    ]


def register_all(
    registry: ToolRegistry,
    *,
    enable: Iterable[str] | None = None,
    fs_root: Path | str | None = None,
) -> list[str]:
    """Register the tool pack into ``registry``.

    ``enable`` optionally filters the tools by name (an iterable of tool
    names). ``fs_root`` sets the sandbox root for filesystem tools (default:
    current working directory). Returns the list of registered tool names.
    """
    allow = set(enable) if enable is not None else None
    registered: list[str] = []
    for tool in all_tools(fs_root):
        if allow is not None and tool.name not in allow:
            continue
        registry.register(tool)
        registered.append(tool.name)
    return registered
