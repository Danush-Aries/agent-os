"""Filesystem tools sandboxed to a configurable root.

Pure stdlib. Every path is resolved and checked to be *inside* the sandbox
root; anything that escapes (``..`` traversal, absolute paths outside the root,
symlink escape) is refused with a ``ValueError``. The root defaults to the
current working directory and can be overridden per factory call.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..tools import Tool


class SandboxError(ValueError):
    """Raised when a path escapes the sandbox root."""


def _resolve_in_root(root: Path, path: str) -> Path:
    """Resolve ``path`` relative to ``root`` and confirm it stays inside."""
    root = root.resolve()
    candidate = (root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise SandboxError(f"path '{path}' escapes sandbox root")
    return candidate


def make_fs_read(root: Path):
    def fs_read(path: str) -> str:
        target = _resolve_in_root(root, path)
        return target.read_text(encoding="utf-8")
    return fs_read


def make_fs_write(root: Path):
    def fs_write(path: str, content: str) -> dict:
        target = _resolve_in_root(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        n = target.write_text(content, encoding="utf-8")
        return {"path": str(target), "bytes_written": n}
    return fs_write


def make_fs_list(root: Path):
    def fs_list(path: str = ".") -> list[dict]:
        target = _resolve_in_root(root, path)
        if not target.exists():
            raise FileNotFoundError(f"no such path: {path}")
        if target.is_file():
            entries = [target]
        else:
            entries = sorted(target.iterdir())
        return [
            {"name": e.name,
             "is_dir": e.is_dir(),
             "size": e.stat().st_size if e.is_file() else None}
            for e in entries
        ]
    return fs_list


# --------------------------------------------------------------------------- #
# Tool factories
# --------------------------------------------------------------------------- #
def fs_read_tool(root: Path | str | None = None) -> Tool:
    root = Path(root) if root is not None else Path.cwd()
    return Tool(
        name="fs_read",
        description="Read a UTF-8 text file inside the sandbox root.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        func=make_fs_read(root),
    )


def fs_write_tool(root: Path | str | None = None) -> Tool:
    root = Path(root) if root is not None else Path.cwd()
    return Tool(
        name="fs_write",
        description="Write text to a file inside the sandbox root (requires approval).",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        func=make_fs_write(root),
        requires_approval=True,
    )


def fs_list_tool(root: Path | str | None = None) -> Tool:
    root = Path(root) if root is not None else Path.cwd()
    return Tool(
        name="fs_list",
        description="List directory entries inside the sandbox root.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
        func=make_fs_list(root),
    )
