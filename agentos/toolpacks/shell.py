"""Shell tool: run a command safely and capture its output.

Commands run with ``shell=False`` (argv split via :mod:`shlex`), so there is no
shell metacharacter interpretation. A conservative denylist refuses obviously
destructive commands. Flagged ``requires_approval`` so the kernel HITL gate
governs any invocation.
"""

from __future__ import annotations

import shlex
import subprocess

from ..tools import Tool

# Substrings that mark a command as too dangerous to run automatically. Matched
# against the normalized (whitespace-collapsed) command string.
_DENY_SUBSTRINGS = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    ":(){",           # fork bomb
    "> /dev/sda",
    "chmod -r 777 /",
    "chown -r",
    "shutdown",
    "reboot",
    "halt",
    "init 0",
    "init 6",
    "wget ",          # avoid silent network fetch-and-run
    "curl ",
)
_DENY_BINARIES = {"mkfs", "dd", "shutdown", "reboot", "halt", "poweroff", "mkswap"}


def _is_denied(command: str) -> bool:
    norm = " ".join(command.strip().lower().split())
    if any(bad in norm for bad in _DENY_SUBSTRINGS):
        return True
    try:
        parts = shlex.split(command)
    except ValueError:
        return True  # unbalanced quotes → refuse rather than guess
    if parts and parts[0].split("/")[-1] in _DENY_BINARIES:
        return True
    return False


def shell_run(command: str, timeout: int = 30) -> dict:
    """Run ``command`` (no shell) and return stdout/stderr/returncode.

    Refuses denylisted destructive commands with ``is_error`` semantics (a
    ``PermissionError`` the registry wraps).
    """
    if _is_denied(command):
        raise PermissionError(f"refused destructive command: {command!r}")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"could not parse command: {exc}") from exc
    if not argv:
        raise ValueError("empty command")

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def shell_run_tool() -> Tool:
    return Tool(
        name="shell_run",
        description="Run a shell command (no shell metachars; denylist enforced; "
        "requires approval).",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
        func=shell_run,
        requires_approval=True,
        rate_limit=20,
    )
