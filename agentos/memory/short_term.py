"""Short-term memory: in-process, session/run-scoped scratchpad.

Nothing here is persisted. It exists for the lifetime of a single run and is
meant for reducer-style accumulation (``append``) and transient scalars
(``set``/``get``). Restarting the process wipes it clean.
"""

from __future__ import annotations

from typing import Any


class ShortTermMemory:
    def __init__(self) -> None:
        self._lists: dict[str, list[Any]] = {}
        self._scalars: dict[str, Any] = {}

    def append(self, key: str, item: Any) -> list[Any]:
        """Reducer-style append: accumulate ``item`` into the list at ``key``."""
        self._lists.setdefault(key, []).append(item)
        return self._lists[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return the list at ``key`` if one exists, else the scalar, else default."""
        if key in self._lists:
            return self._lists[key]
        return self._scalars.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._scalars[key] = value

    def clear(self) -> None:
        self._lists.clear()
        self._scalars.clear()
