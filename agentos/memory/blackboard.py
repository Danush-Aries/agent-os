"""Shared memory for Agent OS: a SQLite-backed blackboard.

Every agent reads and writes the same key/value store, so results produced by
one agent are visible to the next — the "shared memory" that turns isolated
handlers into a cooperating system. Use ``:memory:`` for ephemeral runs or a
file path to persist across boots.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Any


class Blackboard:
    def __init__(self, path: str = ":memory:") -> None:
        # check_same_thread=False so a threaded scheduler can share one board;
        # a lock serializes writes to keep SQLite happy.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
            self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key FROM kv WHERE key LIKE ? ORDER BY key", (f"{prefix}%",)
            ).fetchall()
        return [r[0] for r in rows]

    def items(self, prefix: str = "") -> dict[str, Any]:
        return {k: self.get(k) for k in self.keys(prefix)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
