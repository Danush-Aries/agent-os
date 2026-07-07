"""Long-term memory: SQLite-backed, namespaced key/value that survives restarts.

Unlike the blackboard (a flat kv store) this is namespaced (``ns``) so different
agents or concerns can partition their durable state without key collisions.
Values are JSON-encoded; all SQL is parameterized.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any


class LongTermMemory:
    def __init__(self, path: str = "agentos_ltm.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ns TEXT NOT NULL, "
            "key TEXT NOT NULL, "
            "value_json TEXT NOT NULL, "
            "ts REAL NOT NULL, "
            "UNIQUE(ns, key))"
        )
        self._conn.commit()

    def put(self, ns: str, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (ns, key, value_json, ts) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(ns, key) DO UPDATE SET "
                "value_json = excluded.value_json, ts = excluded.ts",
                (ns, key, json.dumps(value), time.time()),
            )
            self._conn.commit()

    def get(self, ns: str, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json FROM memories WHERE ns = ? AND key = ?", (ns, key)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def all(self, ns: str) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value_json FROM memories WHERE ns = ? ORDER BY key", (ns,)
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}

    def delete(self, ns: str, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE ns = ? AND key = ?", (ns, key))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
