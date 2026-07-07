"""Semantic memory: add text, search by meaning via cosine similarity.

Documents are embedded (default: ``HashingEmbedder``) and kept in an in-memory
list; an optional SQLite file gives durability. Search filters on metadata
(exact match on the provided keys) BEFORE ranking, so a filter narrows the
candidate set and then cosine similarity orders what remains. Cosine is
pure-python — no numpy on the default path.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from typing import Any

from .embedders import Embedder, HashingEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class _Record:
    __slots__ = ("text", "vector", "metadata")

    def __init__(self, text: str, vector: list[float], metadata: dict[str, Any]) -> None:
        self.text = text
        self.vector = vector
        self.metadata = metadata


class SemanticMemory:
    def __init__(
        self,
        embedder: Embedder | None = None,
        path: str | None = None,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._records: list[_Record] = []
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if path is not None:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "text TEXT NOT NULL, "
                "vector_json TEXT NOT NULL, "
                "metadata_json TEXT NOT NULL)"
            )
            self._conn.commit()
            self._load()

    def _load(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT text, vector_json, metadata_json FROM documents ORDER BY id"
        ).fetchall()
        self._records = [
            _Record(t, json.loads(v), json.loads(m)) for t, v, m in rows
        ]

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        meta = dict(metadata) if metadata else {}
        vector = self.embedder.embed(text)
        with self._lock:
            self._records.append(_Record(text, vector, meta))
            if self._conn is not None:
                self._conn.execute(
                    "INSERT INTO documents (text, vector_json, metadata_json) "
                    "VALUES (?, ?, ?)",
                    (text, json.dumps(vector), json.dumps(meta)),
                )
                self._conn.commit()

    @staticmethod
    def _matches(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        return all(metadata.get(k) == v for k, v in filter.items())

    def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        qvec = self.embedder.embed(query)
        with self._lock:
            candidates = list(self._records)
        if filter:
            candidates = [r for r in candidates if self._matches(r.metadata, filter)]
        scored = [
            {"text": r.text, "score": _cosine(qvec, r.vector), "metadata": r.metadata}
            for r in candidates
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:k]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
