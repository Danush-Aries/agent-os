"""Embedders: turn text into vectors for semantic search.

The default ``HashingEmbedder`` is pure-stdlib and deterministic — it hashes
tokens into a fixed-dimensional bag-of-words vector and L2-normalizes it, so no
numpy, no network, no model download. Good enough to rank related text above
unrelated text in tests and small workloads.

``OllamaEmbedder`` is optional and only useful with the ``[vector]`` extra; it
lazily imports ``httpx`` so importing this module never requires it.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """Deterministic hashing bag-of-words embedder (pure stdlib)."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            vec[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class OllamaEmbedder:
    """Optional Ollama-backed embedder. Requires the ``[vector]`` extra (httpx)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def embed(self, text: str) -> list[float]:
        import httpx  # lazy: only needed when actually used

        resp = httpx.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
