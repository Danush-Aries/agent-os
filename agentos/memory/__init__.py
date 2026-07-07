"""Layered memory for Agent OS.

Tiers, from most transient to most durable:

- ``Blackboard``       — shared SQLite kv store (the original cooperating memory)
- ``ShortTermMemory``  — in-process, run-scoped scratchpad (no persistence)
- ``LongTermMemory``   — namespaced, SQLite-backed, survives restarts
- ``SemanticMemory``   — embed + cosine search over remembered text

Plus helpers: ``HashingEmbedder`` (default embedder) and ``window_messages``
(context-window trimming). ``from agentos.memory import Blackboard`` keeps
working exactly as before.
"""

from __future__ import annotations

from .blackboard import Blackboard
from .context import window_messages
from .embedders import Embedder, HashingEmbedder
from .long_term import LongTermMemory
from .semantic import SemanticMemory
from .short_term import ShortTermMemory

__all__ = [
    "Blackboard",
    "ShortTermMemory",
    "LongTermMemory",
    "SemanticMemory",
    "Embedder",
    "HashingEmbedder",
    "window_messages",
]
