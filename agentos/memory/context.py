"""Context-window management: keep a message list under a token budget.

``window_messages`` is a sliding window over chat-style ``{"role", "content"}``
dicts. It keeps the most recent messages whose combined approximate token count
(a word-count heuristic) fits ``max_tokens``. If a ``summarizer`` is supplied,
the dropped older prefix is replaced by a single summary message; otherwise the
prefix is simply dropped.
"""

from __future__ import annotations

from typing import Callable


def _approx_tokens(message: dict) -> int:
    """Rough token estimate: number of whitespace-separated words in content."""
    content = message.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    return max(1, len(content.split()))


def window_messages(
    messages: list[dict],
    max_tokens: int,
    summarizer: Callable[[list[dict]], dict] | None = None,
) -> list[dict]:
    # Walk newest-first, keeping messages until the budget is exhausted.
    kept: list[dict] = []
    total = 0
    cut = 0  # index in original list where kept messages begin
    for i in range(len(messages) - 1, -1, -1):
        cost = _approx_tokens(messages[i])
        if total + cost > max_tokens and kept:
            cut = i + 1
            break
        total += cost
        kept.append(messages[i])
    else:
        cut = 0
    kept.reverse()

    dropped = messages[:cut]
    if dropped and summarizer is not None:
        return [summarizer(dropped), *kept]
    return kept
