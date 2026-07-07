"""Offline tests for the layered memory package."""

from __future__ import annotations

import math

from agentos.memory import (
    Blackboard,
    HashingEmbedder,
    LongTermMemory,
    SemanticMemory,
    ShortTermMemory,
    window_messages,
)


# --- Blackboard (preserved public API) --------------------------------------


def test_blackboard_put_get_keys():
    bb = Blackboard()
    bb.put("a", 1)
    bb.put("b", {"x": 2})
    assert bb.get("a") == 1
    assert bb.get("b") == {"x": 2}
    assert bb.get("missing", "default") == "default"
    assert bb.keys() == ["a", "b"]
    bb.close()


# --- ShortTermMemory --------------------------------------------------------


def test_short_term_append_accumulates():
    stm = ShortTermMemory()
    stm.append("events", "one")
    stm.append("events", "two")
    assert stm.get("events") == ["one", "two"]


def test_short_term_scalar_and_clear():
    stm = ShortTermMemory()
    stm.set("count", 5)
    assert stm.get("count") == 5
    stm.append("log", "x")
    stm.clear()
    assert stm.get("count") is None
    assert stm.get("log") is None


# --- LongTermMemory ---------------------------------------------------------


def test_long_term_persists_across_instances(tmp_path):
    db = str(tmp_path / "ltm.db")
    ltm = LongTermMemory(db)
    ltm.put("agents", "greeting", {"msg": "hello"})
    ltm.put("agents", "count", 3)
    ltm.close()

    fresh = LongTermMemory(db)
    assert fresh.get("agents", "greeting") == {"msg": "hello"}
    assert fresh.all("agents") == {"count": 3, "greeting": {"msg": "hello"}}
    fresh.delete("agents", "count")
    assert fresh.get("agents", "count") is None
    fresh.close()


# --- HashingEmbedder --------------------------------------------------------


def test_hashing_embedder_deterministic_and_unit_norm():
    emb = HashingEmbedder()
    v1 = emb.embed("the quick brown fox")
    v2 = emb.embed("the quick brown fox")
    assert v1 == v2
    assert len(v1) == 256
    norm = math.sqrt(sum(x * x for x in v1))
    assert abs(norm - 1.0) < 1e-9


# --- SemanticMemory ---------------------------------------------------------


def test_semantic_search_ranks_related_above_unrelated():
    sm = SemanticMemory()
    sm.add("cats and dogs are common household pets", {"topic": "animals"})
    sm.add("the stock market rallied on tech earnings", {"topic": "finance"})
    results = sm.search("household pets like cats", k=2)
    assert results[0]["text"] == "cats and dogs are common household pets"
    assert results[0]["score"] > results[1]["score"]


def test_semantic_search_respects_metadata_filter():
    sm = SemanticMemory()
    sm.add("python is a programming language", {"lang": "en"})
    sm.add("le python est un langage de programmation", {"lang": "fr"})
    results = sm.search("programming language python", k=5, filter={"lang": "fr"})
    assert len(results) == 1
    assert results[0]["metadata"]["lang"] == "fr"


# --- window_messages --------------------------------------------------------


def test_window_messages_keeps_recent():
    msgs = [
        {"role": "user", "content": "one two three four five"},
        {"role": "assistant", "content": "six seven eight"},
        {"role": "user", "content": "nine ten"},
    ]
    kept = window_messages(msgs, max_tokens=5)
    assert kept[-1]["content"] == "nine ten"
    assert msgs[0] not in kept  # oldest dropped under tight budget


def test_window_messages_honors_summarizer():
    msgs = [
        {"role": "user", "content": "alpha beta gamma delta"},
        {"role": "assistant", "content": "epsilon zeta eta"},
        {"role": "user", "content": "theta"},
    ]

    def summarize(dropped):
        return {"role": "system", "content": f"summary of {len(dropped)} msgs"}

    kept = window_messages(msgs, max_tokens=2, summarizer=summarize)
    assert kept[0]["role"] == "system"
    assert kept[0]["content"].startswith("summary of")
    assert kept[-1]["content"] == "theta"
