"""Tests for the optional REST server + dashboard (the ``[api]`` extra).

Skipped entirely when FastAPI isn't installed, so the core test run stays
dependency-free. Run with::

    uv run --extra dev --extra api pytest tests/test_server.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agentos.agents import BUILTIN_AGENTS  # noqa: E402
from agentos.hitl import ApprovalGate  # noqa: E402
from agentos.kernel import Kernel  # noqa: E402
from agentos.server import create_app  # noqa: E402


def _kernel(policy=None):
    """A kernel with the built-in agents; deferring approval gate by default."""
    gate = ApprovalGate(policy=policy) if policy is not None else ApprovalGate(
        policy=lambda _t: None
    )
    k = Kernel(approvals=gate)
    for agent_cls in BUILTIN_AGENTS:
        k.register(agent_cls())
    return k


@pytest.fixture()
def client():
    return TestClient(create_app(_kernel()))


def test_default_app_boots_without_kernel():
    # create_app with no kernel must build one with the built-ins registered.
    with TestClient(create_app()) as c:
        assert c.get("/tasks").json() == []


def test_submit_calc_task_completes(client):
    r = client.post("/tasks", json={"kind": "calc", "payload": {"op": "+", "a": 2, "b": 3}})
    assert r.status_code == 200
    tid = r.json()["task_id"]

    tasks = client.get("/tasks").json()
    assert isinstance(tasks, list) and len(tasks) == 1
    task = tasks[0]
    assert task["id"] == tid
    assert task["status"] == "completed"
    assert task["result"] == 5

    one = client.get(f"/tasks/{tid}").json()
    assert one["status"] == "completed"
    assert one["result"] == 5


def test_get_missing_task_404(client):
    assert client.get("/tasks/9999").status_code == 404


def test_metrics_returns_dict(client):
    client.post("/tasks", json={"kind": "echo", "payload": {"message": "hi"}})
    m = client.get("/metrics").json()
    assert isinstance(m, dict)
    assert m["tasks_total"] == 1
    assert m["completed"] == 1


def test_approval_flow(client):
    # Submit a task that requires approval; the deferring gate parks it.
    r = client.post(
        "/tasks",
        json={"kind": "echo", "payload": {"message": "gated"}, "requires_approval": True},
    )
    tid = r.json()["task_id"]

    pending = client.get("/approvals").json()
    assert any(p["task_id"] == tid for p in pending)

    task = client.get(f"/tasks/{tid}").json()
    assert task["status"] == "awaiting_approval"

    # Approve -> endpoint re-runs to quiescence -> task completes.
    ok = client.post(f"/approvals/{tid}", json={"approved": True}).json()
    assert ok["ok"] is True

    assert client.get("/approvals").json() == []
    done = client.get(f"/tasks/{tid}").json()
    assert done["status"] == "completed"
    assert done["result"] == "gated"


def test_approval_deny(client):
    r = client.post(
        "/tasks",
        json={"kind": "echo", "payload": {"message": "no"}, "requires_approval": True},
    )
    tid = r.json()["task_id"]
    client.post(f"/approvals/{tid}", json={"approved": False})
    task = client.get(f"/tasks/{tid}").json()
    assert task["status"] == "failed"


def test_trace_returns_span_and_events(client):
    tid = client.post(
        "/tasks", json={"kind": "calc", "payload": {"op": "*", "a": 4, "b": 6}}
    ).json()["task_id"]
    tr = client.get(f"/trace/{tid}").json()
    assert tr["task_id"] == tid
    assert tr["span"] is not None
    assert isinstance(tr["events"], list) and len(tr["events"]) > 0
    kinds = {e["kind"] for e in tr["events"]}
    assert "task.completed" in kinds


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Agent OS" in r.text
