"""Optional REST + WebSocket server and web dashboard for Agent OS.

This module is deliberately import-safe: it does **not** import FastAPI at
module load time. Everything web-related lives inside ``create_app()`` so the
pure-stdlib core import graph stays dependency-free. Install the extra with::

    uv pip install -e '.[api]'

Then either::

    agentos serve            # via the CLI (if wired)
    python -m agentos.server # runs uvicorn directly

Endpoints
---------
- ``POST /tasks``          submit a task, run to quiescence, return ``{task_id}``
- ``GET  /tasks``          list every task (``Task.as_dict()``)
- ``GET  /tasks/{id}``     one task
- ``GET  /approvals``      pending HITL approvals
- ``POST /approvals/{id}`` resolve an approval, then re-run to quiescence
- ``GET  /metrics``        ``kernel.metrics()``
- ``GET  /trace/{id}``     ``kernel.trace(id)``
- ``GET  /``              the dashboard HTML
- ``WS   /ws/events``      live stream of tracer events as JSON
"""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type checkers only; never imported at runtime by core
    from fastapi import FastAPI

    from .kernel import Kernel

_STATIC_DIR = Path(__file__).parent / "static"
_DASHBOARD = _STATIC_DIR / "dashboard.html"


def _build_default_kernel() -> "Kernel":
    """The fully-loaded kernel the dashboard drives.

    Boots every agent (built-ins + LLM/planner) and the real-world tool
    integrations, wires the LLM provider via ``auto_provider()`` (your Claude
    Max subscription through the ``claude`` CLI when available), and uses a
    *deferring* approval gate so ``requires_approval`` tasks surface in
    ``/approvals`` for a human — which is what makes the dashboard useful.
    """
    from .boot import default_kernel
    from .llm import auto_provider

    return default_kernel(llm=auto_provider(), approval_policy=lambda _task: None)


def create_app(kernel: "Kernel | None" = None) -> "FastAPI":
    """Build the FastAPI app bound to ``kernel`` (or a fresh default one)."""
    import asyncio
    import json

    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel

    from .task import Task

    kern = kernel if kernel is not None else _build_default_kernel()
    # Serialize kernel.run() calls; the kernel run loop is not reentrant.
    run_lock = threading.Lock()

    def _run_to_quiescence() -> None:
        with run_lock:
            kern.run()

    app = FastAPI(title="Agent OS", version="0.1.0")
    app.state.kernel = kern

    # --- request models ------------------------------------------------------

    class TaskIn(BaseModel):
        kind: str
        payload: dict[str, Any] = {}
        priority: int | None = None
        deps: list[int] = []
        requires_approval: bool = False

    class ApprovalIn(BaseModel):
        approved: bool

    # --- tasks ---------------------------------------------------------------

    @app.post("/tasks")
    def submit_task(body: TaskIn) -> dict[str, int]:
        from .task import Priority

        task = Task(
            kind=body.kind,
            payload=body.payload,
            priority=Priority.NORMAL if body.priority is None else body.priority,
            deps=list(body.deps),
            requires_approval=body.requires_approval,
        )
        task_id = kern.submit(task)
        _run_to_quiescence()
        return {"task_id": task_id}

    @app.get("/tasks")
    def list_tasks() -> list[dict[str, Any]]:
        return [t.as_dict() for t in kern.ps()]

    @app.get("/tasks/{task_id}")
    def get_task(task_id: int) -> dict[str, Any]:
        task = kern.scheduler.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"no task {task_id}")
        return task.as_dict()

    # --- approvals -----------------------------------------------------------

    @app.get("/approvals")
    def list_approvals() -> list[dict[str, Any]]:
        return [
            {"task_id": p.task_id, "kind": p.kind, "summary": p.summary}
            for p in kern.approvals.pending()
        ]

    @app.post("/approvals/{task_id}")
    def resolve_approval(task_id: int, body: ApprovalIn) -> dict[str, Any]:
        kern.approvals.resolve(task_id, body.approved)
        # Re-run so the (now decided) task transitions on the spot.
        _run_to_quiescence()
        return {"ok": True, "task_id": task_id, "approved": body.approved}

    # --- observability -------------------------------------------------------

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return kern.metrics()

    @app.get("/trace/{task_id}")
    def trace(task_id: int) -> dict[str, Any]:
        tr = kern.trace(task_id)
        if tr is None:
            raise HTTPException(status_code=404, detail=f"no trace for {task_id}")
        return tr

    # --- dashboard -----------------------------------------------------------

    @app.get("/")
    def dashboard() -> Any:
        if _DASHBOARD.exists():
            return FileResponse(str(_DASHBOARD), media_type="text/html")
        return JSONResponse({"detail": "dashboard.html not found"}, status_code=404)

    # --- live events ---------------------------------------------------------

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket) -> None:
        await ws.accept()
        last_seq = 0
        try:
            while True:
                events = kern.tracer.events()
                fresh = [e for e in events if e.seq > last_seq]
                for ev in fresh:
                    last_seq = max(last_seq, ev.seq)
                    await ws.send_text(json.dumps(ev.as_dict()))
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return
        except Exception:
            # Client vanished or send failed; close quietly.
            try:
                await ws.close()
            except Exception:
                pass

    return app


def run(app=None, open_browser: bool = False) -> None:
    """Console entry point: launch uvicorn on ``AGENTOS_PORT`` (default 8080).

    ``app`` lets a caller pass a prebuilt FastAPI app (e.g. one bound to a
    custom kernel). ``open_browser`` pops the dashboard once the server is up.
    """
    import threading
    import time
    import webbrowser

    import uvicorn

    port = int(os.environ.get("AGENTOS_PORT", "8080"))
    host = os.environ.get("AGENTOS_HOST", "127.0.0.1")
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"

    if open_browser:
        def _open():
            time.sleep(1.2)  # give uvicorn a moment to bind
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  Agent OS dashboard → {url}\n  (Ctrl-C to stop)\n")
    uvicorn.run(app if app is not None else create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    run()
