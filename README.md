# Agent OS

A tiny **operating system for cooperating agents**. Register agents, submit
tasks with priorities and dependencies, and let the kernel dispatch them
against shared memory — the way a real OS schedules processes. What started as a
priority scheduler is now a full agent runtime: concurrency, retries, timeouts,
layered + semantic memory, schema-validated tools, an LLM provider abstraction,
planning / multi-agent decomposition, human-in-the-loop approval, guardrails,
first-class observability, and four interfaces (SDK, CLI, REST, dashboard).

The core is pure standard library and deterministic. Everything that needs a
third-party package (LLM SDKs, the REST server, vector search) lives behind an
opt-in extra, so `pip install agent-os` stays dependency-free.

## Concepts

| OS analogy | Agent OS |
|---|---|
| process | **Task** (`kind`, `payload`, `priority`, `deps`, retries, timeout) |
| program | **Agent** (advertises which task kinds it `handles`) |
| shared memory | **Blackboard** (SQLite key/value, shared by all agents) |
| scheduler | **Scheduler** (priority + dependency-aware, cycle-checked) |
| kernel | **Kernel** (`boot → submit → run` dispatch loop, thread pool) |
| syscalls | **Context** (`spawn`, `delegate`, `call_tool`, `memory`, `llm`) |
| dmesg / audit log | **Tracer** (typed events, spans, metrics) |

## Quickstart

Install once, then a single command boots everything and opens the dashboard:

```bash
uv tool install "agent-os[api]"    # or: pipx install "agent-os[api]"
agent-os                            # boot all agents + tools + LLM, open the dashboard
```

`agent-os` with no arguments starts the REST server on http://localhost:8080,
loads every built-in agent and the real-world tool integrations, wires your LLM
provider, and opens the live dashboard. Other subcommands:

```bash
agent-os up            # same as bare `agent-os`
agent-os demo          # run the built-in fan-out/reduce pipeline
agent-os agents        # list built-in agents
agent-os ps            # run the demo, print the process table
agent-os metrics       # run the demo, print kernel metrics
agent-os trace 1       # pretty-print task 1's trace
agent-os run tasks.json
agent-os serve         # start the REST API without opening a browser
agent-os config --set AGENTOS_LLM=claude-cli   # persist defaults (see below)
```

### Use your Claude Max/Pro subscription (no API key)

Your Max subscription is used by the `claude` CLI, not the paid API. Agent OS
detects it automatically — if the `claude` binary is on your PATH, `agent-os`
routes LLM agents through it via `ClaudeCliProvider` (billed to your
subscription). To pin it explicitly:

```bash
agent-os config --set AGENTOS_LLM=claude-cli   # persisted to ~/.agentos/config.json
# or per-shell: export AGENTOS_LLM=claude-cli
```

Provider precedence for the app: `$AGENTOS_LLM` → Claude CLI (Max) → `ANTHROPIC_API_KEY` → offline MockLLM.

As a library:

```python
from agentos import Kernel, Task
from agentos.agents import BUILTIN_AGENTS

k = Kernel()
for A in BUILTIN_AGENTS:
    k.register(A())

k.submit(Task(kind="sum_pipeline", payload={"numbers": [1, 2, 3, 4]}))
report = k.run()
print(report.completed, "tasks done")   # the pipeline spawns children + reduces to 10
```

### Writing your own agent

```python
from agentos import Agent, Kernel, Task

class GreetAgent(Agent):
    name = "greet"
    handles = {"greet"}
    def handle(self, task, ctx):
        who = task.payload["who"]
        ctx.memory.put("last_greeted", who)      # shared memory
        return f"Hello, {who}!"

k = Kernel().register(GreetAgent())
tid = k.submit(Task(kind="greet", payload={"who": "world"}))
k.run()
print(k.scheduler.get(tid).result)               # "Hello, world!"
```

Agents can spawn follow-up tasks at runtime via `ctx.spawn(kind, payload,
deps=[...])`, call tools via `ctx.call_tool(...)`, and reach an LLM via
`ctx.llm` — which is how the built-in `PipelineAgent` fans work out and then
reduces the children's results.

## The 2026 feature set

**1 — Scheduling, concurrency, retries, timeouts.** Priority + dependency-aware
scheduler with cycle detection; a thread-pool run loop (`max_workers`); per-task
`max_retries` with exponential backoff + deterministic jitter; wall-clock
`timeout_s` per attempt; a dead-letter queue for exhausted tasks.

**2 — Layered + semantic memory.** `Blackboard` (shared SQLite kv), in-process
`ShortTermMemory`, durable namespaced `LongTermMemory`, and `SemanticMemory`
(embed + cosine search) with a stdlib `HashingEmbedder`, plus `window_messages`
for context-window trimming.

**3 — Tools with schema validation.** `Tool`s carry a JSON-Schema for their
arguments, validated by a dependency-free validator before the function runs.
Tools can require approval, be rate-limited, and export an Anthropic/OpenAI-style
spec for handing to a model. Failures are wrapped (`ToolResult.is_error`) so a
bad tool never crashes its caller.

**4 — LLM provider abstraction.** One `LLMProvider` interface with `MockLLM`
(offline, deterministic — used in tests and the default), `OllamaProvider`
(local, lazy `httpx`), and `AnthropicProvider` (lazy `anthropic`), plus a
`FallbackLLM` that fails over on rate limits and `cost_of(...)` token accounting.

**5 — Planner / multi-agent.** Agents decompose work at runtime with
`ctx.spawn` / `ctx.delegate`, expressing pipelines and DAGs; the scheduler
resolves the dependency graph and the blackboard passes results between agents.

**6 — Reliability: checkpoint, HITL, guardrails.** The kernel checkpoints the
task table after every transition and can `Kernel.resume(...)` a crashed run.
`requires_approval` routes a task through an `ApprovalGate` (auto-approve/deny
or a human queue). `Guardrail`s block secrets / prompt-injection and redact PII
on both inputs and outputs — and feed the tracer so secrets never hit logs.

**7 — Observability: events, spans, metrics.** Every state transition emits a
typed `Event`; spans build an OpenTelemetry GenAI-flavored tree
(`gen_ai.*` attributes, token counts, cost); counters roll up into
`kernel.metrics()`; `kernel.trace(id)` returns the span tree + event log for a
root task. A redactor can be attached so the audit log stays clean.

**8 — Interfaces: SDK, CLI, REST, dashboard.** Use it as a Python library, via
the `agentos` CLI, over the FastAPI REST server (`agentos serve`, `[api]`
extra), or through the bundled dashboard the server exposes.

## Install extras

```bash
uv tool install agent-os            # core runtime — zero dependencies
uv tool install "agent-os[llm]"     # Ollama + Anthropic providers (httpx, anthropic)
uv tool install "agent-os[api]"     # REST server + dashboard (fastapi, uvicorn)
uv tool install "agent-os[vector]"  # remote embeddings for semantic memory (httpx)
```

For development, `uv run --extra dev pytest -q` pulls in the test dependencies.

### Connect the real Claude API

The LLM abstraction defaults to an offline `MockLLM`. To run agents against
Claude, install the `[llm]` extra, set your key, and point the provider at
Anthropic — no other code changes:

```bash
uv tool install "agent-os[llm]"
export ANTHROPIC_API_KEY=sk-ant-...
export AGENTOS_LLM=anthropic                 # get_provider() now returns Claude
export AGENTOS_ANTHROPIC_MODEL=claude-opus-4-8   # optional; default
uv run --extra llm python examples/06_claude_agent.py
```

Or wire it explicitly in code:

```python
from agentos import Task, default_kernel
from agentos.llm import AnthropicProvider

k = default_kernel(llm=AnthropicProvider())      # uses ANTHROPIC_API_KEY
k.submit(Task(kind="llm", payload={"prompt": "Summarize agent-os in one line."}))
k.run()
```

The provider uses the official `anthropic` SDK, handles the `refusal` stop
reason, splits `system` messages, supports tool-calling + schema-forced
structured output, and reports token usage and USD cost per call. Switch to a
local model with `AGENTOS_LLM=ollama` instead.

## Examples

Runnable, pure-stdlib scripts in [`examples/`](examples/):

| File | Shows |
|---|---|
| `01_hello_pipeline.py` | boot builtins, run a fan-out/reduce pipeline |
| `02_priorities_and_deps.py` | priority ordering + a dependency chain |
| `03_retries_and_timeout.py` | a flaky task that recovers, and one that times out |
| `04_tools_and_guardrails.py` | a schema-validated tool + secret block / PII redact |
| `05_custom_agent.py` | a custom agent, shared memory, and a spawned follow-up |
| `06_claude_agent.py` | an LLM agent backed by the **real Claude API** (falls back to MockLLM without a key) |

```bash
uv run python examples/01_hello_pipeline.py
```

## Scheduling rules

- A task runs when it is `PENDING` and every dependency is `COMPLETED`.
- Among runnable tasks, higher `priority` wins; ties break FIFO (deterministic).
- If a dependency `FAILED`, the dependent task becomes `BLOCKED` and never runs.
- A handler raising an exception fails only its own task — the OS keeps going.
- Retries re-invoke the handler with exponential backoff; a timeout abandons the
  attempt; exhausted tasks land in the dead-letter queue.

## Tasks file format (`agentos run`)

A JSON list; `deps` reference other tasks by their index in the list:

```json
[
  {"kind": "calc", "payload": {"op": "+", "a": 2, "b": 3}},
  {"kind": "echo", "payload": {"message": "done"}, "deps": [0]}
]
```

## Feature checklist

- [x] Priority + dependency scheduler with cycle detection
- [x] Concurrent run loop (`max_workers`), deterministic single-worker mode
- [x] Per-task retries with exponential backoff + jitter
- [x] Per-attempt wall-clock timeouts and a dead-letter queue
- [x] Layered memory: blackboard, short-term, long-term, semantic
- [x] Schema-validated, rate-limited tools with error isolation
- [x] LLM abstraction: Mock / Ollama / Anthropic + fallback + cost accounting
- [x] Runtime task decomposition (spawn / delegate) for pipelines and DAGs
- [x] Checkpoint / resume of the task table
- [x] Human-in-the-loop approval gate
- [x] Guardrails: secret + injection blocking, PII redaction
- [x] Observability: typed events, OTel-flavored spans, metrics, per-task traces
- [x] Interfaces: Python SDK, CLI, REST API, dashboard

## Development

```bash
uv run --extra dev pytest -q                 # core + offline suites
uv run --extra dev --extra api pytest -q     # include the REST server tests
```

## License

MIT
