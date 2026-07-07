"""Scheduler: decides which pending task(s) run next.

A task is *runnable* when it is PENDING and all of its dependencies have
COMPLETED. Among runnable tasks, higher ``priority`` wins; ties break by
submission order (FIFO) for determinism. If any dependency FAILED/BLOCKED, the
task becomes BLOCKED and never runs.

``next_runnable`` returns one task (single-worker / deterministic mode);
``ready_batch`` returns every currently-runnable task ordered by priority (for
the concurrent kernel). Dependency cycles are rejected at submit time.
"""

from __future__ import annotations

from .task import Task, TaskStatus


class DependencyCycleError(ValueError):
    pass


class Scheduler:
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._seq: dict[int, int] = {}   # task id -> submission order
        self._next_seq = 0

    def add(self, task: Task) -> int:
        self._tasks[task.id] = task
        self._seq[task.id] = self._next_seq
        self._next_seq += 1
        if self._creates_cycle(task.id):
            # roll back the insertion before raising
            del self._tasks[task.id]
            del self._seq[task.id]
            raise DependencyCycleError(f"task {task.id} introduces a dependency cycle")
        return task.id

    def _creates_cycle(self, start: int) -> bool:
        """DFS over dependency edges; a back-edge to a task on the stack = cycle."""
        WHITE, GREY, BLACK = 0, 1, 2
        color: dict[int, int] = {}

        def visit(node: int) -> bool:
            color[node] = GREY
            task = self._tasks.get(node)
            for dep in (task.deps if task else []):
                c = color.get(dep, WHITE)
                if c == GREY:
                    return True
                if c == WHITE and dep in self._tasks and visit(dep):
                    return True
            color[node] = BLACK
            return False

        return visit(start)

    def get(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return [self._tasks[i] for i in sorted(self._tasks)]

    def _deps_state(self, task: Task) -> str:
        blocked = False
        for dep_id in task.deps:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status in (TaskStatus.FAILED, TaskStatus.BLOCKED):
                blocked = True
            elif dep.status != TaskStatus.COMPLETED:
                return "waiting"
        return "blocked" if blocked else "ready"

    def _rank(self, task: Task) -> tuple[int, int]:
        # higher priority first, then earlier submission (FIFO)
        return (-int(task.priority), self._seq[task.id])

    def ready_batch(self) -> list[Task]:
        """All currently-runnable tasks, best-first. Promotes blocked ones."""
        ready: list[Task] = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            state = self._deps_state(task)
            if state == "blocked":
                task.status = TaskStatus.BLOCKED
            elif state == "ready":
                ready.append(task)
        ready.sort(key=self._rank)
        return ready

    def next_runnable(self) -> Task | None:
        batch = self.ready_batch()
        return batch[0] if batch else None

    def has_pending(self) -> bool:
        return any(t.status == TaskStatus.PENDING for t in self._tasks.values())
