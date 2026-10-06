from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class TaskStore(Protocol):
    """Outbound port: contract for async task management."""

    async def submit(self, task: T) -> str:
        """Submit a task and return its identifier."""
        ...

    async def status(self, task_id: str) -> str:
        """Return the status of a task by id."""
        ...

    async def result(self, task_id: str) -> T:
        """Return the result of a completed task by id."""
        ...
