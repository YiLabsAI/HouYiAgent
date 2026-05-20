"""AxisRegistry — register and look up IndexAxis implementations by name."""

from __future__ import annotations

from houyi.adapters.memory.recall.indexer.axis import IndexAxis


class AxisRegistry:
    def __init__(self) -> None:
        self._axes: dict[str, IndexAxis] = {}

    def register(self, axis: IndexAxis) -> None:
        name = axis.name
        if not name:
            raise ValueError("IndexAxis.name must be non-empty")
        self._axes[name] = axis

    def get(self, name: str) -> IndexAxis | None:
        return self._axes.get(name)

    def names(self) -> list[str]:
        return sorted(self._axes)

    def __contains__(self, name: str) -> bool:
        return name in self._axes

    def __len__(self) -> int:
        return len(self._axes)


__all__ = ["AxisRegistry"]
