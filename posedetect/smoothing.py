"""Temporal smoothing so the on-screen label doesn't flicker frame to frame."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Hashable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class LabelSmoother(Generic[T]):
    """Majority vote over the last ``window`` labels."""

    def __init__(self, window: int = 8) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        self._history: deque[T] = deque(maxlen=window)

    def update(self, label: T) -> T:
        self._history.append(label)
        counts = Counter(self._history)
        best = max(counts.values())
        # On a tie, prefer the most recent label among the leaders.
        for candidate in reversed(self._history):
            if counts[candidate] == best:
                return candidate
        return label

    def reset(self) -> None:
        self._history.clear()
