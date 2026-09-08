"""LangGraph-based LLM player agents."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LangGraphPlayer",
    "clear_game_threads",
    "make_thread_id",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import player as _player

        return getattr(_player, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
