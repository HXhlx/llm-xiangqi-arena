"""Adapter for MiniMax-M3 Chat Completions.

Official thinking control is thinking.type = adaptive|disabled.
reasoning_effort is not a Chat Completions parameter and does not
tune MiniMax-M3 reasoning depth.
"""

from typing import Optional

from .base import BaseAdapter


class MiniMaxAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        return {
            "thinking": {"type": "adaptive" if enable_thinking else "disabled"},
        }
