"""Adapter for Doubao / Volcengine models."""

from typing import Optional

from .base import BaseAdapter


class DoubaoAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        body: dict = {
            "thinking": {"type": "enabled" if enable_thinking else "disabled"},
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return body
