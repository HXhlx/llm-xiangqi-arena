"""Adapter for Qwen / DashScope models."""

from typing import Optional

from .base import BaseAdapter


class QwenAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        body: dict = {"enable_thinking": enable_thinking}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return body
