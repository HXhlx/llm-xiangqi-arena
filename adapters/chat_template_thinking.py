"""Adapter for models that toggle thinking via chat_template_kwargs.

Used by dots3-note-prev (Dots Studio): enable_thinking bool inside
chat_template_kwargs; do not send thinking.* or reasoning_effort.
"""

from typing import Optional

from .base import BaseAdapter


class ChatTemplateThinkingAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
