"""Write compressed OpenAI-style messages back into LangGraph state."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from .compress import checkpoint_messages_need_sync, without_orphan_tools


def openai_dicts_to_lc_messages(messages: list[dict[str, Any]]) -> list:
    """Rebuild LangChain messages from the compressed OpenAI request copy."""
    out: list = []
    for raw in messages:
        role = raw.get("role") or "user"
        content = raw.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            out.append(SystemMessage(content=content))
            continue
        if role == "assistant":
            tool_calls = raw.get("tool_calls") or []
            lc_tool_calls = [_lc_tool_call(tc) for tc in tool_calls if isinstance(tc, dict)]
            extra: dict[str, Any] = {}
            if tool_calls:
                extra["additional_kwargs"] = {"tool_calls": tool_calls}
            if lc_tool_calls:
                out.append(AIMessage(content=content, tool_calls=lc_tool_calls, **extra))
            else:
                out.append(AIMessage(content=content, **extra) if extra else AIMessage(content=content))
            continue
        if role == "tool":
            name = raw.get("name") or ""
            kwargs: dict[str, Any] = {
                "content": content,
                "tool_call_id": raw.get("tool_call_id") or "",
            }
            if name:
                kwargs["name"] = name
            out.append(ToolMessage(**kwargs))
            continue
        out.append(HumanMessage(content=content))
    return out


def _lc_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = tc.get("name") or fn.get("name") or ""
    args = tc.get("args", fn.get("arguments", {}))
    if isinstance(args, str):
        try:
            args_obj = json.loads(args) if args else {}
        except json.JSONDecodeError:
            args_obj = {}
    else:
        args_obj = args or {}
    return {
        "id": tc.get("id") or "call_0",
        "name": name,
        "args": args_obj,
        "type": "tool_call",
    }


def replace_checkpoint_messages(compressed: list[dict[str, Any]]) -> list:
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *openai_dicts_to_lc_messages(compressed)]


def merge_model_messages(
    raw_oai: list[dict[str, Any]],
    compressed_oai: list[dict[str, Any]],
    new_messages: list | None = None,
    *,
    replace: bool | None = None,
) -> list:
    """Replace checkpoint history when compress ran; then append this node's messages.

    ``replace`` overrides the default (rewrite when the compressed list differs).
    Pass False when the pre-compress archive failed so Redis keeps the full copy.
    """
    added = list(new_messages or [])
    cleaned = without_orphan_tools(compressed_oai)
    if replace is None:
        replace = checkpoint_messages_need_sync(raw_oai, cleaned)
    if not replace:
        return added
    return [*replace_checkpoint_messages(cleaned), *added]
