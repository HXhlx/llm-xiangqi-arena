"""Context compression helpers for long Xiangqi agent conversations."""

from __future__ import annotations

from typing import Any, Optional

from openai import AsyncOpenAI

from prompt_registry import render_prompt

from .rate_limit import acquire_outbound, record_outbound_tokens

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


def estimate_tokens(messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
    """Estimate token count for OpenAI-style message dicts."""
    texts: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
        if m.get("tool_calls"):
            texts.append(str(m.get("tool_calls")))
        if m.get("name"):
            texts.append(str(m.get("name")))
    blob = "\n".join(texts)
    if tiktoken is not None:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(blob)) + 8 * len(messages)
    # heuristic fallback
    return max(1, len(blob) // 4 + 8 * len(messages))


def estimate_text_tokens(text: str, model: str = "gpt-4o") -> int:
    blob = (text or "").strip()
    if not blob:
        return 0
    return estimate_tokens([{"role": "assistant", "content": blob}], model)


def _tool_call_ids(message: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for tc in message.get("tool_calls") or []:
        if isinstance(tc, dict) and tc.get("id"):
            ids.add(str(tc["id"]))
    return ids


def without_orphan_tools(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop tool rows whose assistant parent is missing, and incomplete tool rounds."""
    offered: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            offered |= _tool_call_ids(message)
        elif message.get("role") == "tool" and message.get("tool_call_id"):
            answered.add(str(message["tool_call_id"]))
    incomplete = offered - answered

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            ids = _tool_call_ids(message)
            if ids & incomplete:
                continue
            seen |= ids
            out.append(message)
            continue
        if role == "tool":
            tid = str(message.get("tool_call_id") or "")
            if tid and tid in seen:
                out.append(message)
            continue
        out.append(message)
    return out


def checkpoint_messages_need_sync(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> bool:
    """True when compress/trim rewrote the conversation (not the same list)."""
    if after is before:
        return False
    return after != before


def should_compress(
    messages: list[dict[str, Any]],
    context_window: int,
    threshold_ratio: float = 0.9,
    model: str = "gpt-4o",
) -> bool:
    if context_window <= 0:
        return False
    limit = int(context_window * threshold_ratio)
    return estimate_tokens(messages, model) >= limit


def _split_keep_recent(
    messages: list[dict[str, Any]],
    keep_tokens: int,
    model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (old, recent) keeping approximately keep_tokens in recent from the end."""
    if not messages:
        return [], []
    # Always keep system messages at the front of recent if present at start of old? handled by caller.
    recent: list[dict[str, Any]] = []
    used = 0
    for msg in reversed(messages):
        cost = estimate_tokens([msg], model)
        if recent and used + cost > keep_tokens:
            break
        recent.append(msg)
        used += cost
    recent.reverse()
    old = messages[: len(messages) - len(recent)]
    return old, recent


SUMMARY_PREFIX = "[对局历史摘要]"


def _extract_old_summary(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """摘出已有的历史摘要消息，返回 (旧摘要文本, 剩余消息)。

    旧摘要是 user 角色、以 SUMMARY_PREFIX 开头。摘出后剩余消息才进入
    本次压缩分区，避免摘要被豁免堆积/被当作普通对话再次总结。
    """
    old_summary = ""
    rest: list[dict[str, Any]] = []
    for m in messages:
        if (
            m.get("role") == "user"
            and isinstance(m.get("content"), str)
            and m.get("content", "").startswith(SUMMARY_PREFIX)
        ):
            old_summary = m["content"][len(SUMMARY_PREFIX):].strip()
            continue
        rest.append(m)
    return old_summary, rest


async def compress_messages(
    messages: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    context_window: int,
    threshold_ratio: float = 0.9,
    keep_ratio: float = 0.2,
    timeout: float | None = None,
    preset: str | None = None,
    rpm: int | None = None,
    tpm: int | None = None,
) -> list[dict[str, Any]]:
    """Summarize older messages when over threshold; keep recent turns verbatim."""
    if not should_compress(messages, context_window, threshold_ratio, model):
        return messages

    keep_tokens = max(1024, int(context_window * keep_ratio))
    # Preserve leading system messages outside the compress window
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    # 摘出上一轮的历史摘要：它是可压缩区的一部分，本次要合并更新而非再次堆叠
    old_summary, non_system = _extract_old_summary(non_system)
    old, recent = _split_keep_recent(non_system, keep_tokens, model)
    if not old:
        # hard trim only
        return without_orphan_tools(system_msgs + recent)

    # Per-message and total transcript caps scale with context window.
    transcript_budget = max(8000, int(context_window * 0.4))
    per_msg_cap = max(1000, transcript_budget // 50)
    transcript_lines = []
    for m in old:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if m.get("tool_calls"):
            content = f"{content}\n[tool_calls]{m.get('tool_calls')}"
        if role == "tool":
            content = f"[tool {m.get('name','')}] {content}"
        transcript_lines.append(f"{role.upper()}: {str(content)[:per_msg_cap]}")
    transcript = "\n".join(transcript_lines)[:transcript_budget]

    summary_text = ""
    previous_hint = ""
    if old_summary:
        previous_hint = (
            "已有旧摘要（在它基础上合并更新，不要从零重写，"
            "不要丢失其中仍然有效的信息）：\n\n" + old_summary
        )
    compress_msgs = [
        {
            "role": "system",
            "content": render_prompt("agent.compress", "system"),
        },
        {
            "role": "user",
            "content": render_prompt(
                "agent.compress",
                "user",
                transcript=transcript,
                previous_hint=previous_hint,
            ),
        },
    ]
    bucket = (preset or "").strip() or model
    await acquire_outbound(
        bucket,
        tokens=estimate_tokens(compress_msgs, model),
        rpm=rpm,
        tpm=tpm,
    )
    try:
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base.rstrip("/"),
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        async with AsyncOpenAI(**client_kwargs) as client:
            resp = await client.chat.completions.create(
                model=model,
                messages=compress_msgs,
                stream=False,
            )
            summary_text = (resp.choices[0].message.content or "").strip()
    except Exception:
        summary_text = ""
    if summary_text:
        await record_outbound_tokens(
            bucket, estimate_text_tokens(summary_text, model), tpm=tpm
        )

    if not summary_text:
        # hard drop old messages
        return without_orphan_tools(system_msgs + recent)

    # 摘要必须放 user 角色：system 会被下次压缩豁免，导致摘要只增不减、无限堆积
    summary_msg = {
        "role": "user",
        "content": f"{SUMMARY_PREFIX}\n{summary_text}",
    }
    return without_orphan_tools(system_msgs + [summary_msg] + recent)


def hard_trim(
    messages: list[dict[str, Any]],
    context_window: int,
    keep_ratio: float = 0.5,
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    keep_tokens = max(1024, int(context_window * keep_ratio))
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]
    _, recent = _split_keep_recent(non_system, keep_tokens, model)
    return without_orphan_tools(system_msgs + recent)
