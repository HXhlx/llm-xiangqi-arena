"""Audience-facing Chinese emotion line for a ply's thought (player voice)."""

from __future__ import annotations

import re
import time
from typing import Any

from openai import AsyncOpenAI

from prompt_registry import render_prompt

from .compress import estimate_text_tokens, estimate_tokens
from .rate_limit import acquire_outbound, record_outbound_tokens

DEFAULT_TIMEOUT = 180.0
EMOTION_MAX_CHARS = 18

_TOOL_SPEAK = re.compile(
    r"(?i)\b(?:make_move|preview|get_legal_moves|get_threats|get_board|"
    r"preview_reset|function|api|json)\b"
)
_ICCS = re.compile(r"\b[a-i][0-9][a-i][0-9]\b", re.I)
_WRAP_QUOTES = re.compile(r'^[\s「」『』""\'‘’]+|[\s「」『』""\'‘’]+$')
_BOARD_SITUATION = re.compile(
    r"(?:夹死|打死|困死|堵死|压死|封死|憋死|白吃|吃掉|兑掉|送掉|换掉|弃掉|丢掉)"
    r".{0,3}?(?:车|马|炮|兵|卒|相|象|士|将|帅)"
)


def build_thought_text(
    reasoning: str = "",
    thinking: str = "",
    assistant_content: str = "",
) -> str:
    parts = []
    seen: set[str] = set()
    for chunk in (reasoning, thinking, assistant_content):
        text = (chunk or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return "\n\n".join(parts).strip()


def is_emotion_line_ok(text: str) -> bool:
    """True when text already passes the emotion-line gate without rewrite."""
    gated = gate_emotion_line(text, truncate=False)
    return bool(gated) and gated == _normalize_emotion_raw(text)


def _normalize_emotion_raw(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Single line only.
    raw = raw.splitlines()[0].strip() if raw else ""
    raw = _WRAP_QUOTES.sub("", raw).strip()
    return raw


def gate_emotion_line(text: str, *, truncate: bool = True) -> str:
    """Normalize and gate a player emotion line.

    Rejects tool-speak / ICCS coordinates. Piece names are allowed so the
    line can stay grounded in the ply. Overlong lines are truncated when
    ``truncate`` is True; otherwise rejected as empty (used by
    ``is_emotion_line_ok``).
    """
    raw = _normalize_emotion_raw(text)
    if not raw:
        return ""
    if _TOOL_SPEAK.search(raw) or _ICCS.search(raw) or _BOARD_SITUATION.search(raw):
        return ""
    if len(raw) > EMOTION_MAX_CHARS:
        if not truncate:
            return ""
        raw = raw[:EMOTION_MAX_CHARS].rstrip()
    return raw


async def summarize_turn(
    *,
    api_base: str,
    api_key: str,
    model: str,
    move: str = "",
    move_zh: str = "",
    reasoning: str = "",
    thinking: str = "",
    assistant_content: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    preset: str | None = None,
    rpm: int | None = None,
    tpm: int | None = None,
) -> dict[str, Any]:
    """
    Sync (awaited) short Chinese emotion line for spectators.
    Does NOT take tool_rounds. Fail-open: returns summary_zh="" on error.
    """
    t0 = time.monotonic()
    thought = build_thought_text(reasoning, thinking, assistant_content)
    if not thought.strip():
        return {
            "summary_zh": "",
            "ok": False,
            "error": "empty_thought",
            "latency_ms": 0,
            "model": model,
        }

    user_prompt = render_prompt(
        "agent.turn_summary",
        "user",
        move=move or "—",
        move_zh=move_zh or "",
        thought_text=thought,
    )

    try:
        summary_msgs = [{"role": "user", "content": user_prompt}]
        bucket = (preset or "").strip() or model
        await acquire_outbound(
            bucket,
            tokens=estimate_tokens(summary_msgs, model),
            rpm=rpm,
            tpm=tpm,
        )
        async with AsyncOpenAI(
            api_key=api_key or "x",
            base_url=(api_base or "").rstrip("/") or None,
            timeout=timeout,
        ) as client:
            resp = await client.chat.completions.create(
                model=model,
                messages=summary_msgs,
                stream=False,
            )
            text = ((resp.choices[0].message.content or "") if resp.choices else "").strip()
            text = gate_emotion_line(text)
            if text:
                await record_outbound_tokens(
                    bucket, estimate_text_tokens(text, model), tpm=tpm
                )
            return {
                "summary_zh": text,
                "ok": bool(text),
                "error": None if text else "empty_or_rejected_output",
                "latency_ms": int((time.monotonic() - t0) * 1000),
                "model": model,
                "chars_in": len(thought),
                "chars_out": len(text),
            }
    except Exception as e:
        return {
            "summary_zh": "",
            "ok": False,
            "error": str(e)[:200],
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "model": model,
        }
