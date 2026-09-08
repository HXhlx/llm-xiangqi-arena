"""Seat tokens, ObserveSession cursor, and MCP tool audit for external agents."""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Optional

from fastapi import HTTPException

from agent.observe_session import ObserveSession

MCP_LOG_DIR_NAME = "mcp"
SEAT_HEADER = "x-xiangqi-seat-token"
MCP_TOOL_HEADER = "x-xiangqi-mcp-tool"
RESULT_TRUNCATE = 2000
ARGS_TRUNCATE = 800


def mcp_log_dir(log_dir: str) -> str:
    path = os.path.join(log_dir, MCP_LOG_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def mcp_jsonl_path(log_dir: str, game_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(game_id))
    return os.path.join(mcp_log_dir(log_dir), f"{safe}.jsonl")


def game_has_external(game) -> bool:
    return game.red_config.type == "external" or game.black_config.type == "external"


def side_config(game, side: str):
    return game.red_config if side == "red" else game.black_config


def ensure_seat_maps(game) -> None:
    if not hasattr(game, "seat_tokens") or game.seat_tokens is None:
        game.seat_tokens = {"red": None, "black": None}
    if not hasattr(game, "observe") or game.observe is None:
        game.observe = {"red": None, "black": None}


def issue_seat_token(game, side: str) -> str:
    """Mint (or rotate) a seat token for an external side."""
    ensure_seat_maps(game)
    side = (side or "").strip().lower()
    if side not in {"red", "black"}:
        raise HTTPException(
            400,
            detail={"error": 'side must be "red" or "black"', "error_class": "rules"},
        )
    if side_config(game, side).type != "external":
        raise HTTPException(
            400,
            detail={
                "error": f"{side} is not an external seat",
                "error_class": "rules",
            },
        )
    token = secrets.token_urlsafe(32)
    game.seat_tokens[side] = token
    reset_observe(game, side)
    return token


def reset_observe(game, side: Optional[str] = None) -> None:
    """Rebuild ObserveSession from the live board (one side or both)."""
    ensure_seat_maps(game)
    sides = [side] if side in {"red", "black"} else ["red", "black"]
    for s in sides:
        if side_config(game, s).type == "external":
            game.observe[s] = ObserveSession(game.board)
        else:
            game.observe[s] = None


def get_observe(game, side: str) -> ObserveSession:
    ensure_seat_maps(game)
    obs = game.observe.get(side)
    if obs is None:
        obs = ObserveSession(game.board)
        game.observe[side] = obs
    return obs


def resolve_side_for_token(game, token: Optional[str]) -> Optional[str]:
    ensure_seat_maps(game)
    if not token:
        return None
    for side in ("red", "black"):
        if game.seat_tokens.get(side) and secrets.compare_digest(
            str(game.seat_tokens[side]), str(token)
        ):
            return side
    return None


def require_seat(
    game,
    *,
    token: Optional[str],
    expected_side: Optional[str] = None,
    mcp_tool: Optional[str] = None,
) -> tuple[str, str]:
    """Validate seat token; return (side, tool_name_for_audit)."""
    ensure_seat_maps(game)
    side = resolve_side_for_token(game, token)
    if side is None:
        raise HTTPException(
            401,
            detail={
                "error": "missing or invalid seat token",
                "error_class": "auth",
            },
        )
    if expected_side and side != expected_side.strip().lower():
        raise HTTPException(
            401,
            detail={
                "error": f"seat token is for {side}, not {expected_side}",
                "error_class": "auth",
            },
        )
    if side_config(game, side).type != "external":
        raise HTTPException(
            400,
            detail={"error": f"{side} is not external", "error_class": "rules"},
        )
    tool = (mcp_tool or "").strip() or "rest_direct"
    return side, tool


def _truncate(value: Any, limit: int) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[: limit - 3] + "..."
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= limit:
        return value
    return text[: limit - 3] + "..."


def append_mcp_jsonl(log_dir: str, game_id: str, record: dict) -> None:
    path = mcp_jsonl_path(log_dir, game_id)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def audit_tool(
    game,
    *,
    log_dir: str,
    side: str,
    tool: str,
    phase: str,
    args: Any = None,
    result: Any = None,
    ok: Optional[bool] = None,
    error_class: Optional[str] = None,
) -> None:
    """Persist tool_call / tool_result to game.events and logs/mcp/{gid}.jsonl."""
    ensure_seat_maps(game)
    safe_args = _truncate(args, ARGS_TRUNCATE)
    safe_result = _truncate(result, RESULT_TRUNCATE)
    if phase == "call":
        try:
            game.__dict__.setdefault("side_last_tool_ts", {})[side] = time.time()
        except Exception:
            pass
        game.broadcast(
            "tool_call",
            {"side": side, "tool": tool, "args": safe_args or {}},
            persist=True,
        )
    elif phase == "result":
        payload: dict[str, Any] = {
            "side": side,
            "tool": tool,
            "result": safe_result if safe_result is not None else "",
        }
        if ok is not None:
            payload["ok"] = ok
        if error_class:
            payload["error_class"] = error_class
        game.broadcast("tool_result", payload, persist=True)
    record = {
        "ts": time.time(),
        "game_id": game.id,
        "side": side,
        "tool": tool,
        "phase": phase,
        "ok": ok,
        "error_class": error_class,
        "args": safe_args,
        "result": safe_result,
    }
    try:
        append_mcp_jsonl(log_dir, game.id, record)
    except OSError as exc:
        print(f"  [mcp-audit] write failed: {exc}")


def auth_http_exception_detail(exc: HTTPException) -> dict:
    detail = exc.detail
    if isinstance(detail, dict):
        return detail
    return {"error": str(detail), "error_class": "rules"}
