"""MCP server exposing the Xiangqi duel arena to external agents.

Contract (see docs/mcp-agent-contract.md):
  - LLM-plays-the-pieces (not a heuristic / engine bot)
  - Players only operate through this tool set; they own thinking and memory
  - Long thinks and tool timeouts are not a penalty; no engine/MCTS/script play
  - Moves / preview / resign need a seat_token; claim / open inject player_brief + rules
  - Dual external defaults to auto_claim=false — each side claim_seat for its own token

Tools: get_player_brief, list_presets, list_games, create_duel, challenge_preset,
claim_seat, wait_my_turn, get_board, get_legal_moves, preview, preview_reset,
get_threats, submit_move, resign, get_result.

This process is a stateless proxy. Game state lives in server.py.
    .venv/bin/python mcp_server.py
    .venv/bin/python mcp_server.py --http 8765
XIANGQI_API_BASE defaults to http://127.0.0.1:8000. Moves are ICCS (h2e2).
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Optional

import httpx
from fastmcp import FastMCP
from starlette.responses import JSONResponse

import mcp_contract
from agent.locale import is_zh

API_BASE = os.environ.get("XIANGQI_API_BASE", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
SEAT_HEADER = "X-Xiangqi-Seat-Token"
MCP_TOOL_HEADER = "X-Xiangqi-Mcp-Tool"

# Test injection: set to httpx.MockTransport so _api stays offline.
_TEST_TRANSPORT: Optional[httpx.AsyncBaseTransport] = None


def _L(en: str, zh: str) -> str:
    return zh if is_zh() else en

mcp = FastMCP("xiangqi-duel")


async def _api(
    method: str,
    path: str,
    *,
    json: Any = None,
    timeout: float = 30.0,
    seat_token: Optional[str] = None,
    mcp_tool: Optional[str] = None,
) -> tuple[int, Any]:
    """One REST call to the duel server; returns (status_code, payload)."""
    headers: dict[str, str] = {}
    if seat_token:
        headers[SEAT_HEADER] = str(seat_token)
    if mcp_tool:
        headers[MCP_TOOL_HEADER] = str(mcp_tool)
    async with httpx.AsyncClient(
        base_url=API_BASE, timeout=timeout, transport=_TEST_TRANSPORT
    ) as client:
        resp = await client.request(method, path, json=json, headers=headers or None)
        try:
            payload: Any = resp.json()
        except ValueError:
            payload = {"detail": resp.text}
        return resp.status_code, payload


def _err(payload: Any, status: int) -> str:
    """Extract a human/agent-readable error message from a REST error body."""
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict):
            return str(detail.get("error") or detail)
        return str(detail)
    return f"HTTP {status}"


def _error_class(payload: Any, status: int) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail", payload)
        if isinstance(detail, dict) and detail.get("error_class"):
            return str(detail["error_class"])
        if payload.get("error_class"):
            return str(payload["error_class"])
    if status in {401, 403}:
        return "auth"
    if status >= 500:
        return "infra"
    return "rules"


def _fail(payload: Any, status: int, **extra: Any) -> dict:
    out: dict[str, Any] = {
        "ok": False,
        "error": _err(payload, status),
        "error_class": _error_class(payload, status),
    }
    out.update(extra)
    return out


_NON_PLAYING_PHASES = ("waiting", "paused", "finished", "interrupted")


def _phase_label(status: str) -> str:
    labels = {
        "waiting": _L("pre-game", "预盘/开局前"),
        "paused": _L("review/pause", "复盘/暂停"),
        "finished": _L("post-game", "终局复盘"),
        "interrupted": _L("interrupted", "中断"),
    }
    return labels.get(status, status)


def _non_playing_hint(status: str) -> str:
    """What an agent should do when the move tool is gated off."""
    if status == "paused":
        return _L(
            "Review/pause: do not move. Keep wait_my_turn until resume returns playing.",
            "复盘/暂停：不要走子。继续 wait_my_turn，直到 resume 回到 playing。",
        )
    if status == "waiting":
        return _L(
            "Pre-game: do not move. After kickoff, wait_my_turn reports "
            "waiting_done_reason=my_turn.",
            "预盘/开局前：不要走子。开局后 wait_my_turn 会回报 waiting_done_reason=my_turn。",
        )
    if status == "finished":
        return _L(
            "Game over; move tools are off. Call get_result.",
            "对局已结束，走子工具已关闭。请调用 get_result。",
        )
    if status == "interrupted":
        return _L(
            "Game interrupted; move tools are off. wait_my_turn or tell the referee.",
            "对局中断，走子工具已关闭。继续 wait_my_turn 或告知裁判。",
        )
    return _L(
        "Not playing: do not move. wait_my_turn until the game is live.",
        "非对弈阶段：不要走子。继续 wait_my_turn 直到对局开始。",
    )


def _board_text(fen: str) -> str:
    from xiangqi import Board

    try:
        return Board(fen).to_text()
    except Exception:
        return ""


def _player_payload(spec: dict) -> dict:
    """Map an MCP-side side spec to a server PlayerConfig payload."""
    spec = spec or {}
    kind = str(spec.get("type") or "external").strip().lower()
    if kind == "preset":
        return {"type": "llm", "preset": str(spec.get("preset") or "").strip()}
    body: dict[str, Any] = {"type": kind}
    name = str(spec.get("name") or "").strip()
    if name:
        body["name"] = name
    if spec.get("preset"):
        body["preset"] = spec["preset"]
    return body


async def _claim(game_id: str, side: str, name: Optional[str] = None) -> dict:
    body: dict[str, Any] = {"side": side}
    if name:
        body["name"] = name
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/claim-seat",
        json=body,
        mcp_tool="claim_seat",
    )
    if status != 200:
        return _fail(payload, status, game_id=game_id, side=side)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid claim-seat response", "error_class": "infra"}
    # Server embeds contract; re-attach locally so mocks / older servers still get rules.
    return mcp_contract.attach_contract(
        {
            "ok": True,
            "game_id": payload.get("game_id") or game_id,
            "side": payload.get("side") or side,
            "seat_token": payload.get("seat_token"),
            "status": payload.get("status"),
            "turn": payload.get("turn"),
        }
    )


async def _start_new_duel(
    fen: str,
    red: dict,
    black: dict,
    *,
    auto_claim: Optional[bool] = None,
) -> dict:
    status, payload = await _api(
        "POST",
        "/api/game/create",
        json={
            "fen": fen,
            "red": _player_payload(red),
            "black": _player_payload(black),
        },
    )
    if status != 200:
        return _fail(payload, status)
    game_id = payload.get("game_id")
    start_status, start_payload = await _api(
        "POST", f"/api/game/{game_id}/start", json={}
    )
    if start_status != 200:
        return {
            **_fail(start_payload, start_status, game_id=game_id),
            "hint": _L(
                f"Game created but not started; retry POST /api/game/{game_id}/start later",
                f"对局已创建但未启动，可稍后重试 POST /api/game/{game_id}/start",
            ),
        }

    red_type = str((_player_payload(red) or {}).get("type") or "").lower()
    black_type = str((_player_payload(black) or {}).get("type") or "").lower()
    external_sides = [
        s for s, t in (("red", red_type), ("black", black_type)) if t == "external"
    ]
    # Unified default: dual external → agents claim themselves; single external → auto claim.
    if auto_claim is None:
        auto_claim = len(external_sides) <= 1

    out: dict[str, Any] = {
        "ok": True,
        "game_id": game_id,
        "fen": fen,
        "auto_claim": bool(auto_claim),
        "join_playbook": "referee_open__agents_claim",
    }
    out.update(mcp_contract.contract_payload(include_brief=True))
    if start_payload.get("local"):
        out["warning"] = _L(
            "Neither side is llm/external; this is a local game — the client must drive it",
            "双方都不是 llm/external，此局为 local 局，需客户端自行驱动",
        )

    out["external_sides"] = external_sides
    out["join_hint"] = (
        "Unified sit: each side claim_seat(own side) and store only that seat_token. "
        "Same script for sub-agents / multi-session / remote MCP."
    )

    if not auto_claim:
        out["open_seats"] = [
            {"game_id": game_id, "side": s, "claimed": False, "joinable": True}
            for s in external_sides
        ]
        out["hint"] = "No auto-claim: each side claim_seat (recommended for dual external / multi-host)"
        return out

    tokens: dict[str, str] = {}
    names = {"red": (red or {}).get("name"), "black": (black or {}).get("name")}
    for side in external_sides:
        claimed = await _claim(game_id, side, names.get(side))
        if not claimed.get("ok"):
            return {
                **claimed,
                "game_id": game_id,
                "hint": _L(
                    f"Game started but {side} claim failed",
                    f"对局已启动但 {side} claim 失败",
                ),
            }
        tokens[side] = claimed["seat_token"]
        out[f"{side}_seat_token"] = claimed["seat_token"]
    if len(tokens) == 1:
        side, tok = next(iter(tokens.items()))
        out["seat_token"] = tok
        out["my_side"] = side
    elif len(tokens) > 1:
        out["warning"] = _L(
            "dual external with auto_claim=true returns both seat_tokens; "
            "independent hosts should omit auto_claim or pass false, then each claim_seat",
            "dual external 且 auto_claim=true 会返回双方 seat_token；"
            "独立宿主请省略 auto_claim 或传 false，再各自 claim_seat",
        )
    return out


@mcp.tool
async def get_player_brief() -> dict:
    """Fetch the LLM-duel contract and player brief (no seat_token). Call before sit/open."""
    return {"ok": True, **mcp_contract.contract_payload(include_brief=True)}


@mcp.tool
async def list_presets() -> dict:
    """List challengeable internal seats (config.yaml -> models:[]). No seat_token."""
    status, payload = await _api("GET", "/api/presets")
    if status != 200:
        return _fail(payload, status)
    return {"ok": True, "presets": payload.get("presets", [])}


@mcp.tool
async def list_games() -> dict:
    """List live games and external seats (claimed/joinable). No seat_token."""
    status, payload = await _api("GET", "/api/games")
    if status != 200:
        return _fail(payload, status)
    games_out: list[dict] = []
    open_seats: list[dict] = []
    joinable_seats: list[dict] = []
    for g in payload.get("games", []):
        gid = g.get("id")
        entry = dict(g)
        st_status, st = await _api("GET", f"/api/game/{gid}/state")
        if st_status == 200:
            sides = st.get("sides") or {}
            entry.update({
                "turn": st.get("turn"),
                "sides": sides,
                "fen": st.get("fen"),
                "reason": st.get("reason"),
            })
            if st.get("status") == "playing":
                for s in ("red", "black"):
                    info = sides.get(s) or {}
                    if info.get("type") != "external":
                        continue
                    claimed = bool(info.get("claimed"))
                    seat = {
                        "game_id": gid,
                        "side": s,
                        "claimed": claimed,
                        "joinable": not claimed,
                        "turn": st.get("turn"),
                        "waiting_for_side": st.get("turn") == s,
                        "name": info.get("name"),
                    }
                    open_seats.append(seat)
                    if not claimed:
                        joinable_seats.append(seat)
        elif g.get("external_seats"):
            # Fallback: /api/games already carries external_seats
            for es in g.get("external_seats") or []:
                if g.get("status") != "playing":
                    continue
                seat = {
                    "game_id": gid,
                    "side": es.get("side"),
                    "claimed": bool(es.get("claimed")),
                    "joinable": bool(es.get("joinable", not es.get("claimed"))),
                    "turn": g.get("turn"),
                    "waiting_for_side": g.get("turn") == es.get("side"),
                    "name": es.get("name"),
                }
                open_seats.append(seat)
                if seat["joinable"]:
                    joinable_seats.append(seat)
        games_out.append(entry)
    return {
        "ok": True,
        "games": games_out,
        "open_seats": open_seats,
        "joinable_seats": joinable_seats,
        "hint": (
            "Remote sit: pick a side from joinable_seats → claim_seat; "
            "claim_seat on an already-claimed seat rotates the token."
        ),
    }


@mcp.tool
async def create_duel(
    red: dict,
    black: dict,
    fen: str = DEFAULT_FEN,
    auto_claim: Optional[bool] = None,
) -> dict:
    """Open a game. Dual external does not auto-claim (each side claim_seat);
    a single external auto-claims. Pass auto_claim to override."""
    return await _start_new_duel(fen, red, black, auto_claim=auto_claim)


@mcp.tool
async def challenge_preset(
    preset: str, my_side: str = "red", my_name: Optional[str] = None
) -> dict:
    """External agent challenges an internal llm preset; auto-claims and returns seat_token."""
    side = str(my_side or "red").strip().lower()
    if side not in {"red", "black"}:
        return {"ok": False, "error": 'my_side must be "red" or "black"', "error_class": "rules"}
    status, payload = await _api("GET", "/api/presets")
    if status != 200:
        return _fail(payload, status)
    presets = {
        str(p.get("name")): p
        for p in (payload.get("presets") or [])
        if isinstance(p, dict) and p.get("name")
    }
    info = presets.get(str(preset or "").strip())
    if not info:
        return {"ok": False, "error": f"unknown preset: {preset}", "error_class": "rules"}
    ptype = str(info.get("type") or "llm").strip().lower()
    if ptype != "llm":
        return {
            "ok": False,
            "error": (
                f"preset {preset!r} has type={ptype!r}; "
                "challenge_preset only challenges type=llm seats"
            ),
            "error_class": "rules",
        }
    me: dict[str, Any] = {"type": "external"}
    if my_name:
        me["name"] = my_name
    opponent = {"type": "preset", "preset": preset}
    if side == "black":
        red, black = opponent, me
    else:
        red, black = me, opponent
    return await _start_new_duel(DEFAULT_FEN, red, black)


@mcp.tool
async def claim_seat(
    game_id: str, side: str, name: Optional[str] = None
) -> dict:
    """Issue or rotate a seat_token for an external seat; response includes player_brief and rules."""
    body_side = str(side or "").strip().lower()
    if body_side not in {"red", "black"}:
        return {"ok": False, "error": 'side must be "red" or "black"', "error_class": "rules"}
    return await _claim(game_id, body_side, name)


@mcp.tool
async def wait_my_turn(
    game_id: str,
    side: str,
    seat_token: str,
    timeout_sec: float = 1800.0,
) -> dict:
    """Block until it is this side's turn. Timeout is normal (ok=true +
    waiting_done_reason=timeout), not an error; call again. my_turn includes the board."""
    body_side = str(side or "").strip().lower()
    if body_side not in {"red", "black"}:
        return {"ok": False, "error": 'side must be "red" or "black"', "error_class": "rules"}
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    try:
        wait_sec = float(timeout_sec)
    except (TypeError, ValueError):
        return {"ok": False, "error": "timeout_sec must be a number", "error_class": "rules"}
    if wait_sec <= 0:
        return {"ok": False, "error": "timeout_sec must be > 0", "error_class": "rules"}
    http_timeout = min(wait_sec, 3600.0) + 30.0
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/wait-turn",
        json={"side": body_side, "timeout_sec": wait_sec},
        timeout=http_timeout,
        seat_token=seat_token,
        mcp_tool="wait_my_turn",
    )
    if status != 200:
        return _fail(payload, status)
    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid wait-turn response", "error_class": "infra"}
    fen = payload.get("fen") or ""
    out = {
        "ok": bool(payload.get("ok", True)),
        "game_id": payload.get("game_id") or game_id,
        "side": body_side,
        "waiting_done_reason": payload.get("waiting_done_reason"),
        "status": payload.get("status"),
        "turn": payload.get("turn"),
        "winner": payload.get("winner"),
        "reason": payload.get("reason"),
        "fen": fen,
        "board": payload.get("board") or _board_text(fen),
        "move_count": payload.get("move_count"),
        "last_move": payload.get("last_move"),
        "sides": payload.get("sides"),
        "activity": payload.get("activity"),
    }
    if payload.get("error_class"):
        out["error_class"] = payload["error_class"]
    if payload.get("suggested_retry_sec"):
        out["suggested_retry_sec"] = payload["suggested_retry_sec"]
    if out["waiting_done_reason"] == "my_turn":
        out["hint"] = _L(
            "Your turn: this response already has fen/board — no extra get_board. "
            "Call get_legal_moves then submit_move (preview optional).",
            "轮到你了：本响应已含最新棋盘（fen/board），无需再调 get_board；"
            "get_legal_moves 后 submit_move（可先 preview）",
        )
        out["rules_reminder"] = (
            mcp_contract.MY_TURN_REMINDER_ZH
            if is_zh()
            else mcp_contract.MY_TURN_REMINDER_EN
        )
    elif out["waiting_done_reason"] == "timeout":
        opp = "black" if body_side == "red" else "red"
        act = (payload.get("activity") or {}) if isinstance(payload.get("activity"), dict) else {}
        think = act.get("current_side_thinking_sec")
        last_tool = (act.get("side_last_tool_sec_ago") or {}).get(opp)
        bits = [
            _L(
                "Wait timeout is not an error and not a resignation: "
                "long opponent thinks (30+ minutes) are normal",
                "等待超时≠错误≠对方退出：对方长考半小时甚至更久属正常节奏",
            )
        ]
        if isinstance(think, (int, float)):
            bits.append(
                _L(
                    f"Opponent has been thinking about {int(think // 60)} minutes",
                    f"对方已思考约 {int(think // 60)} 分钟",
                )
            )
        if isinstance(last_tool, (int, float)):
            bits.append(
                _L(
                    f"Opponent last tool call about {int(last_tool // 60)} minutes ago",
                    f"对方最近一次工具调用约 {int(last_tool // 60)} 分钟前",
                )
            )
        bits.append(
            _L(
                "Only the server ends the game (status=finished); keep wait_my_turn. "
                "Never quit or resign because the other side is quiet.",
                "对局只能由服务器裁决终局（status=finished）；请继续 wait_my_turn 续等，"
                "任何时候都不要因对方沉默而主动退出或认输",
            )
        )
        retry = payload.get("suggested_retry_sec")
        if isinstance(retry, (int, float)):
            bits.append(
                _L(
                    f"Suggested next wait about {int(retry)} seconds",
                    f"建议下次 wait 约 {int(retry)} 秒再查",
                )
            )
        out["hint"] = ("; " if not is_zh() else "；").join(bits)
    else:
        reason = out["waiting_done_reason"]
        if reason in _NON_PLAYING_PHASES:
            phase = _phase_label(reason)
            out["hint"] = _L(
                f"Now in the {phase} phase (status={out.get('status') or reason}); "
                f"submit_move is disabled; do not move; {_non_playing_hint(reason)}",
                f"当前为{phase}阶段（status={out.get('status') or reason}），"
                f"submit_move 已禁用，不要尝试走子；{_non_playing_hint(reason)}",
            )
    return out


@mcp.tool
async def get_board(game_id: str, seat_token: str) -> dict:
    """Observe the position (player channel, needs seat_token; audited)."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, st = await _api(
        "GET",
        f"/api/game/{game_id}/state",
        seat_token=seat_token,
        mcp_tool="get_board",
    )
    if status != 200:
        return _fail(st, status)
    fen = st.get("fen") or ""
    hist = st.get("move_history") or []
    tail = [
        {
            "number": m.get("number"),
            "side": m.get("side"),
            "move": m.get("move"),
            "move_zh": m.get("move_zh"),
            "captured": m.get("captured"),
        }
        for m in hist[-5:]
    ]
    turn = st.get("turn")
    sides = st.get("sides") or {}
    out = {
        "ok": True,
        "game_id": st.get("game_id"),
        "seat": st.get("seat"),
        "fen": fen,
        "board": st.get("board") or _board_text(fen),
        "turn": turn,
        "status": st.get("status"),
        "winner": st.get("winner"),
        "reason": st.get("reason"),
        "sides": sides,
        "last_moves": tail,
        "move_count": len(hist),
        "activity": st.get("activity"),
    }
    turn_side = sides.get(turn) or {}
    if turn_side.get("type") == "external":
        out["hint"] = _L(
            f"It is {turn}'s turn: preview, or get_legal_moves then submit_move",
            f"当前轮到 {turn}：可用 preview 预演，或 get_legal_moves 后 submit_move",
        )
    return out


@mcp.tool
async def get_legal_moves(game_id: str, seat_token: str) -> dict:
    """Legal moves (token channel: ObserveSession work board + annotated)."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, payload = await _api(
        "GET",
        f"/api/game/{game_id}/legal_moves",
        seat_token=seat_token,
        mcp_tool="get_legal_moves",
    )
    if status != 200:
        return _fail(payload, status)
    return {
        "ok": True,
        "game_id": payload.get("game_id"),
        "seat": payload.get("seat"),
        "turn": payload.get("turn"),
        "fen": payload.get("fen"),
        "board": payload.get("board") or "",
        "legal_moves": list(payload.get("legal_moves") or []),
        "legal_count": int(payload.get("legal_count") or 0),
        "annotated": payload.get("annotated"),
        "path": payload.get("path") or [],
        "status": payload.get("status"),
        "live_fen": payload.get("live_fen"),
    }


@mcp.tool
async def preview(game_id: str, seat_token: str, move: str) -> dict:
    """Preview one ply on the work board (does not change live). The work board
    is cumulative: after a preview the turn flips and you may preview the
    reply. preview_reset restores the live position. Success includes
    is_checkmate / is_stalemate / work_turn; illegal moves return reason + hint."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/preview",
        json={"move": str(move or "").strip()},
        seat_token=seat_token,
        mcp_tool="preview",
    )
    if status != 200:
        return _fail(payload, status)
    out = dict(payload) if isinstance(payload, dict) else {"raw": payload}
    out.setdefault("ok", True)
    return out


@mcp.tool
async def preview_reset(game_id: str, seat_token: str) -> dict:
    """Reset the work board back to the live position."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/preview_reset",
        json={},
        seat_token=seat_token,
        mcp_tool="preview_reset",
    )
    if status != 200:
        return _fail(payload, status)
    out = dict(payload) if isinstance(payload, dict) else {"raw": payload}
    out.setdefault("ok", True)
    return out


@mcp.tool
async def get_threats(game_id: str, seat_token: str) -> dict:
    """Work-board threat summary (rules layer; no engine score)."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, payload = await _api(
        "GET",
        f"/api/game/{game_id}/threats",
        seat_token=seat_token,
        mcp_tool="get_threats",
    )
    if status != 200:
        return _fail(payload, status)
    out = dict(payload) if isinstance(payload, dict) else {"raw": payload}
    out.setdefault("ok", True)
    return out


@mcp.tool
async def submit_move(
    game_id: str,
    move: str,
    seat_token: str,
    side: Optional[str] = None,
    note: Optional[str] = None,
    wait_opponent: bool = False,
    wait_timeout_sec: float = 1800.0,
) -> dict:
    """Submit a move (external seats only). Gated off in review/pre-game:
    only status=playing may move (waiting/paused/finished/interrupted →
    error_class=state). Illegal moves return legal_moves + reason + error_class=rules."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    st_status, st = await _api(
        "GET",
        f"/api/game/{game_id}/state",
        seat_token=seat_token,
        mcp_tool="get_board",
    )
    if st_status != 200:
        return _fail(st, st_status)
    st_status_now = str(st.get("status") or "").strip().lower() if isinstance(st, dict) else ""
    if st_status_now and st_status_now not in {"playing"}:
        phase = _phase_label(st_status_now)
        return {
            "ok": False,
            "error": _L(
                f"submit_move is disabled (status={st_status_now}, {phase} phase). "
                "Move tools are only available when status=playing and it is your turn.",
                f"submit_move 已禁用（当前 status={st_status_now}，{phase}阶段不走子）。"
                "走子工具只在 status=playing 且轮到你时可用。",
            ),
            "error_class": "state",
            "status": st_status_now,
            "game_id": game_id,
            "turn": st.get("turn") if isinstance(st, dict) else None,
            "hint": _non_playing_hint(st_status_now),
            "board": st.get("board") if isinstance(st, dict) else None,
            "fen": st.get("fen") if isinstance(st, dict) else None,
        }
    if side:
        body_side = str(side).strip().lower()
    else:
        body_side = str(st.get("turn") or "").lower()
    prev_count = len(st.get("move_history") or [])
    body: dict[str, Any] = {"side": body_side, "move": str(move or "").strip()}
    if note:
        body["note"] = str(note).strip()
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/move",
        json=body,
        seat_token=seat_token,
        mcp_tool="submit_move",
    )
    if status != 200:
        out = _fail(payload, status)
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            for key in (
                "error",
                "reason",
                "reason_code",
                "legal_moves",
                "legal_count",
                "board",
                "fen",
                "error_class",
                "status",
            ):
                if key in detail:
                    out[key] = detail[key]
        if out.get("error_class") == "state" and out.get("status"):
            # 竞态兜底：预检后、落子前对局被暂停/终局，仍给出阶段指引
            out.setdefault("hint", _non_playing_hint(str(out["status"])))
        return out
    pending = payload.get("pending") if isinstance(payload, dict) else None
    state_after = None
    applied = False
    for _ in range(15):
        await asyncio.sleep(0.1)
        _, after = await _api(
            "GET",
            f"/api/game/{game_id}/state",
            seat_token=seat_token,
            mcp_tool="get_board",
        )
        if not isinstance(after, dict) or not after.get("fen"):
            break
        hist = after.get("move_history") or []
        state_after = {
            "fen": after.get("fen"),
            "turn": after.get("turn"),
            "status": after.get("status"),
            "move_count": len(hist),
            "last_move": hist[-1] if hist else None,
        }
        if len(hist) > prev_count:
            applied = True
            break
        if after.get("status") in {"finished", "interrupted", "paused"}:
            break
    if not applied:
        return {
            "ok": False,
            "error": (
                "move accepted by server but not yet applied "
                "(paused/interrupted or still pending); "
                "check get_board — do not resubmit while pending"
            ),
            "error_class": "state",
            "accepted": body["move"],
            "side": body_side,
            "pending": pending,
            "applied": False,
            "state_after": state_after,
        }
    result: dict[str, Any] = {
        "ok": True,
        "accepted": body["move"],
        "side": body_side,
        "pending": pending,
        "applied": True,
        "state_after": state_after,
    }
    if wait_opponent:
        wait_out = await wait_my_turn(
            game_id, body_side, seat_token, timeout_sec=wait_timeout_sec
        )
        result["wait_after"] = wait_out
        if wait_out.get("waiting_done_reason") == "timeout":
            result["hint"] = _L(
                "Move applied; long opponent thinks are normal (silence ≠ quit). "
                "Timeout is not a penalty — keep wait_my_turn; do not leave or resign while waiting.",
                "着法已应用；对方长考属正常（沉默≠退出），等待超时不罚，"
                "请继续 wait_my_turn 续等，不要因等待而退出或认输",
            )
        elif wait_out.get("waiting_done_reason") == "my_turn":
            result["hint"] = _L(
                "Opponent moved; your turn again",
                "对方已走，又轮到你了",
            )
        elif wait_out.get("waiting_done_reason") in {
            "finished",
            "paused",
            "interrupted",
        }:
            result["hint"] = _L(
                f"Game status: {wait_out.get('waiting_done_reason')}",
                f"对局状态：{wait_out.get('waiting_done_reason')}",
            )
    return result


@mcp.tool
async def resign(game_id: str, seat_token: str, side: Optional[str] = None) -> dict:
    """Resign this seat only (needs seat_token)."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    body = {"side": side} if side else {}
    status, payload = await _api(
        "POST",
        f"/api/game/{game_id}/resign",
        json=body,
        seat_token=seat_token,
        mcp_tool="resign",
    )
    if status != 200:
        return _fail(payload, status)
    return {"ok": True, **payload}


@mcp.tool
async def get_result(game_id: str, seat_token: str) -> dict:
    """Result and full game (needs seat_token)."""
    if not (seat_token or "").strip():
        return {"ok": False, "error": "seat_token required", "error_class": "auth"}
    status, st = await _api(
        "GET",
        f"/api/game/{game_id}/state",
        seat_token=seat_token,
        mcp_tool="get_result",
    )
    if status != 200:
        return _fail(st, status)
    hist = st.get("move_history") or []
    moves = [
        {
            "number": m.get("number"),
            "side": m.get("side"),
            "move": m.get("move"),
            "move_zh": m.get("move_zh"),
            "captured": m.get("captured"),
            **({"note": m["note"]} if m.get("note") else {}),
        }
        for m in hist
    ]
    return {
        "ok": True,
        "game_id": st.get("game_id"),
        "status": st.get("status"),
        "winner": st.get("winner"),
        "reason": st.get("reason"),
        "moves": moves,
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({
        "status": "ok",
        "api_base": API_BASE,
        "contract_version": mcp_contract.CONTRACT_VERSION,
        "play_mode": mcp_contract.PLAY_MODE,
        "mcp_path": "/mcp",
        "remote_hint": (
            "Unified sit: dual external create_duel does not deal cards; each side "
            "claim_seat. Same script for sub-agents / remote. Smoke referee: "
            "scripts/mcp_llm_duel_smoke.py. XIANGQI_API_BASE must be reachable from "
            "the MCP process; remote clients connect to /mcp."
        ),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="xiangqi-llm-duel MCP server")
    parser.add_argument(
        "--http",
        nargs="?",
        const=8765,
        type=int,
        default=None,
        metavar="PORT",
        help="Start streamable HTTP (endpoint /mcp); default is stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind address (use 0.0.0.0 for remote)",
    )
    args = parser.parse_args()
    if args.http is not None:
        mcp.run(transport="http", host=args.host, port=args.http)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
