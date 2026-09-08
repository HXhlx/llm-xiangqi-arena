"""
FastAPI backend for Xiangqi LLM Duel (LLM vs LLM Chinese chess arena).
Manages game sessions and proxies LLM / engine interactions.
Supports human, random, LLM, AI, and Pikafish player types.
"""

import asyncio
import uuid
import json
import time
import random
import re
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

import os
import yaml

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from xiangqi import Board
from agent import LangGraphPlayer, clear_game_threads
from agent.compress_settings import (
    resolve_compress_threshold_ratio,
    resolve_context_window,
    resolve_max_input_tokens,
)
from agent.observe import legal_moves_annotated, threats as observe_threats
from agent.rate_limit import close_rate_limiter, init_rate_limiter
from prompt_registry import get_default_prompt_name, get_prompt_profile, resolve_prompt_name
from pikafish_manager import (
    DEFAULT_ENGINE_FILENAME,
    DEFAULT_ENGINE_PATH,
    PikafishEvaluator,
)
import external_mcp
import mcp_contract

SeatTokenHdr = Annotated[Optional[str], Header()]
McpToolHdr = Annotated[Optional[str], Header()]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(external_mcp.mcp_log_dir(LOG_DIR), exist_ok=True)
DEFAULT_ENGINE_RELATIVE_PATH = os.path.join("pikafish", DEFAULT_ENGINE_FILENAME)


# --- Config loading ---

def load_app_config() -> dict:
    """Load config.yaml and return a normalized dict."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_model_presets() -> list[dict]:
    """Load model presets from config.yaml."""
    data = load_app_config()
    if "models" not in data:
        return []
    try:
        models = data["models"]
        if not isinstance(models, list):
            return []
        result = []
        for m in models:
            if isinstance(m, dict) and "name" in m:
                result.append({
                    "name": m["name"],
                    "type": (str(m.get("type") or "").strip() or "llm"),
                    "display_name": (m.get("display_name") or m.get("label") or "").strip() or None,
                    "api_base": m.get("api_base", ""),
                    "api_key": str(m.get("api_key") or "").strip(),
                    "model": m.get("model", m["name"]),
                    "prompt_name": resolve_prompt_name(m.get("prompt_name"), m.get("prompt_lang")),
                    "enable_thinking": m.get("enable_thinking", True),
                    "max_completion_tokens": m.get("max_completion_tokens", m.get("max_output_tokens", 8192)),
                    "context_window": m.get("context_window"),
                    "max_input_tokens": m.get("max_input_tokens"),
                    "compress_threshold_ratio": m.get("compress_threshold_ratio"),
                    "compress_model": m.get("compress_model"),
                    "enable_reasoning_effort": m.get("enable_reasoning_effort", False),
                    "reasoning_effort": m.get("reasoning_effort"),
                    "rpm": m.get("rpm"),
                    "tpm": m.get("tpm"),
                })
        return result
    except Exception:
        return []


def _normalize_engine_path(path: Optional[str], default_path: str = DEFAULT_ENGINE_PATH) -> str:
    candidate = (path or "").strip() or default_path
    candidate = os.path.expandvars(os.path.expanduser(candidate))
    if not os.path.isabs(candidate):
        candidate = os.path.join(BASE_DIR, candidate)
    candidate = os.path.normpath(candidate)
    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, DEFAULT_ENGINE_FILENAME)
    return os.path.normpath(candidate)


def _display_engine_path(path: Optional[str], default_path: str = DEFAULT_ENGINE_RELATIVE_PATH) -> str:
    candidate = (path or "").strip() or default_path
    candidate = os.path.expandvars(os.path.expanduser(candidate))
    if os.path.isabs(candidate):
        try:
            rel = os.path.relpath(candidate, BASE_DIR)
            if not rel.startswith(".."):
                candidate = rel
        except ValueError:
            pass
    return os.path.normpath(candidate)


def get_default_player_pikafish_path() -> str:
    return _display_engine_path(DEFAULT_ENGINE_RELATIVE_PATH)


def get_default_eval_pikafish_path() -> str:
    data = load_app_config()
    pikafish_cfg = data.get("pikafish", {})
    if not isinstance(pikafish_cfg, dict):
        pikafish_cfg = {}
    return _display_engine_path(pikafish_cfg.get("eval_engine_path"), DEFAULT_ENGINE_RELATIVE_PATH)


# --- Models ---

def _player_display_name(config) -> str:
    """Optional player display name from config (empty if unset)."""
    return (getattr(config, "name", None) or "").strip()


def _player_label(config) -> str:
    """Human-readable label for a player config (no API keys)."""
    custom = _player_display_name(config)
    if config.type == "random":
        base = "Random"
    elif config.type in ("llm", "ai"):
        model = config.preset or config.model or "unknown"
        base = f"LLM ({model})"
    elif config.type == "external":
        base = f"External ({config.preset})" if config.preset else "External agent"
    else:
        base = config.type or "unknown"
    if custom:
        return f"{custom} · {base}"
    return base


def _player_filename_label(config) -> str:
    """Short player label suitable for filenames."""
    custom = _player_display_name(config)
    if custom:
        return custom
    if config.type == "random":
        return "Random"
    if config.type in ("llm", "ai"):
        name = config.preset or config.model or "unknown"
        return f"LLM-{name}"
    return config.type or "unknown"


def _sanitize_filename_part(text: str) -> str:
    """Make a Windows-safe filename fragment."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
    return cleaned[:80] or "unknown"


def _append_indented_block(lines: list[str], header: str, content: str):
    """Append a possibly multi-line block with indentation."""
    lines.append(header)
    if not content:
        lines.append("    (empty)")
        return
    for part in str(content).splitlines():
        lines.append(f"    {part}")


def _format_move_text(move_data: dict) -> str:
    move = move_data.get("move", "")
    move_zh = move_data.get("move_zh")
    return f"{move} ({move_zh})" if move_zh else move


def _format_result_text(game) -> str:
    if game.winner == "draw":
        return game.reason or "draw"
    if game.winner:
        return f"{game.winner} wins - {game.reason}"
    return game.reason or "unfinished"


def _merged_stream_events(events: list[dict]) -> list[dict]:
    merged = []
    for event in events:
        if (
            merged
            and event.get("type") in {"thinking", "reasoning"}
            and merged[-1].get("type") == event.get("type")
            and merged[-1].get("side") == event.get("side")
        ):
            merged[-1]["content"] = f"{merged[-1].get('content', '')}{event.get('content', '')}"
            continue
        merged.append(dict(event))

    for event in merged:
        if event.get("type") in {"thinking", "reasoning"}:
            text = str(event.get("content", "")).replace("\r\n", "\n").strip("\n")
            text = re.sub(r"\n{3,}", "\n\n", text)
            event["content"] = text
    return merged


def _append_event_log(lines: list[str], events: list[dict]):
    """Append streamed thinking/tool events to the saved log."""
    lines.append("Detailed Event Log:")

    for event in _merged_stream_events(events):
        event_type = event.get("type")
        side = event.get("side", "-")

        if event_type == "turn":
            lines.append(f"  [{side}] turn")
        elif event_type == "thinking":
            if event.get("content", ""):
                _append_indented_block(lines, f"  [{side}] thinking:", event.get("content", ""))
        elif event_type == "reasoning":
            if event.get("content", ""):
                _append_indented_block(lines, f"  [{side}] reasoning:", event.get("content", ""))
        elif event_type == "tool_call":
            args_text = json.dumps(event.get("args", {}), ensure_ascii=False)
            lines.append(f"  [{side}] tool_call: {event.get('tool', '')}({args_text})")
        elif event_type == "tool_result":
            _append_indented_block(lines, f"  [{side}] tool_result: {event.get('tool', '')}", event.get("result", ""))
        elif event_type == "move":
            captured = f" x{event['captured']}" if event.get("captured") else ""
            lines.append(f"  [{side}] move: #{event.get('number')} {_format_move_text(event)}{captured}")
        elif event_type == "game_over":
            lines.append(f"  [system] game_over: winner={event.get('winner')} reason={event.get('reason')}")


def _mask_secret(secret: Optional[str]) -> Optional[str]:
    if not secret:
        return None
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def _player_config_summary(config) -> dict:
    summary = {
        "type": config.type,
        "name": _player_display_name(config) or None,
    }
    if config.type in ("llm", "ai"):
        summary.update({
            "preset": config.preset,
            "api_base": config.api_base,
            "model": config.model,
            "prompt_name": _resolved_prompt_name(config),
            "enable_thinking": config.enable_thinking,
            "max_completion_tokens": _resolved_max_completion_tokens(config),
            "context_window": _resolved_context_window(config),
            "max_input_tokens": _resolved_max_input_tokens(config),
            "api_key_masked": _mask_secret(config.api_key),
        })
    return {k: v for k, v in summary.items() if v is not None}


def _append_player_configs(lines: list[str], game):
    lines.append("Player Configs:")
    red_config_text = json.dumps(_player_config_summary(game.red_config), ensure_ascii=False, indent=2)
    black_config_text = json.dumps(_player_config_summary(game.black_config), ensure_ascii=False, indent=2)
    _append_indented_block(lines, "  Red Config:", red_config_text)
    _append_indented_block(lines, "  Black Config:", black_config_text)


def _format_move_zh(m: dict) -> str:
    return (m.get("move_zh") or m.get("cn") or m.get("move") or "").strip()


def _side_label_zh(side: str) -> str:
    return "红方" if side == "red" else "黑方" if side == "black" else side



def write_game_log(game):
    """Write one game record per file under logs/."""
    from datetime import datetime
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    ts_filename = now.strftime("%Y%m%d-%H%M%S")
    red_label = _player_label(game.red_config)
    black_label = _player_label(game.black_config)
    red_filename = _sanitize_filename_part(_player_filename_label(game.red_config))
    black_filename = _sanitize_filename_part(_player_filename_label(game.black_config))

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Game: {game.id} | {ts}")
    lines.append(f"Red: {red_label}  vs  Black: {black_label}")
    _append_player_configs(lines, game)
    lines.append(f"Initial FEN: {game.initial_fen}")
    lines.append(f"Result: {_format_result_text(game)}")
    lines.append(f"Moves ({len(game.move_history)}):")
    for m in game.move_history:
        captured = f" x{m['captured']}" if m.get('captured') else ""
        lines.append(f"  #{m['number']} {m['side']}: {_format_move_text(m)}{captured}")
        summary = (m.get("summary_zh") or "").strip()
        if summary:
            _append_indented_block(lines, f"    summary:", summary)
    if external_mcp.game_has_external(game):
        lines.append(f"MCP tool audit JSONL: mcp/{game.id}.jsonl")
    else:
        lines.append(f"Turn thoughts JSONL: turns/{game.id}.jsonl")
    lines.append(f"Final FEN: {game.board.to_fen()}")
    lines.append("")
    _append_event_log(lines, game.events)
    lines.append("")

    filename = f"red-{red_filename}_vs_black-{black_filename}_{ts_filename}_{game.id}.log"
    log_path = os.path.join(LOG_DIR, filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")



# ---------------------------------------------------------------------------
# Log file parser (for history replay)
# ---------------------------------------------------------------------------

_RE_GAME_LINE = re.compile(r"^Game:\s+(\S+)\s+\|\s+(.+)$")
_RE_PLAYERS_LINE = re.compile(r"^Red:\s+(.+?)\s{2,}vs\s{2,}Black:\s+(.+)$")
_RE_INITIAL_FEN = re.compile(r"^Initial FEN:\s+(.+)$")
_RE_RESULT = re.compile(r"^Result:\s+(.+)$")
_RE_RESULT_WINNER = re.compile(r"^(red|black)\s+wins(?:\s*-\s*(.+))?$", re.IGNORECASE)
_RE_MOVES_HEADER = re.compile(r"^Moves\s+\((\d+)\):")
_RE_MOVE_LINE = re.compile(
    r"^\s*#(\d+)\s+(red|black):\s+(\S+)"
    r"(?:\s+\(([^)]+)\))?"
    r"(?:\s+x(\S+))?$"
)


def _parse_result_fields(result_text: str) -> tuple[Optional[str], str]:
    text = (result_text or "").strip()
    lower = text.lower()
    if not text or lower == "unfinished":
        return None, ""
    if lower == "draw":
        return "draw", ""
    if lower.startswith("draw - "):
        return "draw", text[7:].strip()
    match = _RE_RESULT_WINNER.match(text)
    if match:
        return match.group(1).lower(), (match.group(2) or "").strip()
    return None, text


def parse_game_log(filepath: str, include_moves: bool = True) -> dict:
    """Parse a .log file and return structured game data."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    result = {
        "filename": os.path.basename(filepath),
        "game_id": "",
        "timestamp": "",
        "red_label": "",
        "black_label": "",
        "initial_fen": "",
        "result": "",
        "winner": None,
        "reason": "",
        "move_count": 0,
    }

    in_moves = False
    raw_moves = []

    for line in raw_lines:
        line = line.rstrip("\n")

        m = _RE_GAME_LINE.match(line)
        if m:
            result["game_id"] = m.group(1)
            result["timestamp"] = m.group(2).strip()
            continue

        m = _RE_PLAYERS_LINE.match(line)
        if m:
            result["red_label"] = m.group(1).strip()
            result["black_label"] = m.group(2).strip()
            continue

        m = _RE_INITIAL_FEN.match(line)
        if m:
            result["initial_fen"] = m.group(1).strip()
            continue

        m = _RE_RESULT.match(line)
        if m:
            result["result"] = m.group(1).strip()
            winner, reason = _parse_result_fields(result["result"])
            result["winner"] = winner
            result["reason"] = reason
            continue

        m = _RE_MOVES_HEADER.match(line)
        if m:
            result["move_count"] = int(m.group(1))
            in_moves = True
            continue

        if in_moves:
            m = _RE_MOVE_LINE.match(line)
            if m:
                raw_moves.append({
                    "number": int(m.group(1)),
                    "side": m.group(2),
                    "move": m.group(3),
                    "move_zh": m.group(4) or None,
                    "captured": m.group(5) or None,
                })
                continue
            stripped = line.strip()
            if not stripped or stripped == "summary:" or line[:1].isspace():
                continue
            in_moves = False

    if not include_moves:
        return result

    # Replay moves to compute FEN after each step
    moves_with_fen = []
    if result["initial_fen"]:
        try:
            board = Board(result["initial_fen"])
            for rm in raw_moves:
                try:
                    move_result = board.make_move(rm["move"])
                    rm["fen"] = move_result.get("fen_after", board.to_fen())
                except Exception:
                    rm["fen"] = board.to_fen()
                moves_with_fen.append(rm)
        except Exception:
            # If board init fails, include moves without FEN
            for rm in raw_moves:
                rm["fen"] = ""
                moves_with_fen.append(rm)

    result["moves"] = moves_with_fen
    return result


class PlayerConfig(BaseModel):
    type: str = "llm"  # llm | random | external
    name: Optional[str] = None  # player display name
    preset: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_lang: Optional[str] = None  # legacy fallback
    enable_thinking: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    max_input_tokens: Optional[int] = None
    compress_threshold_ratio: Optional[float] = None
    compress_model: Optional[str] = None
    enable_reasoning_effort: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    rpm: Optional[int] = None
    tpm: Optional[int] = None


def _resolved_max_completion_tokens(config: PlayerConfig) -> int:
    return config.max_completion_tokens or config.max_output_tokens or 8192


def _resolved_prompt_name(config: PlayerConfig) -> str:
    return resolve_prompt_name(config.prompt_name, config.prompt_lang)



def _validate_prompt_config(config: PlayerConfig):
    if config.type != "llm":
        return
    try:
        get_prompt_profile(_resolved_prompt_name(config))
    except ValueError as e:
        raise HTTPException(400, str(e))


class PikafishConfig(BaseModel):
    enabled: bool = False
    engine_path: Optional[str] = None
    mode: str = "movetime"    # "movetime" | "depth"
    movetime: int = 1000
    depth: int = 20
    score_type: str = "Elo"  # "Elo" | "PawnValueNormalized" | "Raw"


class TimerConfig(BaseModel):
    """Elapsed-time stats only (no countdown / timeout loss)."""
    enabled: bool = False
    # Kept for backward-compatible clients; ignored in elapsed mode.
    initial_time: int = 0
    increment: int = 0



class CreateGameRequest(BaseModel):
    fen: str = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    red: PlayerConfig
    black: PlayerConfig
    pikafish: PikafishConfig = PikafishConfig()
    timer: TimerConfig = TimerConfig()


class SeekGameRequest(BaseModel):
    ply: int


class FinishGameRequest(BaseModel):
    moves: list[str]  # list of ICCS move strings in order
    winner: Optional[str] = None  # "red" | "black" | "draw" | None
    reason: Optional[str] = None


class SubmitMoveRequest(BaseModel):
    """One move from an external (MCP) agent, deposited for game_loop to apply."""

    side: str  # "red" | "black"
    move: str  # ICCS move string, e.g. "h2e2"
    note: Optional[str] = None  # free-text note recorded with the move


class ResignRequest(BaseModel):
    side: Optional[str] = None  # "red" | "black"; default = side to move


class WaitTurnRequest(BaseModel):
    """Block until it is ``side``'s turn, or the game leaves ``playing``."""

    side: str  # "red" | "black"
    timeout_sec: float = 600.0


class ClaimSeatRequest(BaseModel):
    """Mint / rotate a seat token for an external side."""

    side: str  # "red" | "black"
    name: Optional[str] = None


class PreviewMoveRequest(BaseModel):
    """Walk one ICCS move on the seat's ObserveSession work board."""

    move: str


class GameSession:
    def __init__(self, game_id: str, fen: str, red: PlayerConfig, black: PlayerConfig):
        self.id = game_id
        self.board = Board(fen)
        self.initial_fen = fen
        self.red_config = red
        self.black_config = black
        self.status = "waiting"  # waiting | playing | paused | finished
        self.winner: Optional[str] = None
        self.reason: Optional[str] = None
        self.move_history = []
        self.events = []
        self.task: Optional[asyncio.Task] = None
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        # Long-poll wakeups for POST /wait-turn (MCP wait_my_turn).
        self.waiter_event = asyncio.Event()
        # Pikafish engine evaluator
        self.pikafish: Optional[PikafishEvaluator] = None
        self.pikafish_config: Optional[PikafishConfig] = None
        self.eval_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self.eval_task: Optional[asyncio.Task] = None
        self.eval_pending: set[int] = set()
        # Elapsed-time stats (seconds used, not remaining)
        self.timer_config: Optional[TimerConfig] = None
        self.timer_red: float = 0.0
        self.timer_black: float = 0.0
        self.timer_turn_start: float = 0.0
        self.timer_move_elapsed: float = 0.0
        # Per-side live thought buffers (attached to move_record on move)
        self.thought_buf: dict[str, dict[str, str]] = {
            "red": {"reasoning": "", "thinking": ""},
            "black": {"reasoning": "", "thinking": ""},
        }
        self.eval_ready: dict[int, asyncio.Event] = {}
        # External (MCP) players: /api/game/{id}/move deposits moves here and
        # game_loop consumes them (infinite wait by design — pause/interrupt
        # cancels the task; an unconsumed or mid-apply move is re-queued).
        self.external_moves: dict[str, asyncio.Queue] = {
            "red": asyncio.Queue(),
            "black": asyncio.Queue(),
        }
        # Mirror of the single pending /move per side (snapshot-safe; not Queue internals).
        self.external_pending: dict[str, Optional[dict]] = {"red": None, "black": None}
        # Free-text note attached by the external agent to its next move.
        self.external_notes: dict[str, str] = {}
        # Seat tokens + ObserveSession work boards (external seats only).
        self.seat_tokens: dict[str, Optional[str]] = {"red": None, "black": None}
        self.observe: dict[str, Any] = {"red": None, "black": None}

    # High-frequency stream chunks: live fan-out only, never persisted.
    _EPHEMERAL_EVENT_TYPES = frozenset({
        "reasoning", "thinking", "thought", "ping",
    })

    def broadcast(self, event_type: str, data: dict, *, persist: bool | None = None):
        """Persist non-ephemeral events to the durable log buffer.

        Live fan-out to a browser SSE queue was removed; scripts poll /state.
        """
        should_persist = (
            persist
            if persist is not None
            else event_type not in self._EPHEMERAL_EVENT_TYPES
        )
        if should_persist:
            self.events.append({"type": event_type, **data})
            if len(self.events) > 4000:
                self.events = self.events[-3000:]


# --- Global state ---

games: dict[str, GameSession] = {}


def _build_move_record(move_number: int, side_name: str, result: dict) -> dict:
    return {
        "number": move_number,
        "side": side_name,
        "move": result["move"],
        "move_zh": result.get("move_zh"),
        "piece": result["piece"],
        "captured": result["captured"],
        "fen": result["fen_after"],
        "fen_before": result.get("fen_before"),
        "timestamp": time.time(),
        "cycle": result.get("cycle"),
    }


def _finish_game(game: GameSession, winner: str, reason: str):
    game.status = "finished"
    game.winner = winner
    game.reason = reason
    game.broadcast("game_over", {"winner": winner, "reason": reason})
    wake_waiters(game)


def wake_waiters(game: GameSession) -> None:
    """Unblock POST /wait-turn long-polls so they re-check turn/status."""
    game.waiter_event.set()


def _http_err(
    status: int,
    error: str,
    *,
    error_class: str = "rules",
    **extra: Any,
) -> None:
    raise HTTPException(
        status,
        detail={"error": error, "error_class": error_class, **extra},
    )


def _audit_call(
    game: GameSession,
    side: str,
    tool: str,
    args: Any = None,
) -> None:
    external_mcp.audit_tool(
        game, log_dir=LOG_DIR, side=side, tool=tool, phase="call", args=args
    )


def _audit_result(
    game: GameSession,
    side: str,
    tool: str,
    *,
    result: Any = None,
    ok: bool = True,
    error_class: Optional[str] = None,
) -> None:
    external_mcp.audit_tool(
        game,
        log_dir=LOG_DIR,
        side=side,
        tool=tool,
        phase="result",
        result=result,
        ok=ok,
        error_class=error_class,
    )


def _require_external_seat(
    game: GameSession,
    *,
    token: Optional[str],
    expected_side: Optional[str] = None,
    mcp_tool: Optional[str] = None,
) -> tuple[str, str]:
    return external_mcp.require_seat(
        game,
        token=token,
        expected_side=expected_side,
        mcp_tool=mcp_tool,
    )


def _side_state_entry(game: GameSession, side: str) -> dict:
    cfg = game.red_config if side == "red" else game.black_config
    entry = {
        "type": cfg.type,
        "name": _player_display_name(cfg) or None,
        "label": _player_label(cfg),
    }
    if cfg.type == "external":
        entry["claimed"] = mcp_contract.seat_claimed(game, side)
    return entry


def _game_state_payload(
    game: GameSession,
    *,
    include_pending: bool = False,
) -> dict:
    payload = {
        "game_id": game.id,
        "fen": game.board.to_fen(),
        "turn": _turn_side(game),
        "status": game.status,
        "winner": game.winner,
        "reason": game.reason,
        "move_count": len(game.move_history),
        "move_history": game.move_history,
        "sides": {
            "red": _side_state_entry(game, "red"),
            "black": _side_state_entry(game, "black"),
        },
        "activity": _activity_payload(game),
    }
    if include_pending:
        payload["external_pending"] = _peek_external_pending(game)
    return payload


def _turn_side(game: GameSession) -> str:
    return "red" if game.board.turn == "w" else "black"


def _side_last_tool_sec_ago(game: GameSession, side: str) -> Optional[float]:
    ts = (getattr(game, "side_last_tool_ts", None) or {}).get(side)
    if not ts:
        return None
    return round(time.time() - float(ts), 1)


def _activity_payload(game: GameSession) -> dict:
    """对手活跃度信号：长考≠退出，把客观事实亮给等待方 agent 看。"""
    out: dict = {
        "note": "对方长考（半小时甚至更久）属正常节奏，不代表退出；只有 status=finished 才是对局结束",
        "side_last_tool_sec_ago": {
            "red": _side_last_tool_sec_ago(game, "red"),
            "black": _side_last_tool_sec_ago(game, "black"),
        },
    }
    ts = game.timer_turn_start or getattr(game, "turn_started_ts", 0.0)
    if game.status == "playing" and ts:
        out["current_side_thinking_sec"] = round(time.time() - float(ts), 1)
    return out


def _wait_turn_done_reason(game: GameSession, side: str) -> Optional[str]:
    """Return why wait-turn can stop, or None to keep waiting."""
    if game.status == "finished":
        return "finished"
    if game.status == "paused":
        return "paused"
    if game.status == "interrupted":
        return "interrupted"
    if game.status == "waiting":
        # Still blocking for /start; if this session was reset/removed, stop.
        if games.get(game.id) is not game:
            return "reset"
        return None
    if game.status != "playing":
        return game.status or "not_playing"
    if _turn_side(game) == side:
        return "my_turn"
    return None


def _wait_turn_payload(game: GameSession, *, waiting_done_reason: str) -> dict:
    hist = game.move_history
    return {
        "ok": True,
        "game_id": game.id,
        "waiting_done_reason": waiting_done_reason,
        "status": game.status,
        "turn": _turn_side(game),
        "winner": game.winner,
        "reason": game.reason,
        "fen": game.board.to_fen(),
        "board": game.board.to_text(),
        "move_count": len(hist),
        "last_move": hist[-1] if hist else None,
        "sides": {
            "red": _side_state_entry(game, "red"),
            "black": _side_state_entry(game, "black"),
        },
        "activity": _activity_payload(game),
    }


def _snapshot_path(game_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (game_id or "unknown"))
    return os.path.join(SNAPSHOT_DIR, f"{safe}.json")


def _snapshot_player_config(config: PlayerConfig) -> dict:
    data = config.model_dump()
    data["api_key"] = None
    return data


def _rehydrate_snapshot_player(raw) -> PlayerConfig:
    config = PlayerConfig.model_validate(raw or {"type": "random"})
    if config.type not in {"llm", "external"} or not config.preset:
        return config
    try:
        return resolve_preset(config)
    except HTTPException:
        return config


def _peek_external_pending(game: GameSession) -> dict[str, list[dict]]:
    """Copy pending external move payloads for snapshots."""
    out: dict[str, list[dict]] = {"red": [], "black": []}
    for side in ("red", "black"):
        pending = game.external_pending.get(side)
        if isinstance(pending, dict) and pending.get("move"):
            out[side] = [dict(pending)]
    return out


def _clear_external_move_queues(game: GameSession) -> None:
    """Drop any unconsumed external moves (seek/reset invalidate them)."""
    for side in ("red", "black"):
        queue = game.external_moves[side]
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        game.external_pending[side] = None
    game.external_notes = {}


def _requeue_external_move(game: GameSession, side_name: str, queued: dict) -> None:
    """Put a dequeued external move back so pause/cancel cannot drop it."""
    payload = {
        "move": str(queued.get("move") or "").strip().lower(),
        "note": str(queued.get("note") or "").strip(),
    }
    if not payload["move"]:
        return
    game.external_pending[side_name] = payload
    game.external_notes.pop(side_name, None)
    queue = game.external_moves[side_name]
    # Avoid duplicating if pause raced with a second deposit (shouldn't happen).
    if queue.qsize() == 0:
        queue.put_nowait(dict(payload))


def _restore_external_pending(game: GameSession, pending: Any) -> None:
    """Re-queue external pending moves from a snapshot payload."""
    _clear_external_move_queues(game)
    if not isinstance(pending, dict):
        return
    for side in ("red", "black"):
        items = pending.get(side) or []
        if not isinstance(items, list):
            continue
        for raw in items:
            if isinstance(raw, dict) and raw.get("move"):
                payload = {
                    "move": str(raw["move"]).strip().lower(),
                    "note": str(raw.get("note") or "").strip(),
                }
            elif isinstance(raw, str) and raw.strip():
                payload = {"move": raw.strip().lower(), "note": ""}
            else:
                continue
            game.external_pending[side] = payload
            game.external_moves[side].put_nowait(dict(payload))
            break  # at most one pending per side


def write_game_snapshot(game: GameSession, *, reason: str = "") -> str:
    """Persist board + history so an interrupted game can resume."""
    payload = {
        "v": 1,
        "game_id": game.id,
        "status": game.status,
        "reason": reason or game.reason,
        "initial_fen": game.initial_fen,
        "fen": game.board.to_fen(),
        "turn": "red" if game.board.turn == "w" else "black",
        "winner": game.winner,
        "move_history": list(game.move_history),
        "red": _snapshot_player_config(game.red_config),
        "black": _snapshot_player_config(game.black_config),
        "pikafish": game.pikafish_config.model_dump() if game.pikafish_config else None,
        "timer_config": game.timer_config.model_dump() if game.timer_config else None,
        "timer": {
            "red": game.timer_red,
            "black": game.timer_black,
        },
        # Unconsumed /move deposits survive process restart + restore.
        "external_pending": _peek_external_pending(game),
        "ts": time.time(),
    }
    path = _snapshot_path(game.id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def _interrupt_game(game: GameSession, reason: str):
    """Stop the loop without declaring a winner; keep board for resume."""
    game.status = "interrupted"
    game.winner = None
    game.reason = reason
    game.pause_event.clear()
    try:
        write_game_snapshot(game, reason=reason)
    except Exception as exc:
        print(f"  [snapshot] write failed: {exc}")
    game.broadcast("interrupted", {"reason": reason, "fen": game.board.to_fen()})
    game.broadcast("status", {"status": "interrupted"})
    wake_waiters(game)


def load_game_runtime_snapshot(path: str) -> GameSession:
    """Rebuild a GameSession from a snapshot file (no live engines)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    game_id = str(data.get("game_id") or os.path.splitext(os.path.basename(path))[0])
    initial_fen = data.get("initial_fen") or "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
    red = _rehydrate_snapshot_player(data.get("red") or {"type": "random"})
    black = _rehydrate_snapshot_player(data.get("black") or {"type": "random"})
    game = GameSession(game_id, initial_fen, red, black)
    history = list(data.get("move_history") or [])
    board = Board(initial_fen)
    for rec in history:
        board.make_move(rec["move"])
    expected_fen = data.get("fen")
    if expected_fen and board.to_fen() != expected_fen:
        board = Board(expected_fen)
    game.board = board
    game.move_history = history
    game.status = data.get("status") or "interrupted"
    game.winner = data.get("winner")
    game.reason = data.get("reason")
    if data.get("pikafish"):
        game.pikafish_config = PikafishConfig.model_validate(data["pikafish"])
    if data.get("timer_config"):
        game.timer_config = TimerConfig.model_validate(data["timer_config"])
    timer = data.get("timer") or {}
    game.timer_red = float(timer.get("red") or 0.0)
    game.timer_black = float(timer.get("black") or 0.0)
    _restore_external_pending(game, data.get("external_pending"))
    return game


def _broadcast_timer(game: GameSession, active_side: Optional[str] = None):
    """Broadcast elapsed-time stats to all clients."""
    if not game.timer_config or not game.timer_config.enabled:
        return
    move_elapsed = 0.0
    if game.timer_turn_start:
        move_elapsed = max(0.0, time.time() - game.timer_turn_start)
    red = game.timer_red
    black = game.timer_black
    # Live move time is not yet committed into side totals.
    if active_side == "red":
        pass
    elif active_side == "black":
        pass
    total = red + black + move_elapsed
    game.broadcast("timer", {
        "mode": "elapsed",
        "red": round(red, 1),
        "black": round(black, 1),
        "move": round(move_elapsed, 1),
        "total": round(total, 1),
        "active": active_side,
    })


async def _run_with_turn_timeout(
    game: GameSession,
    side: str,
    side_name: str,
    action,
):
    """Compatibility wrapper: elapsed mode has no turn deadline."""
    return await action


async def _choose_random_move(game: GameSession, side_name: str) -> Optional[str]:
    await asyncio.sleep(0.3)  # Small delay for visual effect
    legal_moves = game.board.get_legal_moves()
    if not legal_moves:
        return None
    move_iccs = random.choice(legal_moves)
    game.broadcast("thinking", {"side": side_name, "content": f"Random move: {move_iccs}"})
    return move_iccs


async def _wait_external_move(game: GameSession, side_name: str) -> dict:
    """Wait indefinitely for an external (MCP) agent to submit a move.

    Returns the raw queue payload. Caller must either apply it or
    ``_requeue_external_move`` if the game left ``playing`` / was cancelled
    before ``make_move``.
    """
    return await game.external_moves[side_name].get()


def _resolved_context_window(config: PlayerConfig) -> int:
    return resolve_context_window(config.context_window)


def _resolved_max_input_tokens(config: PlayerConfig) -> int:
    return resolve_max_input_tokens(
        config.max_input_tokens,
        context_window=_resolved_context_window(config),
    )


def _resolved_compress_threshold_ratio(config: PlayerConfig) -> float:
    return resolve_compress_threshold_ratio(config.compress_threshold_ratio)


def _resolved_reasoning_effort(config: PlayerConfig) -> Optional[str]:
    if not config.enable_reasoning_effort:
        return None
    effort = (config.reasoning_effort or "").strip().lower()
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    return effort if effort in allowed else None


def _append_thought_buf(game: GameSession, side_name: str, reasoning: str, thinking: str):
    buf = game.thought_buf.setdefault(side_name, {"reasoning": "", "thinking": ""})
    if reasoning:
        buf["reasoning"] = (buf.get("reasoning") or "") + reasoning
    if thinking:
        buf["thinking"] = (buf.get("thinking") or "") + thinking


def _take_thought_buf(game: GameSession, side_name: str) -> tuple[str, str]:
    buf = game.thought_buf.get(side_name) or {"reasoning": "", "thinking": ""}
    reasoning = (buf.get("reasoning") or "")
    thinking = (buf.get("thinking") or "")
    game.thought_buf[side_name] = {"reasoning": "", "thinking": ""}
    return reasoning, thinking


async def _persist_llm_turn(
    game: GameSession,
    side_name: str,
    config: PlayerConfig,
    move_record: dict,
    turn_trace: Optional[dict] = None,
) -> None:
    """Write per-ply JSONL + audience Chinese retelling (fail-open)."""
    try:
        from agent.turn_store import append_turn
        from agent.summarize_turn import summarize_turn
    except Exception as e:
        print(f"  [turns] import failed: {e}")
        return

    trace = turn_trace or {}
    reasoning = (trace.get("reasoning") or move_record.get("reasoning") or "") or ""
    thinking = (trace.get("thinking") or move_record.get("thinking") or "") or ""
    tool_rounds = trace.get("tool_rounds") or []
    move = move_record.get("move") or ""
    move_zh = move_record.get("move_zh") or move_record.get("cn") or ""

    summary_zh = ""
    summary_meta: dict = {"ok": False}
    skip_summary = bool(move_record.get("error")) or not move
    try:
        sum_model = (config.compress_model or config.model or "").strip()
        if (not skip_summary) and sum_model and (config.api_base or config.type == "ai"):
            # type=ai already resolved to llm with credentials in create_game
            api_base = config.api_base or ""
            api_key = config.api_key or ""
            if api_base and api_key:
                summary_meta = await summarize_turn(
                    api_base=api_base,
                    api_key=api_key,
                    model=sum_model,
                    move=move,
                    move_zh=move_zh,
                    reasoning=reasoning,
                    thinking=thinking,
                    preset=config.preset or config.name or config.model,
                    rpm=config.rpm,
                    tpm=config.tpm,
                )
                summary_zh = summary_meta.get("summary_zh") or ""
    except Exception as e:
        summary_meta = {"ok": False, "error": str(e)[:200]}

    if summary_zh:
        move_record["summary_zh"] = summary_zh

    record = {
        "ply": move_record.get("number"),
        "side": side_name,
        "move": move,
        "move_zh": move_zh,
        "captured": move_record.get("captured"),
        "fen_before": move_record.get("fen_before") or "",
        "fen_after": move_record.get("fen") or move_record.get("fen_after") or "",
        "model": config.model,
        "reasoning": reasoning,
        "thinking": thinking,
        "assistant_content": trace.get("assistant_content") or thinking,
        "tool_rounds": tool_rounds,
        "summary_zh": summary_zh,
        "summary_meta": summary_meta,
        "error": move_record.get("error") or None,
    }
    try:
        append_turn(game.id, record)
    except Exception as e:
        print(f"  [turns] append failed: {e}")


async def _persist_failed_llm_turn(
    game: GameSession,
    side_name: str,
    config: PlayerConfig,
    error: str,
) -> None:
    last_player = getattr(game, "_last_llm_player", None)
    turn_trace = None
    if last_player is not None and getattr(last_player, "last_turn_trace", None):
        turn_trace = dict(last_player.last_turn_trace)
    reasoning, thinking = _take_thought_buf(game, side_name)
    fen = game.board.to_fen()
    move_record = {
        "number": len(game.move_history) + 1,
        "fen_before": fen,
        "fen": fen,
        "error": error,
        "reasoning": reasoning,
        "thinking": thinking,
    }
    await _persist_llm_turn(game, side_name, config, move_record, turn_trace=turn_trace)


async def _request_llm_move(game: GameSession, side: str, side_name: str, config: PlayerConfig) -> Optional[str]:
    player = LangGraphPlayer(
        api_base=config.api_base,
        api_key=config.api_key,
        model=config.model,
        game_id=game.id,
        side_name=side_name,
        prompt_name=_resolved_prompt_name(config),
        enable_thinking=True if config.enable_thinking is None else config.enable_thinking,
        max_completion_tokens=_resolved_max_completion_tokens(config),
        context_window=_resolved_context_window(config),
        max_input_tokens=_resolved_max_input_tokens(config),
        compress_threshold_ratio=_resolved_compress_threshold_ratio(config),
        compress_model=config.compress_model or config.model,
        enable_reasoning_effort=bool(config.enable_reasoning_effort),
        reasoning_effort=_resolved_reasoning_effort(config),
        preset=config.preset or config.name or config.model,
        rpm=config.rpm,
        tpm=config.tpm,
        # 走子阶段闸门：非 playing（复盘/预盘/终局/中断）时 make_move 禁用
        status_probe=lambda: game.status,
    )

    # Fresh thought buffer for this turn
    game.thought_buf[side_name] = {"reasoning": "", "thinking": ""}
    game._last_llm_player = player  # type: ignore[attr-defined]

    # Thought push is rate-limited upstream (~1Hz append chunks).
    # Ephemeral only — never append thought bodies to game.events history.
    async for event in player.request_move(game.board, side):
        if game.status != "playing":
            return None
        et = event.get("type")
        if et == "thought":
            reasoning = event.get("reasoning") or ""
            thinking = event.get("thinking") or ""
            if not reasoning and not thinking:
                continue
            _append_thought_buf(game, side_name, reasoning, thinking)
            game.broadcast(
                "thought",
                {
                    "side": side_name,
                    "mode": "append",
                    "reasoning": reasoning,
                    "thinking": thinking,
                },
                persist=False,
            )
        elif et in ("reasoning", "thinking"):
            content = event.get("content") or ""
            if not content:
                continue
            r = content if et == "reasoning" else ""
            t = content if et == "thinking" else ""
            _append_thought_buf(game, side_name, r, t)
            game.broadcast(
                "thought",
                {
                    "side": side_name,
                    "mode": "append",
                    "reasoning": r,
                    "thinking": t,
                },
                persist=False,
            )
        elif et == "move":
            return event["move"]
        elif et == "aborted":
            # 走子阶段闸门：对局已非 playing（复盘/预盘/终局/中断），
            # 回合被终止且不是模型失败 —— 不计错误、不触发 _interrupt_game
            return None
        elif et == "error":
            msg = f"{side_name} error: {event['message']}"
            await _persist_failed_llm_turn(game, side_name, config, msg)
            _interrupt_game(game, msg)
            return None

    return None


# --- Pikafish evaluation helper ---

def _signal_eval_ready(game: GameSession, move_number: int):
    ev = game.eval_ready.get(move_number)
    if ev is None:
        ev = asyncio.Event()
        game.eval_ready[move_number] = ev
    ev.set()


async def _pikafish_evaluate(game: GameSession, fen: str, move_number: int):
    """Run Pikafish evaluation and broadcast the result."""
    try:
        score_type = game.pikafish_config.score_type if game.pikafish_config else "Elo"

        def on_result(move_num, score):
            for move_record in game.move_history:
                if move_record.get("number") == move_num:
                    move_record["eval"] = score
                    move_record["score_type"] = score_type
                    break
            game.broadcast("eval", {"move_number": move_num, "score": score, "score_type": score_type})
            _signal_eval_ready(game, move_num)

        await game.pikafish.evaluate(fen, move_number, on_result)
    except Exception:
        _signal_eval_ready(game, move_number)
    else:
        # Ensure waiter unblocks even if engine returned no score callback
        rec = next((m for m in game.move_history if m.get("number") == move_number), None)
        if rec is not None and rec.get("eval") is None:
            _signal_eval_ready(game, move_number)


async def _eval_worker(game: GameSession):
    try:
        while True:
            move_number, fen = await game.eval_queue.get()
            try:
                if game.pikafish and game.pikafish_config and game.pikafish_config.enabled:
                    await _pikafish_evaluate(game, fen, move_number)
            finally:
                game.eval_pending.discard(move_number)
                game.eval_queue.task_done()
    except asyncio.CancelledError:
        pass
    finally:
        game.eval_task = None


async def _start_eval_worker(game: GameSession):
    if not game.pikafish or not game.pikafish_config or not game.pikafish_config.enabled:
        return
    if game.eval_task and not game.eval_task.done():
        return
    game.eval_task = asyncio.create_task(_eval_worker(game))


def _is_terminal_fen(fen: str) -> bool:
    try:
        board = Board(fen)
        is_over, _, _ = board.is_game_over()
        return is_over
    except Exception:
        return False


async def _queue_pikafish_eval(game: GameSession, fen: str, move_number: int):
    if not game.pikafish or not game.pikafish_config or not game.pikafish_config.enabled:
        _signal_eval_ready(game, move_number)
        return
    if _is_terminal_fen(fen):
        _signal_eval_ready(game, move_number)
        return
    if move_number in game.eval_pending:
        return
    for move_record in game.move_history:
        if move_record.get("number") == move_number and move_record.get("eval") is not None:
            _signal_eval_ready(game, move_number)
            return
    if move_number not in game.eval_ready:
        game.eval_ready[move_number] = asyncio.Event()
    game.eval_pending.add(move_number)
    await game.eval_queue.put((move_number, fen))
    await _start_eval_worker(game)


async def _queue_missing_evals(game: GameSession):
    if not game.pikafish or not game.pikafish_config or not game.pikafish_config.enabled:
        return
    for move_record in game.move_history:
        if move_record.get("eval") is None:
            await _queue_pikafish_eval(game, move_record["fen"], move_record["number"])


async def _stop_eval_worker(game: GameSession):
    if game.eval_task and not game.eval_task.done():
        game.eval_task.cancel()
        try:
            await game.eval_task
        except asyncio.CancelledError:
            pass
    game.eval_task = None
    game.eval_queue = asyncio.Queue()
    game.eval_pending.clear()


async def _release_eval_engine(game: GameSession):
    """Stop the eval worker and shut down Pikafish so interrupt/reset do not leak processes."""
    await _stop_eval_worker(game)
    if not game.pikafish:
        return
    try:
        await game.pikafish.shutdown()
    except Exception:
        pass
    game.pikafish = None



async def _start_eval_engine(game: GameSession):
    if not game.pikafish_config or not game.pikafish_config.enabled or game.pikafish:
        return
    try:
        cfg = game.pikafish_config
        eval_engine_path = _resolved_eval_engine_path(cfg)
        eval_engine_path_display = _display_engine_path(cfg.engine_path, get_default_eval_pikafish_path())
        print(
            f"  [Pikafish] Starting evaluator: engine={eval_engine_path_display}, "
            f"mode={cfg.mode}, movetime={cfg.movetime}, depth={cfg.depth}, score_type={cfg.score_type}"
        )
        evaluator = PikafishEvaluator(
            engine_path=eval_engine_path,
            movetime=cfg.movetime if cfg.mode == "movetime" else None,
            depth=cfg.depth if cfg.mode == "depth" else None,
            score_type=cfg.score_type,
        )
        await evaluator.start()
        game.pikafish = evaluator
        await _start_eval_worker(game)
        await _queue_missing_evals(game)
    except Exception as e:
        print(f"  [Pikafish] Failed to start evaluator: {e}")


# --- Game loop ---

async def game_loop(game: GameSession):
    """Main game loop: alternates between players based on their type."""
    try:
        await _game_loop_inner(game)
    finally:
        if game.status in {"finished", "interrupted"}:
            try:
                write_game_snapshot(game)
            except Exception:
                pass
        if game.status == "finished" and game.pikafish:
            try:
                await game.eval_queue.join()
            except Exception:
                pass
        if game.status == "finished":
            game.broadcast("status", {"status": "finished"})
        if game.status in {"finished", "interrupted"}:
            try:
                write_game_log(game)
            except Exception:
                pass
        if game.status in {"finished", "interrupted"}:
            await _release_eval_engine(game)
        if game.status == "finished":
            try:
                await clear_game_threads(game.id)
            except Exception as exc:
                print(f"  [redis] clear threads failed for {game.id}: {exc}")
            asyncio.get_event_loop().call_later(
                300, lambda gid=game.id: games.pop(gid, None)
            )


async def _game_loop_inner(game: GameSession):
    game.status = "playing"
    game.broadcast("status", {"status": "playing"})
    wake_waiters(game)

    # Start server-side Pikafish evaluator when analysis is enabled.
    await _start_eval_engine(game)

    move_number = len(game.move_history)

    while game.status == "playing":
        await game.pause_event.wait()
        if game.status != "playing":
            break

        side = game.board.turn
        side_name = "red" if side == 'w' else "black"
        config = game.red_config if side == 'w' else game.black_config

        game.broadcast("turn", {"side": side_name, "fen": game.board.to_fen()})
        game.turn_started_ts = time.time()  # 与计时器无关：供 activity 长考信号用

        # Start elapsed-time clock for this turn
        if game.timer_config and game.timer_config.enabled:
            game.timer_turn_start = time.time()
            game.timer_move_elapsed = 0.0
            _broadcast_timer(game, active_side=side_name)

        move_made = False
        latest_move_record = None
        queued_external: Optional[dict] = None

        try:
            move_iccs = None
            if config.type == "random":
                move_iccs = await _run_with_turn_timeout(
                    game, side, side_name, _choose_random_move(game, side_name)
                )

            elif config.type == "llm":
                move_iccs = await _run_with_turn_timeout(
                    game, side, side_name, _request_llm_move(game, side, side_name, config)
                )

            elif config.type == "external":
                queued_external = await _wait_external_move(game, side_name)
                move_iccs = str(queued_external.get("move") or "").strip().lower()
                note = (queued_external.get("note") or "").strip()
                if note:
                    game.external_notes[side_name] = note
                game.broadcast(
                    "thinking",
                    {"side": side_name, "content": f"External move: {move_iccs}"},
                )

            if game.status != "playing":
                if queued_external is not None:
                    _requeue_external_move(game, side_name, queued_external)
                return

            if move_iccs:
                try:
                    result = game.board.make_move(move_iccs)
                except ValueError as e:
                    if queued_external is not None:
                        # Illegal after dequeue should not happen (/move pre-checked);
                        # drop pending so the seat can submit again.
                        game.external_pending[side_name] = None
                        game.external_notes.pop(side_name, None)
                    _interrupt_game(game, f"Invalid move by {side_name}: {e}")
                    return
                move_number += 1
                move_record = _build_move_record(move_number, side_name, result)
                # Attach this side's thought buffer (LLM) for client history
                reasoning, thinking = _take_thought_buf(game, side_name)
                turn_trace = None
                last_player = getattr(game, "_last_llm_player", None)
                if last_player is not None and getattr(last_player, "last_turn_trace", None):
                    turn_trace = dict(last_player.last_turn_trace)
                    if not reasoning:
                        reasoning = turn_trace.get("reasoning") or ""
                    if not thinking:
                        thinking = turn_trace.get("thinking") or turn_trace.get("assistant_content") or ""
                if reasoning:
                    move_record["reasoning"] = reasoning
                if thinking:
                    move_record["thinking"] = thinking
                note = game.external_notes.pop(side_name, None)
                if note:
                    move_record["note"] = note
                # Persist full turn + audience retelling (LLM only; fail-open)
                if config.type == "llm":
                    await _persist_llm_turn(
                        game, side_name, config, move_record, turn_trace=turn_trace
                    )
                game.move_history.append(move_record)
                if queued_external is not None:
                    game.external_pending[side_name] = None
                external_mcp.reset_observe(game)
                game.broadcast("move", move_record)
                wake_waiters(game)
                latest_move_record = move_record
                move_made = True
                queued_external = None  # consumed

        except asyncio.CancelledError:
            if queued_external is not None and not move_made:
                _requeue_external_move(game, side_name, queued_external)
            raise
        except Exception as e:
            _interrupt_game(game, f"{side_name} exception: {str(e)[:200]}")
            return

        if not move_made:
            if game.status != "playing":
                # 非 playing（复盘/预盘/终局/中断/暂停竞态）：不覆盖状态、
                # 不重复 interrupt —— 走子闸门 abort 的回合在此静默结束
                return
            _interrupt_game(game, f"{side_name} failed to make a move")
            return

        # Commit elapsed time after move (no timeout loss)
        if game.timer_config and game.timer_config.enabled and game.timer_turn_start:
            elapsed = max(0.0, time.time() - game.timer_turn_start)
            if side == 'w':
                game.timer_red += elapsed
            else:
                game.timer_black += elapsed
            game.timer_move_elapsed = 0.0
            game.timer_turn_start = 0.0
            _broadcast_timer(game, active_side=None)

        # Check game over
        is_over, winner, reason = game.board.is_game_over()
        if is_over:
            _finish_game(game, winner, reason)
            return

        if game.pikafish and latest_move_record:
            await _queue_pikafish_eval(game, latest_move_record["fen"], latest_move_record["number"])

        await asyncio.sleep(0.3)


# --- FastAPI app ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_rate_limiter(yaml_data=load_app_config())
    except Exception as exc:
        print(f"  [rate] init failed: {exc}")
    try:
        yield
    finally:
        await close_rate_limiter()
        for g in games.values():
            if g.task and not g.task.done():
                g.task.cancel()
            await _release_eval_engine(g)


app = FastAPI(
    title="Xiangqi LLM Duel",
    description="Backend-only Xiangqi arena: LangGraph LLM players, MCP seat gating, LiteLLM routing.",
    lifespan=lifespan,
)


# --- API endpoints ---

@app.get("/api/health")
async def health_check():
    """Simple readiness probe for monitoring scripts (replaces /api/presets hack)."""
    return {"status": "ok", "service": "xiangqi-llm-duel"}


@app.get("/api/presets")
async def get_presets():
    """Return model presets from config.yaml (safe fields only, no keys exposed)."""
    presets = load_model_presets()
    safe_presets = []
    for p in presets:
        try:
            cw = resolve_context_window(p.get("context_window"))
        except ValueError:
            # Misconfigured preset: missing context_window; still expose
            # other metadata so frontends can show the broken preset.
            cw = None
        safe_presets.append({
            "name": p["name"],
            "type": p.get("type") or "llm",
            "display_name": p.get("display_name"),
            "prompt_name": p.get("prompt_name", get_default_prompt_name()),
            "enable_thinking": p.get("enable_thinking", True),
            "max_completion_tokens": p.get("max_completion_tokens", 8192),
            "context_window": cw,
            "max_input_tokens": (
                resolve_max_input_tokens(p.get("max_input_tokens"), context_window=cw)
                if cw is not None
                else p.get("max_input_tokens")
            ),
            "compress_threshold_ratio": resolve_compress_threshold_ratio(
                p.get("compress_threshold_ratio")
            ),
            "compress_model": p.get("compress_model"),
            "enable_reasoning_effort": p.get("enable_reasoning_effort", False),
            "reasoning_effort": p.get("reasoning_effort"),
        })
    return {
        "presets": safe_presets,
        "default_pikafish_path": get_default_player_pikafish_path(),
        "default_eval_pikafish_path": get_default_eval_pikafish_path(),
    }


def resolve_preset(config: PlayerConfig) -> PlayerConfig:
    """If config uses a preset name, fill in details from config.yaml."""
    if not config.preset:
        return config
    presets = load_model_presets()
    entry = next((p for p in presets if p["name"] == config.preset), None)
    if entry is None:
        if config.type == "llm":
            raise HTTPException(400, f"Preset '{config.preset}' not found in config.yaml")
        return config

    # External seat: a config.yaml preset declaring `type: external` is a slot
    # for an out-of-process agent (joined via MCP tools), so it carries no
    # api_base/key/model — only a display name.
    if entry.get("type") == "external":
        if config.type not in {"external", "llm"}:
            raise HTTPException(
                400,
                f"Preset '{config.preset}' is an external seat; player type must be external",
            )
        yaml_display = str(entry.get("display_name") or "").strip()
        incoming = (config.name or "").strip()
        resolved_name = (
            incoming
            if incoming and incoming != config.preset
            else (yaml_display or incoming or None)
        )
        return PlayerConfig(type="external", name=resolved_name, preset=config.preset)

    if config.type == "external":
        raise HTTPException(
            400,
            f"Preset '{config.preset}' is not an external seat (type={entry.get('type') or 'llm'})",
        )
    if config.type != "llm":
        return config
    return PlayerConfig(
        type="llm",
        name=(
            (config.name or "").strip()
            if (config.name or "").strip() and config.name != config.preset
            else None
        )
        or str(entry.get("display_name") or "").strip()
        or (config.name or "").strip()
        or None,
        preset=config.preset,
        api_base=entry["api_base"],
        api_key=str(entry.get("api_key") or "").strip(),
        model=entry["model"],
        prompt_name=resolve_prompt_name(
            config.prompt_name or entry.get("prompt_name"),
            config.prompt_lang,
        ),
        enable_thinking=(
            config.enable_thinking
            if config.enable_thinking is not None
            else entry.get("enable_thinking", True)
        ),
        max_completion_tokens=(
            config.max_completion_tokens
            or config.max_output_tokens
            or entry.get("max_completion_tokens")
            or entry.get("max_output_tokens", 8192)
        ),
        context_window=resolve_context_window(entry.get("context_window")),
        max_input_tokens=resolve_max_input_tokens(
            entry.get("max_input_tokens"),
            context_window=resolve_context_window(entry.get("context_window")),
        ),
        compress_threshold_ratio=resolve_compress_threshold_ratio(
            entry.get("compress_threshold_ratio")
        ),
        # Prefer explicit/preset; otherwise compress with game model.
        compress_model=config.compress_model or entry.get("compress_model") or entry.get("model"),
        enable_reasoning_effort=(
            config.enable_reasoning_effort
            if config.enable_reasoning_effort is not None
            else entry.get("enable_reasoning_effort", False)
        ),
        reasoning_effort=config.reasoning_effort or entry.get("reasoning_effort"),
        rpm=config.rpm if config.rpm is not None else entry.get("rpm"),
        tpm=config.tpm if config.tpm is not None else entry.get("tpm"),
    )


def _validate_player_type(config: PlayerConfig):
    if config.type not in {"random", "llm", "ai", "external"}:
        raise HTTPException(400, f"Unsupported player type: {config.type}")


def _resolved_eval_engine_path(config: PikafishConfig) -> str:
    return _normalize_engine_path(config.engine_path, DEFAULT_ENGINE_RELATIVE_PATH)


def _validate_eval_pikafish_config(config: PikafishConfig):
    if not config.enabled:
        return
    if config.mode not in {"movetime", "depth"}:
        raise HTTPException(400, f"Invalid evaluation Pikafish mode: {config.mode}")


@app.post("/api/game/create")
async def create_game(req: CreateGameRequest):
    try:
        test_board = Board(req.fen)
        test_board.to_fen()
    except Exception as e:
        raise HTTPException(400, f"Invalid FEN: {e}")

    _validate_player_type(req.red)
    _validate_player_type(req.black)
    game_id = str(uuid.uuid4())[:8]
    red = resolve_preset(req.red)
    black = resolve_preset(req.black)
    _validate_prompt_config(red)
    _validate_prompt_config(black)
    _validate_eval_pikafish_config(req.pikafish)

    game = GameSession(game_id, req.fen, red, black)
    game.pikafish_config = req.pikafish
    if req.timer.enabled:
        game.timer_config = req.timer
        game.timer_red = 0.0
        game.timer_black = 0.0
        game.timer_turn_start = 0.0
        game.timer_move_elapsed = 0.0
    games[game_id] = game
    return {"game_id": game_id, "fen": req.fen}


@app.get("/api/game/{game_id}/state")
async def get_game_state(
    game_id: str,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    token = (x_xiangqi_seat_token or "").strip() or None
    if token:
        side, tool = _require_external_seat(
            game, token=token, mcp_tool=x_xiangqi_mcp_tool
        )
        _audit_call(game, side, tool, args={"op": "get_board"})
        payload = _game_state_payload(game, include_pending=True)
        payload["seat"] = side
        payload["board"] = game.board.to_text()
        _audit_result(game, side, tool, result={"status": payload["status"], "turn": payload["turn"], "move_count": payload["move_count"]}, ok=True)
        return payload
    # Spectator: no pending deposits.
    return _game_state_payload(game, include_pending=False)


@app.post("/api/game/{game_id}/start")
async def start_game(game_id: str):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status == "playing":
        raise HTTPException(400, "Game already playing")
    if game.status == "finished":
        raise HTTPException(400, "Game already finished, create a new one")
    if game.status == "interrupted":
        raise HTTPException(400, "Game interrupted; POST /resume to continue")

    # If neither side is LLM/external, the game is "local" — client runs its own
    # loop. Server still tracks state for /api/games and writes log on /finish.
    # External games ARE server-driven: game_loop waits on the /move queue.
    is_local = (
        game.red_config.type not in {"llm", "external"}
        and game.black_config.type not in {"llm", "external"}
    )
    if is_local:
        game.status = "playing"
        game.broadcast("status", {"status": "playing"})
    else:
        game.status = "playing"
        game.task = asyncio.create_task(game_loop(game))
    wake_waiters(game)

    return {"status": "started", "local": is_local}


@app.post("/api/game/{game_id}/pause")
async def pause_game(game_id: str):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status != "playing":
        raise HTTPException(400, "Game is not playing")
    game.pause_event.clear()
    game.status = "paused"
    if game.task and not game.task.done():
        game.task.cancel()
        try:
            await game.task
        except asyncio.CancelledError:
            pass
    game.task = None
    try:
        write_game_snapshot(game, reason=game.reason or "paused")
    except Exception as exc:
        print(f"  [snapshot] pause write failed: {exc}")
    game.broadcast("status", {"status": "paused"})
    wake_waiters(game)
    return {"status": "paused"}


@app.post("/api/game/{game_id}/resume")
async def resume_game(game_id: str):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status not in {"paused", "interrupted"}:
        raise HTTPException(400, "Game is not paused or interrupted")
    game.winner = None
    game.reason = None
    game.status = "playing"
    game.pause_event.set()
    if not game.task or game.task.done():
        game.task = asyncio.create_task(game_loop(game))
    await _start_eval_engine(game)
    await _queue_missing_evals(game)
    wake_waiters(game)
    return {"status": "resumed"}


@app.post("/api/game/{game_id}/restore")
async def restore_game(game_id: str):
    """Load an interrupted snapshot from disk if the in-memory session is gone."""
    if game_id in games:
        game = games[game_id]
        return {
            "status": game.status,
            "game_id": game.id,
            "fen": game.board.to_fen(),
            "reason": game.reason,
            "move_count": len(game.move_history),
        }
    path = _snapshot_path(game_id)
    if not os.path.isfile(path):
        raise HTTPException(404, "Snapshot not found")
    try:
        game = load_game_runtime_snapshot(path)
    except Exception as e:
        raise HTTPException(400, f"Invalid snapshot: {e}")
    games[game.id] = game
    return {
        "status": game.status,
        "game_id": game.id,
        "fen": game.board.to_fen(),
        "reason": game.reason,
        "move_count": len(game.move_history),
    }


@app.post("/api/game/{game_id}/seek")
async def seek_game(game_id: str, req: SeekGameRequest):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status != "paused":
        raise HTTPException(400, "Game must be paused before seeking")

    target_ply = req.ply
    if target_ply < 0 or target_ply > len(game.move_history):
        raise HTTPException(400, f"Invalid ply: {target_ply}")

    board = Board(game.initial_fen)
    kept_history = [dict(m) for m in game.move_history[:target_ply]]
    for move_record in kept_history:
        try:
            board.make_move(move_record["move"])
        except ValueError as e:
            raise HTTPException(500, f"Failed to replay move history: {e}")

    game.board = board
    game.move_history = kept_history
    game.winner = None
    game.reason = None
    game.events = []
    # Seek rewinds the board; any queued external move is now stale.
    _clear_external_move_queues(game)
    external_mcp.reset_observe(game)
    await _stop_eval_worker(game)
    try:
        await clear_game_threads(game.id)
    except Exception:
        pass

    # Stop any ongoing Pikafish analysis
    if game.pikafish:
        try:
            await game.pikafish.stop_analysis()
        except Exception:
            pass
        await _start_eval_worker(game)
        await _queue_missing_evals(game)

    response = {
        "status": "seeked",
        "ply": target_ply,
        "fen": game.board.to_fen(),
        "turn": "red" if game.board.turn == 'w' else "black",
        "move_history": game.move_history,
    }
    game.broadcast("seek", response)
    wake_waiters(game)
    return response


@app.post("/api/game/{game_id}/reset")
async def reset_game(game_id: str):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.task and not game.task.done():
        game.task.cancel()
        try:
            await game.task
        except asyncio.CancelledError:
            pass
        game.task = None
    await _release_eval_engine(game)
    try:
        await clear_game_threads(game.id)
    except Exception:
        pass
    game.board = Board(game.initial_fen)
    game.status = "waiting"
    game.winner = None
    game.reason = None
    game.move_history.clear()
    game.events.clear()
    _clear_external_move_queues(game)
    external_mcp.reset_observe(game)
    game.seat_tokens = {"red": None, "black": None}
    game.pause_event.set()
    game.broadcast("status", {"status": "reset"})
    wake_waiters(game)
    del games[game_id]
    return {"status": "reset"}


@app.post("/api/game/{game_id}/finish")
async def finish_game(game_id: str, req: FinishGameRequest):
    """Finalize a client-driven (local) game: replay moves on server, write log.
    Idempotent — calling twice on a finished game is a no-op.
    """
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status == "finished":
        try:
            await clear_game_threads(game.id)
        except Exception as exc:
            print(f"  [redis] clear threads failed for {game.id}: {exc}")
        return {"status": "already_finished"}

    # Replay from initial FEN to reconstruct server-side move history
    board = Board(game.initial_fen)
    rebuilt_history = []
    move_number = 0
    for move_iccs in req.moves:
        try:
            result = board.make_move(move_iccs)
        except ValueError as e:
            raise HTTPException(400, f"Invalid move in history: {move_iccs}: {e}")
        move_number += 1
        side_name = "black" if board.turn == 'w' else "red"  # just flipped
        rebuilt_history.append(_build_move_record(move_number, side_name, result))

    game.board = board
    game.move_history = rebuilt_history

    # Determine winner/reason: trust client if provided, otherwise detect
    winner = req.winner
    reason = req.reason or ""
    if not winner:
        is_over, w, r = board.is_game_over()
        if is_over:
            winner = w
            reason = r
    if winner:
        _finish_game(game, winner, reason)

    # Write log file
    try:
        write_game_log(game)
    except Exception:
        pass

    if game.status == "finished":
        try:
            await clear_game_threads(game.id)
        except Exception as exc:
            print(f"  [redis] clear threads failed for {game.id}: {exc}")
        asyncio.get_event_loop().call_later(
            300, lambda gid=game.id: games.pop(gid, None)
        )
    return {"status": "finished", "winner": winner, "reason": reason}


# --- External (MCP) player APIs ---

@app.post("/api/game/{game_id}/claim-seat")
async def claim_seat(game_id: str, req: ClaimSeatRequest):
    """Issue (or rotate) a seat token for an external side; embeds player contract."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side = (req.side or "").strip().lower()
    token = external_mcp.issue_seat_token(game, side)
    if req.name and str(req.name).strip():
        cfg = game.red_config if side == "red" else game.black_config
        cfg.name = str(req.name).strip()
    _audit_call(game, side, "claim_seat", args={"side": side, "name": req.name})
    out = mcp_contract.attach_contract(
        {
            "ok": True,
            "game_id": game.id,
            "side": side,
            "seat_token": token,
            "status": game.status,
            "turn": _turn_side(game),
        }
    )
    _audit_result(game, side, "claim_seat", result={"side": side}, ok=True)
    return out


@app.get("/api/game/{game_id}/legal_moves")
async def get_legal_moves(
    game_id: str,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Legal ICCS moves. With seat token: annotated list on ObserveSession work board."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    token = (x_xiangqi_seat_token or "").strip() or None
    if not token:
        legal = game.board.get_legal_moves()
        return {
            "game_id": game.id,
            "turn": _turn_side(game),
            "fen": game.board.to_fen(),
            "board": game.board.to_text(),
            "legal_moves": legal,
            "legal_count": len(legal),
            "status": game.status,
            "spectator": True,
        }
    side, tool = _require_external_seat(
        game, token=token, mcp_tool=x_xiangqi_mcp_tool
    )
    _audit_call(game, side, tool, args={"op": "get_legal_moves"})
    obs = external_mcp.get_observe(game, side)
    annotated = legal_moves_annotated(obs.work)
    legal = obs.work.get_legal_moves()
    out = {
        "game_id": game.id,
        "seat": side,
        "turn": "red" if obs.work.turn == "w" else "black",
        "fen": obs.work.to_fen(),
        "board": obs.work.to_text(),
        "legal_moves": legal,
        "legal_count": len(legal),
        "annotated": annotated,
        "path": list(obs.path),
        "status": game.status,
        "live_fen": game.board.to_fen(),
    }
    _audit_result(
        game,
        side,
        tool,
        result={"legal_count": out["legal_count"], "path_len": len(obs.path)},
        ok=True,
    )
    return out


@app.post("/api/game/{game_id}/preview")
async def preview_move(
    game_id: str,
    req: PreviewMoveRequest,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Walk one move on the seat's work board (does not change live)."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side, tool = _require_external_seat(
        game, token=x_xiangqi_seat_token, mcp_tool=x_xiangqi_mcp_tool
    )
    move = (req.move or "").strip().lower()
    _audit_call(game, side, tool, args={"move": move})
    obs = external_mcp.get_observe(game, side)
    text = obs.preview(move)
    illegal = "Illegal move" in text or "Invalid move" in text
    out = {
        "ok": not illegal,
        "game_id": game.id,
        "seat": side,
        "text": text,
        "path": list(obs.path),
        "fen_work": obs.work.to_fen(),
        "fen_live": game.board.to_fen(),
        "error_class": None if not illegal else "rules",
        "work_board_mode": "cumulative",
        "work_turn": "red" if obs.work.turn == "w" else "black",
    }
    if illegal:
        diag = obs.work.explain_illegal_move(move) or {}
        if diag.get("reason"):
            out["reason"] = diag["reason"]
        if diag.get("code"):
            out["reason_code"] = diag["code"]
        out["hint"] = (
            "preview 是累积式工作盘：预演一步后轮次换边，"
            "可继续预演对方应着，或 preview_reset 回到真实盘重新预演"
        )
    else:
        opp_in_check = obs.work.is_in_check(obs.work.turn)
        opp_has_moves = bool(obs.work.get_legal_moves())
        out["is_checkmate"] = opp_in_check and not opp_has_moves
        out["is_stalemate"] = (not opp_in_check) and not opp_has_moves
    _audit_result(
        game,
        side,
        tool,
        result={"path": out["path"], "ok": out["ok"]},
        ok=out["ok"],
        error_class=out["error_class"],
    )
    return out


@app.post("/api/game/{game_id}/preview_reset")
async def preview_reset(
    game_id: str,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Reset the seat's ObserveSession work board to live."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side, tool = _require_external_seat(
        game, token=x_xiangqi_seat_token, mcp_tool=x_xiangqi_mcp_tool
    )
    _audit_call(game, side, tool, args={"op": "preview_reset"})
    obs = external_mcp.get_observe(game, side)
    text = obs.reset()
    out = {
        "ok": True,
        "game_id": game.id,
        "seat": side,
        "text": text,
        "path": list(obs.path),
        "fen_work": obs.work.to_fen(),
        "fen_live": game.board.to_fen(),
    }
    _audit_result(game, side, tool, result={"path": []}, ok=True)
    return out


@app.get("/api/game/{game_id}/threats")
async def get_threats(
    game_id: str,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Threat summary on the seat's work board."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side, tool = _require_external_seat(
        game, token=x_xiangqi_seat_token, mcp_tool=x_xiangqi_mcp_tool
    )
    _audit_call(game, side, tool, args={"op": "get_threats"})
    obs = external_mcp.get_observe(game, side)
    text = observe_threats(obs.work, obs.work.turn)
    out = {
        "ok": True,
        "game_id": game.id,
        "seat": side,
        "text": obs.wrap(text) if hasattr(obs, "wrap") else text,
        "path": list(obs.path),
        "fen_work": obs.work.to_fen(),
    }
    _audit_result(game, side, tool, result={"path_len": len(obs.path)}, ok=True)
    return out


@app.post("/api/game/{game_id}/wait-turn")
async def wait_turn(
    game_id: str,
    req: WaitTurnRequest,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Long-poll until ``side`` to move, or the game leaves ``playing``.

    Used by MCP ``wait_my_turn`` / ``submit_move(wait_opponent=true)`` so agents
    do not busy-poll. Wakes on move / finish / pause / interrupt / resume / seek.
    Timeout does not change the game result.
    """
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side = (req.side or "").strip().lower()
    if side not in {"red", "black"}:
        _http_err(400, "side must be 'red' or 'black'", error_class="rules")
    seat_side, tool = _require_external_seat(
        game,
        token=x_xiangqi_seat_token,
        expected_side=side,
        mcp_tool=x_xiangqi_mcp_tool,
    )
    try:
        timeout_sec = float(req.timeout_sec)
    except (TypeError, ValueError):
        _http_err(400, "timeout_sec must be a number", error_class="rules")
    if timeout_sec <= 0:
        _http_err(400, "timeout_sec must be > 0", error_class="rules")
    # Cap to keep a single HTTP request bounded (MCP can re-call).
    timeout_sec = min(timeout_sec, 3600.0)

    _audit_call(
        game,
        seat_side,
        tool,
        args={"side": side, "timeout_sec": timeout_sec},
    )
    deadline = time.monotonic() + timeout_sec
    while True:
        done = _wait_turn_done_reason(game, side)
        if done is not None:
            payload = _wait_turn_payload(game, waiting_done_reason=done)
            _audit_result(
                game,
                seat_side,
                tool,
                result={"waiting_done_reason": done},
                ok=True,
                error_class=payload.get("error_class"),
            )
            return payload

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            payload = {
                **_wait_turn_payload(game, waiting_done_reason="timeout"),
                "ok": True,
                "suggested_retry_sec": 60,
            }
            _audit_result(
                game,
                seat_side,
                tool,
                result={"waiting_done_reason": "timeout"},
                ok=True,
            )
            return payload

        game.waiter_event.clear()
        # Re-check after clear to avoid missing a wake that raced with clear.
        done = _wait_turn_done_reason(game, side)
        if done is not None:
            payload = _wait_turn_payload(game, waiting_done_reason=done)
            _audit_result(
                game,
                seat_side,
                tool,
                result={"waiting_done_reason": done},
                ok=True,
            )
            return payload

        try:
            await asyncio.wait_for(game.waiter_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            payload = {
                **_wait_turn_payload(game, waiting_done_reason="timeout"),
                "ok": True,
                "suggested_retry_sec": 60,
            }
            _audit_result(
                game,
                seat_side,
                tool,
                result={"waiting_done_reason": "timeout"},
                ok=True,
            )
            return payload


@app.post("/api/game/{game_id}/move")
async def submit_external_move(
    game_id: str,
    req: SubmitMoveRequest,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Deposit one move for an external (MCP) player; game_loop applies it.

    Illegal moves are rejected with the legal-move list so the agent can
    retry without ending the game. Only one pending move per side is accepted.
    Requires a valid seat token bound to ``side``.
    """
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    side = (req.side or "").strip().lower()
    if side not in {"red", "black"}:
        _http_err(400, "side must be 'red' or 'black'", error_class="rules")
    seat_side, tool = _require_external_seat(
        game,
        token=x_xiangqi_seat_token,
        expected_side=side,
        mcp_tool=x_xiangqi_mcp_tool,
    )
    config = game.red_config if side == "red" else game.black_config
    if config.type != "external":
        _http_err(
            400,
            f"{side} is not an external player (type={config.type})",
            error_class="rules",
        )
    move = (req.move or "").strip().lower()
    _audit_call(
        game,
        seat_side,
        tool,
        args={"side": side, "move": move, "note": (req.note or "").strip() or None},
    )
    if game.status != "playing":
        detail = {
            "error": (
                f"Game is not playing (status={game.status}). "
                "submit_move 仅 status=playing 可用：复盘/预盘/终局/中断阶段禁用走子"
            ),
            "error_class": "state",
            "status": game.status,
        }
        _audit_result(game, seat_side, tool, result=detail, ok=False, error_class="state")
        raise HTTPException(400, detail=detail)
    turn_side = _turn_side(game)
    if side != turn_side:
        detail = {
            "error": f"Not {side}'s turn (it is {turn_side}'s turn)",
            "error_class": "rules",
        }
        _audit_result(game, seat_side, tool, result=detail, ok=False, error_class="rules")
        raise HTTPException(400, detail=detail)
    queue = game.external_moves[side]
    if game.external_pending.get(side) is not None or queue.qsize():
        detail = {
            "error": f"{side} already has a pending move waiting to be applied",
            "error_class": "rules",
        }
        _audit_result(game, seat_side, tool, result=detail, ok=False, error_class="rules")
        raise HTTPException(400, detail=detail)

    legal = game.board.get_legal_moves()
    if move not in legal:
        diag = game.board.explain_illegal_move(move) or {}
        detail = {
            "error": f"Illegal move: {move or '(empty)'}",
            "error_class": "rules",
            "reason": diag.get("reason"),
            "reason_code": diag.get("code"),
            "legal_moves": legal,
            "legal_count": len(legal),
            "board": game.board.to_text(),
            "fen": game.board.to_fen(),
        }
        _audit_result(game, seat_side, tool, result=detail, ok=False, error_class="rules")
        raise HTTPException(400, detail=detail)
    payload = {"move": move, "note": (req.note or "").strip()}
    game.external_pending[side] = payload
    queue.put_nowait(dict(payload))
    out = {
        "ok": True,
        "accepted": move,
        "turn": side,
        "pending": 1 if game.external_pending.get(side) else queue.qsize(),
    }
    _audit_result(game, seat_side, tool, result=out, ok=True)
    return out


@app.post("/api/game/{game_id}/resign")
async def resign_game(
    game_id: str,
    req: ResignRequest,
    x_xiangqi_seat_token: SeatTokenHdr = None,
    x_xiangqi_mcp_tool: McpToolHdr = None,
):
    """Finish the game: given side resigns, the other side wins.

    When the game has an external seat, resignation requires that seat's token
    and can only resign the token-bound side.
    """
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Game not found")
    if game.status != "playing":
        _http_err(
            400,
            f"Game is not playing (status={game.status})",
            error_class="state",
        )
    side = (req.side or "").strip().lower()
    if not side:
        side = _turn_side(game)
    if side not in {"red", "black"}:
        _http_err(400, "side must be 'red' or 'black'", error_class="rules")

    if external_mcp.game_has_external(game):
        seat_side, tool = _require_external_seat(
            game,
            token=x_xiangqi_seat_token,
            expected_side=side,
            mcp_tool=x_xiangqi_mcp_tool,
        )
        _audit_call(game, seat_side, tool, args={"side": side})
        if external_mcp.side_config(game, side).type != "external":
            detail = {
                "error": "can only resign your own external seat",
                "error_class": "auth",
            }
            _audit_result(
                game, seat_side, tool, result=detail, ok=False, error_class="auth"
            )
            raise HTTPException(401, detail=detail)
    else:
        seat_side, tool = side, "resign"

    winner = "black" if side == "red" else "red"
    _finish_game(game, winner, "resignation")
    if game.task and not game.task.done():
        game.task.cancel()
        try:
            await game.task
        except asyncio.CancelledError:
            pass
    game.task = None
    out = {"status": "finished", "winner": winner, "reason": "resignation"}
    if external_mcp.game_has_external(game):
        _audit_result(game, seat_side, tool, result=out, ok=True)
    return out


# --- Game log history APIs ---

@app.get("/api/games")
async def list_active_games():
    """List all active game sessions (not yet cleaned up)."""
    result = []
    for gid, g in games.items():
        external_seats = []
        for side in ("red", "black"):
            cfg = g.red_config if side == "red" else g.black_config
            if cfg.type != "external":
                continue
            claimed = mcp_contract.seat_claimed(g, side)
            external_seats.append({
                "side": side,
                "claimed": claimed,
                "joinable": not claimed,
                "name": _player_display_name(cfg) or None,
            })
        result.append({
            "id": gid,
            "status": g.status,
            "red_label": _player_label(g.red_config),
            "black_label": _player_label(g.black_config),
            "move_count": len(g.move_history),
            "winner": g.winner,
            "reason": g.reason,
            "turn": _turn_side(g) if g.status == "playing" else None,
            "external_seats": external_seats,
        })
    return {"games": result}


@app.get("/")
async def root():
    return {
        "service": "xiangqi-llm-duel",
        "description": "象棋大模型对决",
        "docs": "/docs",
    }

if __name__ == "__main__":
    import socket
    import uvicorn

    host, port = "127.0.0.1", 8000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
    except OSError:
        print(f"\n  [ERROR] Port {port} is already in use.")
        print(f"  Please close the other process or use a different port.\n")
        exit(1)

    print("\n  xiangqi-llm-duel - 象棋大模型对决")
    print("  =================================")
    print(f"  http://localhost:{port}")
    print(f"  API docs: http://localhost:{port}/docs")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port)
