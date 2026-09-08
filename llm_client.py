"""
LLM client with OpenAI SDK for Xiangqi.
Supports OpenAI-compatible endpoints via configurable base_url.
"""

import asyncio
import json
import re

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from adapters import get_adapter
from agent.observe_session import ObserveSession, is_submit_ok
from prompt_registry import get_prompt_profile, load_prompt
from xiangqi import Board, PIECE_NAMES_ZH


def _tool_definitions() -> list[dict]:
    spec = load_prompt("player.tools")

    def desc(name: str, field: str) -> str:
        return str(spec[name][field]).strip()

    def tool(name: str, properties: dict, required: list[str] | None = None) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": str(spec[name]["description"]).strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required or [],
                },
            },
        }

    return [
        tool(
            "make_move",
            {"move": {"type": "string", "description": desc("make_move", "move")}},
            ["move"],
        ),
        tool(
            "preview",
            {"move": {"type": "string", "description": desc("preview", "move")}},
            ["move"],
        ),
        tool("preview_reset", {}),
        tool(
            "get_legal_moves",
            {
                "squares": {"type": "string", "description": desc("get_legal_moves", "squares")},
                "pieces": {"type": "string", "description": desc("get_legal_moves", "pieces")},
            },
        ),
        tool("get_threats", {}),
        tool("get_board", {}),
    ]


TOOL_DEFINITIONS = _tool_definitions()


ICCS_PATTERN = re.compile(r"[a-i][0-9][a-i][0-9]")

# Display labels for LLM prompts only (engine FEN stays letter-based).
PIECE_LABEL_ZH = {
    "K": "红帅", "A": "红仕", "B": "红相", "N": "红马", "R": "红车", "C": "红炮", "P": "红兵",
    "k": "黑将", "a": "黑士", "b": "黑象", "n": "黑马", "r": "黑车", "c": "黑炮", "p": "黑卒",
}

DEFAULT_MAX_TOOL_ROUNDS = 10000


def _piece_label(piece: str | None) -> str:
    if not piece:
        return "·"
    return PIECE_LABEL_ZH.get(piece, PIECE_NAMES_ZH.get(piece, piece))


def _get_piece_positions(board: Board) -> str:
    """Return piece positions grouped by side (Chinese labels for LLM)."""
    red_pieces = []
    black_pieces = []

    for row in range(10):
        for col in range(9):
            piece = board.get_piece(col, row)
            if not piece:
                continue
            coord = f"{chr(col + ord('a'))}{row}"
            entry = f"{coord}: {_piece_label(piece)}"
            if piece.isupper():
                red_pieces.append(entry)
            else:
                black_pieces.append(entry)

    return "\n".join([
        "Red: " + (", ".join(red_pieces) if red_pieces else "(none)"),
        "Black: " + (", ".join(black_pieces) if black_pieces else "(none)"),
    ])


def _render_board_diagram(board: Board) -> str:
    """Scheme-C Chinese board diagram with 红/黑 prefixes (LLM display only)."""
    # Each cell is exactly 2 CJK chars (or ··); separate with single space for a–i alignment.
    def cell(piece) -> str:
        if not piece:
            return "··"
        return _piece_label(piece)  # always 2 CJK chars (红X / 黑X)

    header = "   " + " ".join(f" {c} " for c in "abcdefghi")
    lines = [header]
    for row in range(9, -1, -1):
        cells = [cell(board.get_piece(col, row)) for col in range(9)]
        lines.append(f"{row}  " + " ".join(cells) + f"  {row}")
        if row == 5:
            lines.append("   " + "─" * 35)
            lines.append("   " + " " * 11 + "楚 河 汉 界")
            lines.append("   " + "─" * 35)
    lines.append(header)
    side = "Red" if board.turn == "w" else "Black"
    lines.append(f"Side to move: {side}")
    return "\n".join(lines)


def _get_last_opponent_move(board: Board) -> str:
    if not board.move_history:
        return ""
    last = board.move_history[-1]
    move = last.get("move", "")
    move_zh = last.get("move_zh", "")
    if move_zh:
        return f"{move_zh}（{move}）"
    return move


def _cursor_status_copy(prompt_profile: dict) -> str:
    name = str(prompt_profile.get("name") or "")
    if name == "en":
        return (
            "Working board starts at live; after that, trust the tool-result header. "
            "Multiple previews in one turn share one path; header to_move is the side "
            "to move on the working board."
        )
    return (
        "开局工作盘=live；之后以工具返回头为准。"
        "同一轮多个 preview 串在一条 path 上；工具头 to_move 才是当前工作盘行棋方。"
    )


def _build_prompt_params(board: Board, side: str, prompt_name: str | None = None) -> tuple[dict, dict]:
    from agent.observe import file_legend, own_pieces

    side_name = "Red" if side == "w" else "Black"
    side_name_zh = "红方" if side == "w" else "黑方"
    prompt_profile = get_prompt_profile(prompt_name)
    params = dict(
        side_name=side_name,
        side_name_zh=side_name_zh,
        fen=board.to_fen(),
        last_opponent_move=_get_last_opponent_move(board) or "尚无（开局）",
        file_legend=file_legend(side),
        own_pieces=own_pieces(board, side),
        cursor_status=_cursor_status_copy(prompt_profile),
    )
    return prompt_profile, params


def build_system_prompt(board: Board, side: str, prompt_name: str | None = None) -> str:
    prompt_profile, params = _build_prompt_params(board, side, prompt_name)
    return prompt_profile["system_prompt"].format(**params)


def _turn_prompt(board: Board, side: str, prompt_name: str | None = None) -> str:
    prompt_profile, params = _build_prompt_params(board, side, prompt_name)
    return prompt_profile["turn_prompt"].format(**params)


def _tool_retry_prompt(board: Board, side: str, prompt_name: str | None = None) -> str:
    prompt_profile, params = _build_prompt_params(board, side, prompt_name)
    return prompt_profile["tool_retry_prompt"].format(**params)


def execute_tool(target: Board | ObserveSession, tool_name: str, args: dict | None = None) -> str:
    """Execute a tool against a live board or an observe session."""
    session = target if isinstance(target, ObserveSession) else ObserveSession(target)
    return session.dispatch(tool_name, args or {})
