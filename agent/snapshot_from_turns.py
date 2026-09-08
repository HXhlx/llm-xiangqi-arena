"""Rebuild a server snapshot from turns JSONL when the live session is gone."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _player(preset: str) -> dict[str, Any]:
    return {
        "type": "llm",
        "name": preset,
        "preset": preset,
        "api_key": None,
    }


def rebuild_snapshot_from_turns(
    turns_path: Path,
    *,
    game_id: str,
    red_preset: str,
    black_preset: str,
    dest: Path,
    reason: str = "restored from turns",
) -> dict[str, Any]:
    """Replay compact turn rows onto a board and write an interrupted snapshot."""
    board = Board(START_FEN)
    history: list[dict[str, Any]] = []
    for line in turns_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        move = row.get("move")
        if not move:
            continue
        result = board.make_move(str(move))
        history.append(
            {
                "number": int(row.get("ply") or len(history) + 1),
                "side": row.get("side") or ("red" if result["piece"].isupper() else "black"),
                "move": result["move"],
                "move_zh": row.get("move_zh") or result.get("move_zh"),
                "piece": result["piece"],
                "captured": result["captured"],
                "fen": board.to_fen(),
                "fen_before": result.get("fen_before") or row.get("fen_before"),
            }
        )
    payload = {
        "v": 1,
        "game_id": game_id,
        "status": "interrupted",
        "reason": reason,
        "initial_fen": START_FEN,
        "fen": board.to_fen(),
        "turn": "red" if board.turn == "w" else "black",
        "winner": None,
        "move_history": history,
        "red": _player(red_preset),
        "black": _player(black_preset),
        "pikafish": None,
        "timer_config": {"enabled": True, "initial_time": 0, "increment": 0},
        "timer": {"red": 0.0, "black": 0.0},
        "ts": time.time(),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload
