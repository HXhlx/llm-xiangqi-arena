"""Per-ply turn records: thoughts, tools, moves → logs/turns/{game_id}.jsonl."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TURNS_DIR = BASE_DIR / "logs" / "turns"

_lock = threading.Lock()

def turns_path(game_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (game_id or "unknown"))
    return TURNS_DIR / f"{safe}.jsonl"


def append_turn(game_id: str, record: dict[str, Any]) -> Path:
    """Append one JSON line for a completed ply. Returns path written."""
    path = turns_path(game_id)
    row = dict(record)
    row.setdefault("v", 1)
    row.setdefault("game_id", game_id)
    row.setdefault("ts", datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    return path


def iter_turns(game_id: str) -> list[dict[str, Any]]:
    path = turns_path(game_id)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def load_last_turn(game_id: str, side: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Return last turn record, optionally filtered by side (red/black)."""
    rows = iter_turns(game_id)
    if side:
        side_l = side.lower()
        rows = [r for r in rows if (r.get("side") or "").lower() == side_l]
    return rows[-1] if rows else None


def format_inject_block(record: dict[str, Any]) -> str:
    """Format a prior own-side turn for next-turn prompt injection (full text)."""
    ply = record.get("ply") or record.get("number") or "?"
    move = record.get("move") or ""
    move_zh = record.get("move_zh") or ""
    reasoning = str(record.get("reasoning") or "")
    thinking = str(record.get("thinking") or "")
    tools = record.get("tool_rounds") or []

    tool_lines: list[str] = []
    for rnd in tools:
        if not isinstance(rnd, dict):
            continue
        for tc in rnd.get("tool_calls") or []:
            name = tc.get("name") or "?"
            args = tc.get("args") if "args" in tc else tc.get("arguments")
            if not isinstance(args, str):
                args_s = json.dumps(args or {}, ensure_ascii=False)
            else:
                args_s = args
            tool_lines.append(f"- call {name}({args_s})")
        for tr in rnd.get("tool_results") or []:
            name = tr.get("name") or "?"
            content = str(tr.get("content") or "")
            tool_lines.append(f"- result {name}: {content}")

    parts = [
        "[本方上一手记录 — 完整思考原文]",
        f"ply={ply} 着法={move}" + (f" ({move_zh})" if move_zh else ""),
    ]
    if reasoning.strip():
        parts.append("思考原文（reasoning）：")
        parts.append(reasoning.strip())
    if thinking.strip():
        parts.append("可见输出（thinking/content）：")
        parts.append(thinking.strip())
    if tool_lines:
        parts.append("工具调用：")
        parts.extend(tool_lines)
    return "\n".join(parts)
