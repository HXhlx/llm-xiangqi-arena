"""Disk archives of the full Redis conversation window taken just before compress."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .compress import estimate_tokens

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_DIRNAME = "memory"
_SEQ_RE = re.compile(r"^([A-Za-z0-9_-]+)-(\d+)\.json$")
_lock = threading.Lock()


def safe_id(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (value or "unknown"))


def archive_dir(game_id: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else BASE_DIR
    return base / "logs" / MEMORY_DIRNAME / safe_id(game_id)


def next_memory_seq(dest_dir: Path, side: str) -> int:
    highest = 0
    if dest_dir.is_dir():
        for path in dest_dir.iterdir():
            match = _SEQ_RE.match(path.name)
            if not match or match.group(1) != safe_id(side):
                continue
            highest = max(highest, int(match.group(2)))
    return highest + 1


def write_pre_compress_snapshot(
    game_id: str,
    side: str,
    *,
    fen: str,
    messages: list[dict[str, Any]],
    root: Path | None = None,
) -> Path:
    """Write the uncompressed window. Raises on IO failure."""
    dest = archive_dir(game_id, root)
    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "v": 1,
        "kind": "pre_compress",
        "game_id": game_id,
        "side": side,
        "fen": fen,
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "tokens": estimate_tokens(messages),
        "messages": list(messages),
    }
    text = json.dumps(payload, ensure_ascii=False)
    with _lock:
        seq = next_memory_seq(dest, side)
        path = dest / f"{safe_id(side)}-{seq:03d}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return path


def try_write_pre_compress_snapshot(
    game_id: str,
    side: str,
    *,
    fen: str,
    messages: list[dict[str, Any]],
    root: Path | None = None,
) -> bool:
    """Write the pre-compress window. False if Redis must keep the full copy."""
    try:
        write_pre_compress_snapshot(game_id, side, fen=fen, messages=messages, root=root)
        return True
    except Exception as exc:
        print(f"  [memory] archive failed game={game_id} side={side}: {exc}")
        return False
