"""TTL config and stale LangGraph checkpoint purge for Redis Stack."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping, Optional

DEFAULT_REDIS_TTL_SECONDS = 0
DEFAULT_REDIS_PREFIX = "xiangqi:lg:"
LIVE_GAME_STATUSES = frozenset({"waiting", "playing", "paused", "interrupted"})
THREAD_KEY_KINDS = (
    "checkpoint",
    "checkpoint_write",
    "write_keys_zset",
    "checkpoint_latest",
)


class GamesFetchError(ValueError):
    """/api/games was unreachable or returned an invalid payload."""


def checkpoint_prefix(prefix: str | None = None) -> str:
    if prefix is not None:
        return prefix
    return os.getenv("XIANGQI_REDIS_PREFIX", DEFAULT_REDIS_PREFIX)


def require_checkpoint_prefix(prefix: str | None = None) -> str:
    marker = checkpoint_prefix(prefix)
    if not marker.strip() or any(ch in marker for ch in "*?[]"):
        raise ValueError("XIANGQI_REDIS_PREFIX is empty or unsafe")
    return marker


def parse_games_payload(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping) or "games" not in payload:
        raise GamesFetchError("invalid /api/games payload")
    games = payload.get("games")
    if not isinstance(games, list):
        raise GamesFetchError("games is not a list")
    return [row for row in games if isinstance(row, Mapping)]


def redis_ttl_seconds(env: Mapping[str, str] | None = None) -> int:
    """Checkpoint TTL in seconds. 0 disables expiry."""
    src = env if env is not None else os.environ
    raw = str(src.get("XIANGQI_REDIS_TTL", DEFAULT_REDIS_TTL_SECONDS) or "").strip()
    if not raw:
        return DEFAULT_REDIS_TTL_SECONDS
    try:
        seconds = int(raw)
    except ValueError:
        return DEFAULT_REDIS_TTL_SECONDS
    return max(0, seconds)


def langgraph_ttl_config(env: Mapping[str, str] | None = None) -> Optional[dict[str, Any]]:
    """LangGraph Redis saver TTL dict (minutes). None means persist forever."""
    seconds = redis_ttl_seconds(env)
    if seconds <= 0:
        return None
    return {
        "default_ttl": seconds / 60.0,
        "refresh_on_read": True,
    }


def game_id_from_redis_key(key: str, prefix: str | None = None) -> Optional[str]:
    marker = checkpoint_prefix(prefix)
    idx = key.find(marker)
    if idx < 0:
        return None
    rest = key[idx + len(marker) :]
    if not rest:
        return None
    game_id = rest.split(":", 1)[0].strip()
    return game_id or None


def should_retain_redis_key(
    key: str,
    keep_game_ids: Iterable[str],
    prefix: str | None = None,
) -> bool:
    marker = checkpoint_prefix(prefix)
    if marker not in key:
        return True
    keep = keep_game_ids if isinstance(keep_game_ids, (set, frozenset)) else set(keep_game_ids)
    game_id = game_id_from_redis_key(key, marker)
    if game_id is None:
        return True
    return game_id in keep


def resumable_snapshot_game_ids(snapshot_dir: str | os.PathLike[str] | None) -> set[str]:
    """Game ids whose on-disk snapshots can still be restored after a restart."""
    if not snapshot_dir:
        return set()
    root = os.fspath(snapshot_dir)
    if not os.path.isdir(root):
        return set()
    keep: set[str] = set()
    for name in os.listdir(root):
        if not name.endswith(".json"):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        stem = os.path.splitext(name)[0].strip()
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.loads(handle.read())
        except (OSError, UnicodeError, json.JSONDecodeError):
            if stem:
                keep.add(stem)
            continue
        if not isinstance(data, Mapping):
            if stem:
                keep.add(stem)
            continue
        game_id = str(data.get("game_id") or stem).strip()
        status = str(data.get("status") or "interrupted").strip()
        if game_id and status in LIVE_GAME_STATUSES:
            keep.add(game_id)
    return keep


def live_game_ids(games: Iterable[Mapping[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in games:
        if not isinstance(row, Mapping):
            continue
        game_id = str(row.get("id") or "").strip()
        status = str(row.get("status") or "").strip()
        if game_id and status in LIVE_GAME_STATUSES:
            out.add(game_id)
    return out


def _thread_key_rest(key: str, kind: str, thread_id: str) -> Optional[str]:
    head = f"{kind}:{thread_id}:"
    if key.startswith(head):
        return key[len(head) :]
    return None


def _checkpoint_id_from_thread_key(key: str, thread_id: str) -> Optional[str]:
    rest = _thread_key_rest(key, "checkpoint", thread_id)
    if rest is not None:
        parts = rest.split(":")
        return parts[-1] if len(parts) >= 2 else None
    rest = _thread_key_rest(key, "write_keys_zset", thread_id)
    if rest is not None:
        parts = rest.split(":")
        return parts[-1] if parts else None
    rest = _thread_key_rest(key, "checkpoint_write", thread_id)
    if rest is not None:
        parts = rest.split(":")
        return parts[1] if len(parts) >= 2 else None
    return None


def thread_owned_keys(keys: Iterable[str], thread_id: str) -> list[str]:
    """LangGraph keys that belong to this exact thread (kind:thread_id:...)."""
    thread_id = _require_thread_id(thread_id)
    owned: list[str] = []
    for raw in keys:
        key = _as_text(raw)
        if any(_thread_key_rest(key, kind, thread_id) is not None for kind in THREAD_KEY_KINDS):
            owned.append(key)
    return owned


def older_checkpoint_keys(keys: Iterable[str], thread_id: str) -> list[str]:
    """Keys for this thread that are not the latest checkpoint or its pointer."""
    owned: list[str] = []
    latest_ids: list[str] = []
    for raw in keys:
        key = _as_text(raw)
        if _thread_key_rest(key, "checkpoint_latest", thread_id) is not None:
            owned.append(key)
            continue
        ckpt_id = _checkpoint_id_from_thread_key(key, thread_id)
        if ckpt_id is None:
            continue
        owned.append(key)
        if _thread_key_rest(key, "checkpoint", thread_id) is not None:
            latest_ids.append(ckpt_id)
    keep_id = max(latest_ids) if latest_ids else None
    drop: list[str] = []
    for key in owned:
        if _thread_key_rest(key, "checkpoint_latest", thread_id) is not None:
            continue
        ckpt_id = _checkpoint_id_from_thread_key(key, thread_id)
        if keep_id is None or ckpt_id != keep_id:
            drop.append(key)
    return drop


def _require_thread_id(thread_id: str) -> str:
    if not thread_id or any(ch in thread_id for ch in "*?[]"):
        raise ValueError("thread_id is empty or unsafe")
    return thread_id


def prune_thread_checkpoints(client: Any, thread_id: str, *, batch_size: int = 500) -> int:
    """Delete older LangGraph keys for one thread; keep latest + checkpoint_latest."""
    thread_id = _require_thread_id(thread_id)
    found = _scan_thread_keys(client, thread_id, batch_size=batch_size)
    drop = older_checkpoint_keys(found, thread_id)
    _delete_keys(client, drop)
    return len(drop)


def wipe_thread_keys(client: Any, thread_id: str, *, batch_size: int = 500) -> int:
    """Delete every LangGraph key for one thread (finish / seek / reset)."""
    thread_id = _require_thread_id(thread_id)
    found = _scan_thread_keys(client, thread_id, batch_size=batch_size)
    drop = thread_owned_keys(found, thread_id)
    _delete_keys(client, drop)
    return len(drop)


async def wipe_thread_keys_async(
    client: Any, thread_id: str, *, batch_size: int = 500
) -> int:
    thread_id = _require_thread_id(thread_id)
    found = await _scan_thread_keys_async(client, thread_id, batch_size=batch_size)
    drop = thread_owned_keys(found, thread_id)
    if not drop:
        return 0
    await _delete_keys_async(client, drop)
    return len(drop)


async def prune_thread_checkpoints_async(
    client: Any, thread_id: str, *, batch_size: int = 500
) -> int:
    thread_id = _require_thread_id(thread_id)
    found = await _scan_thread_keys_async(client, thread_id, batch_size=batch_size)
    drop = older_checkpoint_keys(found, thread_id)
    if not drop:
        return 0
    await _delete_keys_async(client, drop)
    return len(drop)


def _scan_thread_keys(client: Any, thread_id: str, *, batch_size: int) -> list[str]:
    found: list[str] = []
    cursor = 0
    match = f"*{thread_id}*"
    while True:
        cursor, keys = client.scan(cursor=cursor, match=match, count=batch_size)
        found.extend(_as_text(raw) for raw in keys)
        if int(cursor) == 0:
            break
    return found


async def _scan_thread_keys_async(
    client: Any, thread_id: str, *, batch_size: int
) -> list[str]:
    found: list[str] = []
    cursor = 0
    match = f"*{thread_id}*"
    while True:
        cursor, keys = await client.scan(cursor=cursor, match=match, count=batch_size)
        found.extend(_as_text(raw) for raw in keys)
        if int(cursor) == 0:
            break
    return found


async def _delete_keys_async(client: Any, keys: list[str]) -> None:
    if not keys:
        return
    unlink = getattr(client, "unlink", None)
    op = unlink if callable(unlink) else client.delete
    result = op(*keys)
    if hasattr(result, "__await__"):
        await result


def _as_text(key: Any) -> str:
    if isinstance(key, bytes):
        return key.decode("utf-8", errors="replace")
    return str(key)


def _delete_keys(client: Any, keys: list[str]) -> None:
    if not keys:
        return
    unlink = getattr(client, "unlink", None)
    if callable(unlink):
        unlink(*keys)
        return
    client.delete(*keys)


def purge_inactive_keys(
    client: Any,
    keep_game_ids: Iterable[str],
    *,
    prefix: str | None = None,
    ttl_seconds: int | None = None,
    batch_size: int = 500,
    refresh_keep: Any = None,
) -> dict[str, int]:
    """Delete our checkpoint keys except live games; apply TTL to leftovers."""
    marker = require_checkpoint_prefix(prefix)
    if ttl_seconds is None:
        ttl_seconds = redis_ttl_seconds()
    keep = {str(gid).strip() for gid in keep_game_ids if str(gid).strip()}
    deleted = 0
    retained = 0
    ttl_applied = 0
    drop_batch: list[str] = []
    expire_batch: list[str] = []

    def current_keep() -> set[str]:
        extra: set[str] = set()
        if refresh_keep is not None:
            extra = {
                str(gid).strip()
                for gid in (refresh_keep() or [])
                if str(gid).strip()
            }
        return keep | extra

    def flush_drop() -> None:
        nonlocal deleted
        if not drop_batch:
            return
        still_drop = [
            key
            for key in drop_batch
            if not should_retain_redis_key(key, current_keep(), marker)
        ]
        drop_batch.clear()
        if not still_drop:
            return
        _delete_keys(client, still_drop)
        deleted += len(still_drop)

    def flush_expire() -> None:
        nonlocal ttl_applied
        if not expire_batch or ttl_seconds <= 0:
            expire_batch.clear()
            return
        pipe = client.pipeline(transaction=False) if hasattr(client, "pipeline") else None
        if pipe is None:
            for key in expire_batch:
                client.expire(key, ttl_seconds)
                ttl_applied += 1
        else:
            for key in expire_batch:
                pipe.expire(key, ttl_seconds)
            pipe.execute()
            ttl_applied += len(expire_batch)
        expire_batch.clear()

    cursor = 0
    match = f"*{marker}*"
    while True:
        cursor, keys = client.scan(cursor=cursor, match=match, count=batch_size)
        for raw in keys:
            key = _as_text(raw)
            if should_retain_redis_key(key, keep, marker):
                retained += 1
                if ttl_seconds > 0:
                    expire_batch.append(key)
                    if len(expire_batch) >= batch_size:
                        flush_expire()
                continue
            drop_batch.append(key)
            if len(drop_batch) >= batch_size:
                flush_drop()
        if int(cursor) == 0:
            break
    flush_drop()
    flush_expire()
    return {
        "deleted": deleted,
        "retained": retained,
        "ttl_applied": ttl_applied,
    }
