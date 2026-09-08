"""Redis Stack checkpointer factory (AsyncRedisSaver)."""

from __future__ import annotations

import os
import urllib.parse
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from .redis_gc import (
    langgraph_ttl_config,
    prune_thread_checkpoints_async,
    wipe_thread_keys_async,
)

_checkpointer: Optional[AsyncRedisSaver] = None
_cm = None


def get_redis_url() -> str:
    url = (os.getenv("REDIS_URL") or "").strip()
    if url:
        return url
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")
    if password:
        pw = urllib.parse.quote(password, safe="")
        return f"redis://:{pw}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def redis_prefix() -> str:
    """Logical app prefix (used for thread_id namespacing)."""
    return os.getenv("XIANGQI_REDIS_PREFIX", "xiangqi:lg:")


async def get_checkpointer() -> AsyncRedisSaver:
    """Return a process-wide AsyncRedisSaver (Redis Stack / RediSearch required)."""
    global _checkpointer, _cm
    if _checkpointer is not None:
        return _checkpointer
    url = get_redis_url()
    _cm = AsyncRedisSaver.from_conn_string(url, ttl=langgraph_ttl_config())
    _checkpointer = await _cm.__aenter__()
    await _checkpointer.asetup()
    return _checkpointer


async def close_checkpointer() -> None:
    global _checkpointer, _cm
    if _cm is not None:
        try:
            await _cm.__aexit__(None, None, None)
        except Exception:
            pass
    _checkpointer = None
    _cm = None


def _saver_client(cp: AsyncRedisSaver):
    return getattr(cp, "_redis", None) or getattr(cp, "conn", None) or getattr(cp, "redis", None)


async def delete_thread(thread_id: str) -> None:
    """Drop a thread via the index, then scan-wipe leftovers past the 10k cap."""
    cp = await get_checkpointer()
    try:
        await cp.adelete_thread(thread_id)
    finally:
        client = _saver_client(cp)
        if client is not None and callable(getattr(client, "scan", None)):
            try:
                await wipe_thread_keys_async(client, thread_id)
            except Exception as exc:
                print(f"  [memory] scan wipe failed {thread_id}: {exc}")


async def prune_thread_to_latest(thread_id: str) -> None:
    """Keep only the latest checkpoint for a thread after each ply."""
    cp = await get_checkpointer()
    aprune = getattr(cp, "aprune", None)
    if callable(aprune):
        try:
            await aprune([thread_id], strategy="keep_latest")
        except Exception as exc:
            print(f"  [memory] aprune failed {thread_id}: {exc}")
    client = _saver_client(cp)
    if client is None or not callable(getattr(client, "scan", None)):
        return
    try:
        await prune_thread_checkpoints_async(client, thread_id)
    except Exception as exc:
        print(f"  [memory] scan prune failed {thread_id}: {exc}")


async def ping_redis() -> bool:
    try:
        cp = await get_checkpointer()
        # setup succeeded implies connectivity; also try a lightweight graph thread delete of missing id
        client = _saver_client(cp)
        if client is not None and hasattr(client, "ping"):
            pong = await client.ping()
            return bool(pong)
        return True
    except Exception:
        return False


@asynccontextmanager
async def temporary_checkpointer() -> AsyncIterator[AsyncRedisSaver]:
    async with AsyncRedisSaver.from_conn_string(
        get_redis_url(), ttl=langgraph_ttl_config()
    ) as cp:
        await cp.asetup()
        yield cp
