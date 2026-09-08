#!/usr/bin/env python3
"""Drop LangGraph checkpoints except live API games and resumable snapshots."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.memory import get_redis_url, redis_prefix
from agent.redis_gc import (
    GamesFetchError,
    live_game_ids,
    parse_games_payload,
    purge_inactive_keys,
    redis_ttl_seconds,
    require_checkpoint_prefix,
    resumable_snapshot_game_ids,
)


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def fetch_api_games(base: str) -> list[dict]:
    url = base.rstrip("/") + "/api/games"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise GamesFetchError(f"API unavailable ({url}): {exc}") from exc
    return list(parse_games_payload(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow deleting all xiangqi checkpoints when /api/games has no live sessions",
    )
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        prefix = require_checkpoint_prefix(redis_prefix())
    except ValueError as exc:
        print(exc)
        return 2

    base = os.environ.get("XIANGQI_BASE", "http://127.0.0.1:8010")
    try:
        games = fetch_api_games(base)
    except GamesFetchError as exc:
        print(f"{exc}; aborting so live threads are not wiped")
        return 2

    snap_dir = Path(os.environ.get("XIANGQI_SNAPSHOT_DIR") or ROOT / "logs" / "snapshots")
    snap_keep = resumable_snapshot_game_ids(snap_dir)
    keep = live_game_ids(games) | snap_keep
    ttl = redis_ttl_seconds()
    print(f"prefix={prefix} ttl_seconds={ttl} live={sorted(keep) or '-'}")
    for row in games:
        gid = row.get("id")
        if gid in keep:
            print(f"  keep {gid} status={row.get('status')} moves={row.get('move_count')}")
    for gid in sorted(snap_keep - live_game_ids(games)):
        print(f"  keep {gid} status=snapshot")

    if not keep and not args.force:
        print("no live games; pass --force to delete all xiangqi checkpoints")
        return 3

    import redis

    client = redis.from_url(get_redis_url(), decode_responses=True)
    before = client.info("memory").get("used_memory_human")

    def refresh_keep() -> set[str]:
        extra: set[str] = set()
        try:
            extra = live_game_ids(fetch_api_games(base))
        except GamesFetchError:
            extra = set()
        return extra | resumable_snapshot_game_ids(snap_dir) | keep

    stats = purge_inactive_keys(
        client,
        keep,
        prefix=prefix,
        ttl_seconds=ttl,
        refresh_keep=refresh_keep,
    )
    after = client.info("memory").get("used_memory_human")
    print(
        f"deleted={stats['deleted']} retained={stats['retained']} "
        f"ttl_applied={stats['ttl_applied']} memory {before} -> {after}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
