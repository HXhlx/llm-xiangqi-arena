"""Redis checkpoint TTL + stale-key purge."""

from __future__ import annotations

import os
import sys
import unittest
from fnmatch import fnmatch
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.redis_gc import (
    GamesFetchError,
    game_id_from_redis_key,
    langgraph_ttl_config,
    live_game_ids,
    parse_games_payload,
    purge_inactive_keys,
    redis_ttl_seconds,
    require_checkpoint_prefix,
    resumable_snapshot_game_ids,
    should_retain_redis_key,
    thread_owned_keys,
    wipe_thread_keys,
)


class FakeRedis:
    def __init__(self, keys: list[str]):
        self.store = {key: b"1" for key in keys}
        self.expires: dict[str, int] = {}

    def scan(self, cursor=0, match=None, count=10):
        del cursor, count
        keys = list(self.store)
        if match:
            keys = [key for key in keys if fnmatch(key, match)]
        return 0, keys

    def unlink(self, *keys):
        removed = 0
        for key in keys:
            if self.store.pop(key, None) is not None:
                self.expires.pop(key, None)
                removed += 1
        return removed

    def expire(self, key, seconds):
        if key not in self.store:
            return False
        self.expires[key] = int(seconds)
        return True


class RedisTtlConfigTests(unittest.TestCase):
    def test_default_disables_ttl(self):
        self.assertEqual(redis_ttl_seconds({}), 0)
        self.assertIsNone(langgraph_ttl_config({}))

    def test_zero_disables_ttl(self):
        self.assertEqual(redis_ttl_seconds({"XIANGQI_REDIS_TTL": "0"}), 0)
        self.assertIsNone(langgraph_ttl_config({"XIANGQI_REDIS_TTL": "0"}))

    def test_langgraph_uses_minutes_and_refresh(self):
        cfg = langgraph_ttl_config({"XIANGQI_REDIS_TTL": "86400"})
        self.assertEqual(cfg["default_ttl"], 1440.0)
        self.assertTrue(cfg["refresh_on_read"])


class RedisKeyClassifyTests(unittest.TestCase):
    def test_extracts_game_id_from_checkpoint_shapes(self):
        prefix = "xiangqi:lg:"
        self.assertEqual(
            game_id_from_redis_key(
                "checkpoint:xiangqi:lg:abc123:red:__empty__:ckpt", prefix
            ),
            "abc123",
        )
        self.assertEqual(
            game_id_from_redis_key(
                "write_keys_zset:xiangqi:lg:abc123:black:__empty__:ckpt", prefix
            ),
            "abc123",
        )

    def test_keeps_foreign_keys_and_live_games_only(self):
        prefix = "xiangqi:lg:"
        keep = {"live1"}
        self.assertTrue(should_retain_redis_key("hc", keep, prefix))
        self.assertTrue(
            should_retain_redis_key("checkpoint:xiangqi:lg:live1:red:x", keep, prefix)
        )
        self.assertFalse(
            should_retain_redis_key("checkpoint:xiangqi:lg:dead1:red:x", keep, prefix)
        )
        self.assertTrue(should_retain_redis_key("xiangqi:lg:", keep, prefix))
        self.assertTrue(should_retain_redis_key("xiangqi:rl:rpm:site", keep, prefix))
        self.assertTrue(
            should_retain_redis_key("xiangqi:rl:rpm:model:grok46", keep, prefix)
        )

    def test_live_ids_skip_finished(self):
        ids = live_game_ids(
            [
                {"id": "a", "status": "playing"},
                {"id": "b", "status": "interrupted"},
                {"id": "c", "status": "finished"},
                {"id": "", "status": "playing"},
            ]
        )
        self.assertEqual(ids, {"a", "b"})

    def test_invalid_games_payload_raises(self):
        with self.assertRaises(GamesFetchError):
            parse_games_payload(["not", "an", "object"])
        with self.assertRaises(GamesFetchError):
            parse_games_payload({"status": "ok"})

    def test_empty_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            require_checkpoint_prefix("")
        redis = FakeRedis(["unrelated"])
        with self.assertRaises(ValueError):
            purge_inactive_keys(redis, set(), prefix="")


class RedisPurgeTests(unittest.TestCase):
    def test_purge_drops_inactive_and_ttls_live(self):
        prefix = "xiangqi:lg:"
        redis = FakeRedis(
            [
                "checkpoint:xiangqi:lg:live1:red:ckpt",
                "checkpoint:xiangqi:lg:dead1:red:ckpt",
                "other:app:key",
            ]
        )
        stats = purge_inactive_keys(
            redis,
            {"live1"},
            prefix=prefix,
            ttl_seconds=60,
        )
        self.assertEqual(stats["deleted"], 1)
        self.assertEqual(stats["retained"], 1)
        self.assertNotIn("checkpoint:xiangqi:lg:dead1:red:ckpt", redis.store)
        self.assertIn("checkpoint:xiangqi:lg:live1:red:ckpt", redis.store)
        self.assertIn("other:app:key", redis.store)
        self.assertEqual(redis.expires["checkpoint:xiangqi:lg:live1:red:ckpt"], 60)

    def test_refresh_keep_rescues_new_live_game(self):
        prefix = "xiangqi:lg:"
        redis = FakeRedis(
            [
                "checkpoint:xiangqi:lg:late1:red:ckpt",
            ]
        )
        stats = purge_inactive_keys(
            redis,
            set(),
            prefix=prefix,
            ttl_seconds=30,
            refresh_keep=lambda: {"late1"},
        )
        self.assertEqual(stats["deleted"], 0)
        self.assertIn("checkpoint:xiangqi:lg:late1:red:ckpt", redis.store)


class ResumableSnapshotIdsTests(unittest.TestCase):
    def test_keeps_interrupted_and_skips_finished(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "live.json").write_text(
                '{"game_id":"live1","status":"interrupted"}', encoding="utf-8"
            )
            (root / "done.json").write_text(
                '{"game_id":"done1","status":"finished"}', encoding="utf-8"
            )
            (root / "legacy.json").write_text('{"game_id":"legacy1"}', encoding="utf-8")
            (root / "bad.json").write_text("{not-json", encoding="utf-8")
            self.assertEqual(
                resumable_snapshot_game_ids(root),
                {"live1", "legacy1", "bad"},
            )


class WipeThreadKeysTests(unittest.TestCase):
    def test_owned_keys_require_kind_and_trailing_colon(self):
        thread = "xiangqi:lg:g1:red"
        self.assertEqual(
            set(
                thread_owned_keys(
                    [
                        f"checkpoint:{thread}:__empty__:ckpt",
                        f"checkpoint_latest:{thread}:__empty__",
                        f"checkpoint_write:{thread}:__empty__:ckpt:task:0",
                        f"write_keys_zset:{thread}:__empty__:ckpt",
                        f"checkpoint:{thread}2:__empty__:ckpt",
                        "checkpoint:xiangqi:lg:g1:black:__empty__:ckpt",
                        "unrelated",
                    ],
                    thread,
                )
            ),
            {
                f"checkpoint:{thread}:__empty__:ckpt",
                f"checkpoint_latest:{thread}:__empty__",
                f"checkpoint_write:{thread}:__empty__:ckpt:task:0",
                f"write_keys_zset:{thread}:__empty__:ckpt",
            },
        )

    def test_wipe_unlinks_all_owned_keys_for_thread(self):
        thread = "xiangqi:lg:g1:red"
        redis = FakeRedis(
            [
                f"checkpoint:{thread}:__empty__:ckpt1",
                f"checkpoint:{thread}:__empty__:ckpt2",
                f"checkpoint_latest:{thread}:__empty__",
                "checkpoint:xiangqi:lg:g1:black:__empty__:ckpt",
                "other",
            ]
        )
        dropped = wipe_thread_keys(redis, thread)
        self.assertEqual(dropped, 3)
        self.assertNotIn(f"checkpoint:{thread}:__empty__:ckpt1", redis.store)
        self.assertNotIn(f"checkpoint_latest:{thread}:__empty__", redis.store)
        self.assertIn("checkpoint:xiangqi:lg:g1:black:__empty__:ckpt", redis.store)
        self.assertIn("other", redis.store)


class CheckpointerTtlWireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        from agent import memory as mem

        mem._checkpointer = None
        mem._cm = None

    async def test_get_checkpointer_passes_ttl(self):
        from agent import memory as mem

        captured: dict = {}

        class DummyCM:
            async def __aenter__(self):
                return DummyCP()

            async def __aexit__(self, *args):
                return None

        class DummyCP:
            async def asetup(self):
                return None

        def fake_from_conn_string(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return DummyCM()

        mem._checkpointer = None
        mem._cm = None
        with patch.object(mem.AsyncRedisSaver, "from_conn_string", fake_from_conn_string):
            with patch.dict(
                os.environ,
                {"REDIS_URL": "redis://localhost/0", "XIANGQI_REDIS_TTL": "3600"},
                clear=False,
            ):
                await mem.get_checkpointer()
        self.assertEqual(captured["ttl"]["default_ttl"], 60.0)
        self.assertTrue(captured["ttl"]["refresh_on_read"])

    async def test_delete_thread_wipes_keys_left_by_index_cap(self):
        from agent import memory as mem

        thread = "xiangqi:lg:g1:red"
        store = {
            f"checkpoint:{thread}:__empty__:ckpt": b"1",
            f"checkpoint_write:{thread}:__empty__:ckpt:task:0": b"1",
            "other": b"1",
        }

        class DummyRedis:
            async def scan(self, cursor=0, match=None, count=10):
                del cursor, count
                keys = list(store)
                if match:
                    keys = [key for key in keys if fnmatch(key, match)]
                return 0, keys

            async def unlink(self, *keys):
                for key in keys:
                    store.pop(key, None)

        class DummyCP:
            _redis = DummyRedis()

            async def adelete_thread(self, thread_id):
                del thread_id

        mem._checkpointer = DummyCP()
        mem._cm = object()
        await mem.delete_thread(thread)
        self.assertNotIn(f"checkpoint:{thread}:__empty__:ckpt", store)
        self.assertNotIn(f"checkpoint_write:{thread}:__empty__:ckpt:task:0", store)
        self.assertIn("other", store)


if __name__ == "__main__":
    unittest.main()
