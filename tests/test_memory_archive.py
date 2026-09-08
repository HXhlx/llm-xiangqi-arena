"""Pre-compress memory archives land on disk; Redis replace is fail-closed."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, RemoveMessage

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.checkpoint_sync import merge_model_messages
from agent.memory_archive import (
    archive_dir,
    next_memory_seq,
    try_write_pre_compress_snapshot,
    write_pre_compress_snapshot,
)
from agent.redis_gc import older_checkpoint_keys, prune_thread_checkpoints


class MemoryArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_seq_files_with_pre_compress_payload(self):
        raw = [{"role": "user", "content": "full history"}]
        path = write_pre_compress_snapshot(
            "g1",
            "red",
            fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
            messages=raw,
            root=self.root,
        )
        self.assertEqual(path, archive_dir("g1", self.root) / "red-001.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["v"], 1)
        self.assertEqual(payload["kind"], "pre_compress")
        self.assertEqual(payload["game_id"], "g1")
        self.assertEqual(payload["side"], "red")
        self.assertEqual(payload["messages"], raw)
        self.assertGreater(payload["tokens"], 0)

    def test_second_compress_increments_seq(self):
        write_pre_compress_snapshot("g1", "red", fen="f1", messages=[{"role": "user", "content": "a"}], root=self.root)
        path = write_pre_compress_snapshot("g1", "red", fen="f2", messages=[{"role": "user", "content": "b"}], root=self.root)
        self.assertEqual(path.name, "red-002.json")
        self.assertTrue((archive_dir("g1", self.root) / "red-001.json").is_file())
        self.assertEqual(next_memory_seq(archive_dir("g1", self.root), "red"), 3)

    def test_does_not_touch_board_snapshot_dir(self):
        write_pre_compress_snapshot("g1", "black", fen="f", messages=[{"role": "user", "content": "x"}], root=self.root)
        self.assertFalse((self.root / "logs" / "snapshots").exists())
        self.assertFalse((self.root / "logs" / "turns").exists())

    def test_try_write_false_on_io_error(self):
        blocked = self.root / "blocked"
        blocked.write_text("not-a-dir", encoding="utf-8")
        ok = try_write_pre_compress_snapshot(
            "g1",
            "red",
            fen="f",
            messages=[{"role": "user", "content": "x"}],
            root=blocked,
        )
        self.assertFalse(ok)


class ArchiveFailClosedTests(unittest.TestCase):
    def test_failed_archive_does_not_replace(self):
        raw = [{"role": "user", "content": "long " * 80}]
        slim = [{"role": "system", "content": "[对局历史摘要]\n短"}]
        update = merge_model_messages(raw, slim, [AIMessage(content="ok")], replace=False)
        self.assertFalse(any(isinstance(m, RemoveMessage) for m in update))
        self.assertEqual(len(update), 1)

    def test_successful_archive_replaces(self):
        raw = [{"role": "user", "content": "long " * 80}]
        slim = [{"role": "system", "content": "[对局历史摘要]\n短"}]
        update = merge_model_messages(raw, slim, [AIMessage(content="ok")], replace=True)
        self.assertIsInstance(update[0], RemoveMessage)

    def test_no_compress_skips_archive(self):
        raw = [{"role": "user", "content": "hi"}]
        with patch("agent.memory_archive.write_pre_compress_snapshot") as writer:
            update = merge_model_messages(raw, raw, [AIMessage(content="ok")])
        writer.assert_not_called()
        self.assertFalse(any(isinstance(m, RemoveMessage) for m in update))


class PruneOlderCheckpointsTests(unittest.TestCase):
    def test_drops_older_checkpoint_family_keeps_latest_and_pointer(self):
        thread = "xiangqi:lg:g1:red"
        keys = [
            f"checkpoint:{thread}:__empty__:ckpt1",
            f"checkpoint:{thread}:__empty__:ckpt2",
            f"checkpoint_latest:{thread}:__empty__",
            f"write_keys_zset:{thread}:__empty__:ckpt1",
            f"write_keys_zset:{thread}:__empty__:ckpt2",
            f"checkpoint_write:{thread}:__empty__:ckpt1:task:0",
            f"checkpoint_write:{thread}:__empty__:ckpt2:task:0",
            "checkpoint:xiangqi:lg:g1:black:__empty__:other",
            "unrelated",
        ]
        drop = older_checkpoint_keys(keys, thread)
        self.assertEqual(
            set(drop),
            {
                f"checkpoint:{thread}:__empty__:ckpt1",
                f"write_keys_zset:{thread}:__empty__:ckpt1",
                f"checkpoint_write:{thread}:__empty__:ckpt1:task:0",
            },
        )

    def test_prune_unlinks_only_older_keys(self):
        thread = "xiangqi:lg:g1:red"
        store = {
            f"checkpoint:{thread}:__empty__:a": b"1",
            f"checkpoint:{thread}:__empty__:b": b"1",
            f"checkpoint_latest:{thread}:__empty__": b"b",
            "other": b"1",
        }

        class Client:
            def scan(self, cursor=0, match=None, count=10):
                del cursor, match, count
                return 0, list(store)

            def unlink(self, *ks):
                for key in ks:
                    store.pop(key, None)

        dropped = prune_thread_checkpoints(Client(), thread)
        self.assertEqual(dropped, 1)
        self.assertIn(f"checkpoint:{thread}:__empty__:b", store)
        self.assertIn(f"checkpoint_latest:{thread}:__empty__", store)
        self.assertIn("other", store)
        self.assertNotIn(f"checkpoint:{thread}:__empty__:a", store)


if __name__ == "__main__":
    unittest.main()
