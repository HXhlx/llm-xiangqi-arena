"""Tests for turn JSONL store and inject formatting."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import turn_store
from agent.summarize_turn import build_thought_text


class TestTurnStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.turns_dir = Path(self.tmp.name) / "turns"
        self.turns_dir.mkdir()
        self.patcher = mock.patch.object(turn_store, "TURNS_DIR", self.turns_dir)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_append_and_load_last(self):
        turn_store.append_turn("g1", {"ply": 1, "side": "red", "move": "h2e2", "reasoning": "r1"})
        turn_store.append_turn("g1", {"ply": 2, "side": "black", "move": "h9g7", "reasoning": "r2"})
        turn_store.append_turn("g1", {"ply": 3, "side": "red", "move": "b2e2", "reasoning": "r3"})
        last = turn_store.load_last_turn("g1")
        self.assertEqual(last["ply"], 3)
        last_red = turn_store.load_last_turn("g1", "red")
        self.assertEqual(last_red["ply"], 3)
        last_black = turn_store.load_last_turn("g1", "black")
        self.assertEqual(last_black["ply"], 2)

    def test_format_inject_includes_thought_and_tools(self):
        rec = {
            "ply": 1,
            "move": "h2e2",
            "move_zh": "炮二平五",
            "reasoning": "我想中炮",
            "thinking": "ok",
            "tool_rounds": [
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [{"name": "make_move", "content": "OK: Move h2e2"}],
                }
            ],
        }
        block = turn_store.format_inject_block(rec)
        self.assertIn("我想中炮", block)
        self.assertIn("make_move", block)
        self.assertIn("h2e2", block)

    def test_build_thought_no_tools(self):
        t = build_thought_text("hello reason", "hello think")
        self.assertIn("hello reason", t)
        self.assertIn("hello think", t)
        self.assertNotIn("make_move", t)

    def test_build_thought_keeps_full_text(self):
        reason = "R" * 20_000
        think = "T" * 20_000
        t = build_thought_text(reason, think)
        self.assertIn(reason, t)
        self.assertIn(think, t)
        self.assertNotIn("…", t)


if __name__ == "__main__":
    unittest.main()
