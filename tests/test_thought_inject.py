"""Tests that prepare_turn inject path loads prior turn text."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import turn_store
from agent.turn_store import format_inject_block, load_last_turn


class TestThoughtInject(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.turns_dir = Path(self.tmp.name) / "turns"
        self.turns_dir.mkdir()
        p = mock.patch.object(turn_store, "TURNS_DIR", self.turns_dir)
        p.start()
        self.addCleanup(p.stop)

    def test_prepare_would_see_prior_reasoning(self):
        turn_store.append_turn(
            "gameX",
            {
                "ply": 5,
                "side": "red",
                "move": "b0c2",
                "move_zh": "马八进七",
                "reasoning": "FULL_REASONING_TEXT_ABC",
                "thinking": "",
                "tool_rounds": [],
            },
        )
        prev = load_last_turn("gameX", "red")
        self.assertIsNotNone(prev)
        block = format_inject_block(prev)
        self.assertIn("FULL_REASONING_TEXT_ABC", block)
        self.assertIn("马八进七", block)
        # audience summary optional — inject should not require it
        self.assertNotIn("观众摘要", block)

    def test_inject_keeps_full_thought_without_char_cap(self):
        long_reason = "甲" * 30_000
        long_think = "乙" * 30_000
        rec = {
            "ply": 1,
            "move": "h2e2",
            "move_zh": "炮二平五",
            "reasoning": long_reason,
            "thinking": long_think,
            "tool_rounds": [
                {
                    "tool_calls": [{"name": "preview_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "preview_move", "content": "preview " + ("丙" * 2000)}
                    ],
                }
            ],
        }
        block = format_inject_block(rec)
        self.assertIn(long_reason, block)
        self.assertIn(long_think, block)
        self.assertNotIn("截断", block)
        self.assertIn("丙" * 2000, block)


if __name__ == "__main__":
    unittest.main()
