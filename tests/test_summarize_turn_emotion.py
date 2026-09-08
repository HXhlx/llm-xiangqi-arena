"""Emotion-line gate for summary_zh (≤18 chars, no tool/move speak)."""

from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.summarize_turn import (
    EMOTION_MAX_CHARS,
    gate_emotion_line,
    is_emotion_line_ok,
)


class EmotionGateTests(unittest.TestCase):
    def test_accepts_short_emotion(self):
        self.assertEqual(gate_emotion_line("这波不亏"), "这波不亏")
        self.assertTrue(is_emotion_line_ok("疼"))

    def test_strips_whitespace_and_quotes(self):
        self.assertEqual(gate_emotion_line('  「裂开了」  '), "裂开了")

    def test_rejects_tool_speak(self):
        self.assertEqual(gate_emotion_line("我调用 make_move 了"), "")
        self.assertEqual(gate_emotion_line("先 preview 一下"), "")

    def test_rejects_iccs_coords(self):
        self.assertEqual(gate_emotion_line("走 h2e2 稳了"), "")

    def test_allows_board_nouns(self):
        self.assertEqual(gate_emotion_line("被迫吃炮解将"), "被迫吃炮解将")
        self.assertEqual(gate_emotion_line("丢子后急着扑车"), "丢子后急着扑车")
        self.assertTrue(is_emotion_line_ok("硬补一手仕"))

    def test_truncates_overlong(self):
        long = "这波真的不亏我还能翻盘哈哈哈太爽了吧"  # > 18
        out = gate_emotion_line(long)
        self.assertLessEqual(len(out), EMOTION_MAX_CHARS)
        self.assertTrue(out)

    def test_empty_input(self):
        self.assertEqual(gate_emotion_line(""), "")
        self.assertEqual(gate_emotion_line("   "), "")


if __name__ == "__main__":
    unittest.main()
