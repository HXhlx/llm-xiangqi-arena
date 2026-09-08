"""Tests for LLM board diagram + Chinese labels + observe tools."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import (
    DEFAULT_MAX_TOOL_ROUNDS,
    TOOL_DEFINITIONS,
    _get_piece_positions,
    _render_board_diagram,
    build_system_prompt,
    execute_tool,
)
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class TestBoardDiagram(unittest.TestCase):
    def test_start_diagram_has_kings_and_river(self):
        b = Board(START_FEN)
        d = _render_board_diagram(b)
        self.assertIn("红帅", d)
        self.assertIn("黑将", d)
        self.assertIn("楚 河 汉 界", d)
        self.assertIn("Side to move: Red", d)
        # column headers
        self.assertIn("a", d)
        self.assertIn("i", d)

    def test_diagram_updates_after_move(self):
        b = Board(START_FEN)
        before = _render_board_diagram(b)
        b.make_move("h2e2")
        after = _render_board_diagram(b)
        self.assertNotEqual(before, after)
        self.assertIn("Side to move: Black", after)
        # red cannon should no longer sit only on h-file back; e2 area has 红炮
        self.assertIn("红炮", after)

    def test_piece_positions_chinese(self):
        b = Board(START_FEN)
        pos = _get_piece_positions(b)
        self.assertIn("a0: 红车", pos)
        self.assertIn("e9: 黑将", pos)
        self.assertNotIn("a0: R", pos)

    def test_system_prompt_is_thin_snapshot(self):
        b = Board(START_FEN)
        prompt = build_system_prompt(b, "w", "zh")
        self.assertIn(START_FEN.split()[0], prompt)  # letter FEN preserved
        self.assertNotIn("get_recent_moves", prompt)
        self.assertNotIn("preview_undo", prompt)
        self.assertIn("make_move", prompt)
        self.assertIn("preview", prompt)
        self.assertIn("preview_reset", prompt)
        self.assertIn("get_legal_moves", prompt)
        self.assertIn("a（最左）", prompt)
        self.assertIn("红九路", prompt)
        self.assertIn("红炮 h2", prompt)
        self.assertIn("尚无", prompt)
        self.assertIn("开局工作盘=live", prompt)
        self.assertIn("串在一条 path", prompt)
        self.assertNotIn("preview_move", prompt)
        self.assertNotRegex(prompt, r"炮二平五\s+h2e2")
        self.assertNotIn("马八进七", prompt)

    def test_engine_fen_unchanged(self):
        b = Board(START_FEN)
        self.assertEqual(b.to_fen().split()[0], START_FEN.split()[0])
        _render_board_diagram(b)
        self.assertEqual(b.to_fen().split()[0], START_FEN.split()[0])


class TestObserveTools(unittest.TestCase):
    def test_make_move_still_works(self):
        b = Board(START_FEN)
        ok = execute_tool(b, "make_move", {"move": "h2e2"})
        self.assertIn("OK:", ok)
        bad = execute_tool(b, "make_move", {"move": "a0a9"})
        self.assertTrue("Illegal" in bad or "Invalid" in bad)

    def test_preview_tool(self):
        b = Board(START_FEN)
        out = execute_tool(b, "preview", {"move": "h2e2"})
        self.assertIn("preview h2e2", out)
        self.assertIn("炮二平五", out)
        self.assertIn("work +1", out)
        self.assertTrue(b.to_fen().startswith(START_FEN.split()[0]))
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("preview", names)
        self.assertNotIn("preview_move", names)
        self.assertNotIn("preview_undo", names)
        self.assertNotIn("get_recent_moves", names)

    def test_get_board_has_no_english_side_line(self):
        out = execute_tool(Board(START_FEN), "get_board", {})
        self.assertIn("以下为工作盘轮走方=红", out)
        self.assertNotIn("Side to move", out)

    def test_tool_round_ceiling_is_not_a_product_cap(self):
        self.assertGreaterEqual(DEFAULT_MAX_TOOL_ROUNDS, 10000)


if __name__ == "__main__":
    unittest.main()
