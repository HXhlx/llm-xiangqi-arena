"""L1 fixture: replay 82b41b63 hanging-rook ply without an LLM."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.observe import legal_moves_annotated
from agent.observe_session import ObserveSession
from llm_client import execute_tool
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

# Flash duel 82b41b63 through ply 14 (black 炮9平7), red to move before 车三平五.
FIXTURE_82B41B63_PLIES = [
    "h2e2", "h9g7", "h0g2", "b9c7", "i0h0", "i9h9",
    "b0c2", "g6g5", "h0h6", "h7i7", "h6g6", "i7i8",
    "g6g5", "i8g8",
]


def _board_after_fixture() -> Board:
    board = Board(START_FEN)
    for move in FIXTURE_82B41B63_PLIES:
        board.make_move(move)
    return board


class TestFixture82b41b63(unittest.TestCase):
    def test_preview_center_rook_is_black_can_capture_not_red_safe(self):
        live = _board_after_fixture()
        self.assertEqual(live.turn, "w")
        session = ObserveSession(live)
        preview = session.preview("g5e5")
        self.assertIn("preview g5e5", preview)
        self.assertIn("to_move=黑", preview)
        text = execute_tool(session, "get_threats", {})
        self.assertIn("轮走方=黑", text)
        self.assertIn("红车 e5", text)
        self.assertIn("仍可被吃", text)
        self.assertIn("可吃", text)
        self.assertNotIn("红方安全", text)
        self.assertNotRegex(text, r"对方红车.+有根，当前可吃")

    def test_black_opening_legal_moves_match_iccs(self):
        board = Board(START_FEN)
        board.make_move("h2e2")
        text = legal_moves_annotated(board)
        self.assertIn("以下为工作盘轮走方=黑", text)
        self.assertIn("马8进7 h9g7", text)
        self.assertIn("马2进3 b9c7", text)
        self.assertEqual(board.to_chinese_move("i9h9"), "车9平8")
        self.assertNotIn("马9进8", text)
        self.assertNotIn("将6进2", text)
        self.assertNotIn("卒6进2", text)


if __name__ == "__main__":
    unittest.main()
