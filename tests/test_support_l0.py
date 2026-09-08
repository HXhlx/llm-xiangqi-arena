"""L0 support-layer contracts: facts, labels, no engine language."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.observe import SCORE_FORBIDDEN, legal_moves_annotated, threats
from agent.observe_session import ObserveSession
from llm_client import TOOL_DEFINITIONS, build_system_prompt, execute_tool
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
HANGING_BEFORE_G5E5 = [
    "h2e2", "h9g7", "h0g2", "b9c7", "i0h0", "i9h9",
    "b0c2", "g6g5", "h0h6", "h7i7", "h6g6", "i7i8",
    "g6g5", "i8g8",
]


def _assert_no_score_language(text: str) -> None:
    lowered = text.lower()
    for token in SCORE_FORBIDDEN:
        assert token.lower() not in lowered, f"score language leaked: {token!r} in {text}"


class TestBlackChineseL0(unittest.TestCase):
    def test_opening_black_names(self):
        board = Board(START_FEN)
        board.make_move("h2e2")
        self.assertEqual(board.to_chinese_move("h9g7"), "马8进7")
        self.assertEqual(board.to_chinese_move("b9c7"), "马2进3")
        self.assertEqual(board.to_chinese_move("e6e5"), "卒5进1")
        self.assertEqual(board.to_chinese_move("e9e8"), "将5进1")
        self.assertEqual(board.to_chinese_move("i9h9"), "车9平8")


class TestThreatLabelsL0(unittest.TestCase):
    def test_preview_center_rook_threats_use_side_to_move(self):
        live = Board(START_FEN)
        for move in HANGING_BEFORE_G5E5:
            live.make_move(move)
        session = ObserveSession(live)
        session.preview("g5e5")
        text = execute_tool(session, "get_threats", {})
        self.assertIn("以下为工作盘轮走方=黑", text)
        self.assertIn("轮走方=黑 可吃：", text)
        self.assertIn("红车 e5", text)
        self.assertIn("仍可被吃", text)
        self.assertNotIn("对方", text.split("\n", 1)[-1])
        self.assertNotRegex(text, r"对方.+有根，当前可吃")
        _assert_no_score_language(text)

    def test_query_tools_repeat_side_to_move(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        for name, args in (
            ("get_legal_moves", {}),
            ("get_board", {}),
            ("get_threats", {}),
        ):
            text = execute_tool(session, name, args)
            self.assertIn("以下为工作盘轮走方=黑", text, name)
            _assert_no_score_language(text)


class TestChainedPreviewHintL0(unittest.TestCase):
    def test_second_own_preview_is_illegal_without_naming_reset(self):
        session = ObserveSession(Board(START_FEN))
        first = session.preview("h2e2")
        self.assertIn("to_move=黑", first)
        second = session.preview("b0c2")
        self.assertIn("Illegal move", second)
        self.assertIn("工作盘已轮到对方", second)
        self.assertNotIn("preview_reset", second)
        self.assertEqual(session.path, ["h2e2"])


class TestThinSnapshotAndCatalogL0(unittest.TestCase):
    def test_thin_snapshot_has_no_full_legal_table(self):
        prompt = build_system_prompt(Board(START_FEN), "w", "zh")
        self.assertNotRegex(prompt, r"炮二平五\s+h2e2")
        self.assertNotIn("马八进七", prompt)
        self.assertIn("开局工作盘=live", prompt)
        self.assertIn("串在一条 path", prompt)
        self.assertNotIn("至少调用一种查询工具", prompt)
        self.assertIn("不要把「炮二平五」等开局名称固定成 h2e2", prompt)

    def test_legal_list_is_not_ranked(self):
        text = legal_moves_annotated(Board(START_FEN))
        self.assertIn("炮二平五", text)
        self.assertNotIn("最佳", text)
        self.assertNotIn("推荐", text)
        _assert_no_score_language(text)

    def test_tool_bodies_have_no_score_words(self):
        session = ObserveSession(Board(START_FEN))
        for name in (
            "get_board",
            "get_legal_moves",
            "get_threats",
            "preview",
        ):
            args = {"move": "h2e2"} if name == "preview" else {}
            _assert_no_score_language(execute_tool(session, name, args))
            session.reset()

    def test_catalog_still_excludes_engine_tools(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        self.assertNotIn("bestmove", names)
        self.assertNotIn("analyze", names)


if __name__ == "__main__":
    unittest.main()
