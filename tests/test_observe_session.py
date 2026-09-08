import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.observe import SCORE_FORBIDDEN, own_pieces
from agent.observe_session import ObserveSession
from llm_client import TOOL_DEFINITIONS, build_system_prompt, execute_tool
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CAPTURE_PREVIEW = "4k4/9/9/4r4/9/9/9/9/4R4/4K4 w"
KING_SHUTTLE = "5k3/9/9/9/9/9/9/9/R8/4K4 w"


def _assert_no_score_language(text: str) -> None:
    lowered = text.lower()
    for token in SCORE_FORBIDDEN:
        assert token.lower() not in lowered, f"score language leaked: {token!r} in {text}"


def _tool_names() -> set[str]:
    return {t["function"]["name"] for t in TOOL_DEFINITIONS}


class TestObserveSessionCursor(unittest.TestCase):
    def test_header_at_live_root(self):
        session = ObserveSession(Board(START_FEN))
        header = session.header()
        self.assertIn("work +0", header)
        self.assertIn("path: -", header)
        self.assertIn("to_move=红", header)

    def test_preview_advances_work_not_live(self):
        live = Board(START_FEN)
        session = ObserveSession(live)
        text = session.preview("h2e2")
        self.assertIn("preview h2e2", text)
        self.assertIn("炮二平五", text)
        self.assertIn("吃：无", text)
        self.assertIn("work +1", text)
        self.assertIn("path:", text)
        self.assertIn("1. 红 h2e2 炮二平五", text)
        self.assertIn("to_move=黑", text)
        self.assertNotIn("Side to move", text)
        self.assertTrue(live.to_fen().startswith(START_FEN.split()[0]))
        self.assertNotEqual(session.work.to_fen(), live.to_fen())
        _assert_no_score_language(text)

    def test_preview_can_play_opponent_reply(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        text = session.preview("h9g7")
        self.assertIn("path:", text)
        self.assertIn("1. 红 h2e2", text)
        self.assertIn("2. 黑 h9g7", text)
        self.assertIn("work +2", text)
        self.assertIn("to_move=红", text)

    def test_illegal_preview_does_not_move_cursor(self):
        session = ObserveSession(Board(START_FEN))
        text = session.preview("a0a9")
        self.assertIn("Illegal move", text)
        self.assertIn("work +0", text)
        self.assertEqual(session.path, [])

    def test_reset_clears_path(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        session.preview("h9g7")
        reset = session.reset()
        self.assertEqual(session.path, [])
        self.assertIn("work +0", reset)
        self.assertIn("live", reset)

    def test_path_event_line_only_shows_preview_plies(self):
        live = Board(START_FEN)
        live.make_move("h2e2")
        live.make_move("h9g7")
        session = ObserveSession(live)
        text = session.preview("b0c2")
        self.assertIn("1. 红 b0c2", text)
        self.assertNotIn("h2e2", text.split("path:", 1)[-1].split("preview ", 1)[0])
        self.assertIn("work +1", text)

    def test_soft_warn_after_eight_plies(self):
        session = ObserveSession(Board(START_FEN))
        session.path = ["h2e2"] * 8
        header = session.header()
        self.assertIn("path 已 8 半步，仍须对 live 提交", header)
        self.assertIn("path:", header)


class TestLiveOnlyMakeMove(unittest.TestCase):
    def test_accepts_live_move_after_preview_line(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        session.preview("h9g7")
        text = session.submit_live("h2e2")
        self.assertTrue(text.splitlines()[-1].startswith("OK:"))
        self.assertIn("work +2", text)

    def _assert_no_legal_dump(self, text: str) -> None:
        self.assertNotIn("请从下列", text)
        self.assertNotIn("Legal live moves", text)
        self.assertNotIn("Legal moves are:", text)

    def test_rejects_opponent_reply_on_live(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        text = session.submit_live("h9g7")
        self.assertIn("Rejected", text)
        self.assertIn("working board", text)
        self.assertIn("真实盘", text)
        self.assertIn("live", text.lower())
        self._assert_no_legal_dump(text)

    def test_rejects_deeper_own_ply(self):
        session = ObserveSession(Board(CAPTURE_PREVIEW))
        session.preview("e1e4")
        # Black to move on work; a black king shuffle is not a live red move.
        reply = session.work.get_legal_moves()[0]
        text = session.submit_live(reply)
        self.assertIn("Rejected", text)
        self.assertNotIn("OK:", text)
        self._assert_no_legal_dump(text)

    def test_illegal_live_move_labels_live_not_work(self):
        text = execute_tool(Board(START_FEN), "make_move", {"move": "a0a9"})
        self.assertIn("Illegal move", text)
        self.assertIn("live, not work", text)
        self.assertIn("不要再交同一着", text)
        self.assertNotIn("b0c2", text)
        self.assertNotIn("开局名称", text)
        self._assert_no_legal_dump(text)

    def test_illegal_live_move_work_legal_is_not_opening_boilerplate(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        text = session.submit_live("h9g7")
        self.assertIn("Rejected", text)
        self.assertIn("working board", text)
        self.assertNotIn("开局名称", text)
        self._assert_no_legal_dump(text)

    def test_threefold_live_reject_names_cycle_not_opening(self):
        board = Board(KING_SHUTTLE)
        for move in ("e0d0", "f9f8", "d0e0", "f8f9", "e0d0", "f9f8", "d0e0"):
            board.make_move(move)
        text = execute_tool(board, "make_move", {"move": "f8f9"})
        self.assertIn("Illegal move", text)
        self.assertIn("third occurrence", text)
        self.assertNotIn("开局名称", text)
        self._assert_no_legal_dump(text)


class TestQueryToolsReadWork(unittest.TestCase):
    def test_legal_moves_after_preview_are_opponents(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        text = execute_tool(session, "get_legal_moves", {})
        self.assertIn("h9g7", text)
        self.assertIn("黑", text)
        self.assertIn("work +1", text)
        self.assertNotIn("红炮 h2", text)

    def test_legal_moves_filter_square_and_piece(self):
        session = ObserveSession(Board(START_FEN))
        by_sq = execute_tool(session, "get_legal_moves", {"squares": "h2"})
        self.assertIn("h2e2", by_sq)
        self.assertNotIn("红马 b0", by_sq)
        by_piece = execute_tool(session, "get_legal_moves", {"pieces": "炮"})
        self.assertIn("红炮", by_piece)
        self.assertNotIn("红马", by_piece)

    def test_board_and_threats_follow_work(self):
        session = ObserveSession(Board(CAPTURE_PREVIEW))
        session.preview("e1e4")
        board_text = execute_tool(session, "get_board", {})
        self.assertIn("FEN：", board_text)
        self.assertIn("R", board_text)
        self.assertIn("work +1", board_text)
        threats_text = execute_tool(session, "get_threats", {})
        self.assertTrue("车" in threats_text or "将" in threats_text)
        _assert_no_score_language(board_text)
        _assert_no_score_language(threats_text)
        self.assertNotIn("Side to move", board_text)

    def test_deep_search_via_reset_and_replay(self):
        session = ObserveSession(Board(START_FEN))
        session.preview("h2e2")
        session.preview("h9g7")
        self.assertEqual(session.path, ["h2e2", "h9g7"])
        session.reset()
        session.preview("h2e2")
        session.preview("b9c7")
        self.assertEqual(session.path, ["h2e2", "b9c7"])
        self.assertIn("to_move=红", session.header())


class TestThinPromptAndToolCatalog(unittest.TestCase):
    def test_system_prompt_is_thin_snapshot(self):
        prompt = build_system_prompt(Board(START_FEN), "w", "zh")
        self.assertIn(START_FEN.split()[0], prompt)
        self.assertIn("红炮 h2", prompt)
        self.assertIn("开局工作盘=live", prompt)
        self.assertIn("串在一条 path", prompt)
        self.assertIn("to_move", prompt)
        self.assertIn("get_legal_moves", prompt)
        self.assertIn("preview", prompt)
        self.assertIn("make_move", prompt)
        self.assertNotIn("preview_move", prompt)
        self.assertNotRegex(prompt, r"炮二平五\s+h2e2")
        self.assertNotIn("马八进七", prompt)
        self.assertNotIn("按子分组", prompt)
        en_prompt = build_system_prompt(Board(START_FEN), "w", "en")
        self.assertIn(START_FEN.split()[0], en_prompt)
        self.assertIn("get_legal_moves", en_prompt)

    def test_own_pieces_are_side_only(self):
        text = own_pieces(Board(START_FEN), "w")
        self.assertIn("红炮 h2", text)
        self.assertNotIn("黑", text)

    def test_exposed_tools(self):
        names = _tool_names()
        self.assertEqual(
            names,
            {
                "make_move",
                "preview",
                "preview_reset",
                "get_legal_moves",
                "get_threats",
                "get_board",
            },
        )
        self.assertNotIn("preview_move", names)
        self.assertNotIn("preview_undo", names)
        self.assertNotIn("get_recent_moves", names)

    def test_preview_move_alias_acts_as_preview(self):
        session = ObserveSession(Board(START_FEN))
        text = execute_tool(session, "preview_move", {"move": "h2e2"})
        self.assertIn("preview h2e2", text)
        self.assertIn("work +1", text)
        self.assertEqual(session.path, ["h2e2"])
        self.assertNotIn("Unknown tool", text)

    def test_removed_tools_return_soft_hints(self):
        session = ObserveSession(Board(START_FEN))
        undo = execute_tool(session, "preview_undo", {})
        self.assertIn("已移除", undo)
        self.assertIn("preview_reset", undo)
        self.assertNotIn("Unknown tool", undo)
        recent = execute_tool(session, "get_recent_moves", {"n": 4})
        self.assertIn("已移除", recent)
        self.assertIn("path: 事件线", recent)
        self.assertNotIn("Unknown tool", recent)

    def test_cycle_preview_warns_without_diagram(self):
        board = Board(KING_SHUTTLE)
        for move in ("e0d0", "f9f8", "d0e0", "f8f9"):
            board.make_move(move)
        session = ObserveSession(board)
        text = session.preview("e0d0")
        self.assertIn("循环", text)
        self.assertIn("第2次", text)
        self.assertNotIn("Side to move", text)
        _assert_no_score_language(text)


if __name__ == "__main__":
    unittest.main()
