import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.observe import (
    SCORE_FORBIDDEN,
    board_diagram,
    file_legend,
    legal_moves_annotated,
    own_pieces,
    threats,
)
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
HANGING_ROOK = "3k5/7r1/9/6N2/9/9/9/9/9/5K3 w"
EMPTY_CANNON = "4k4/9/9/9/9/9/9/4C4/9/4K4 w"
CHECK_FEN = "4k4/4R4/9/9/9/9/9/9/9/4K4 b"


def _assert_no_score_language(text: str) -> None:
    lowered = text.lower()
    for token in SCORE_FORBIDDEN:
        assert token.lower() not in lowered, f"score language leaked: {token!r} in {text}"


class TestBoardDiagram(unittest.TestCase):
    def test_start_diagram_is_single_char_not_scheme_c(self):
        d = board_diagram(Board(START_FEN))
        self.assertIn("K", d)
        self.assertIn("k", d)
        self.assertIn("楚河汉界", d.replace(" ", ""))
        self.assertNotIn("红帅", d)
        self.assertIn("a b c d e f g h i", d)
        self.assertNotIn("Side to move", d)

    def test_last_move_landing_marked(self):
        b = Board(START_FEN)
        b.make_move("h2e2")
        d = board_diagram(b)
        self.assertIn("*", d)
        self.assertRegex(d, r"e2.*\*|.*e2")
        self.assertNotIn("Side to move", d)


class TestFileLegend(unittest.TestCase):
    def test_red_files_nine_to_one(self):
        text = file_legend("w")
        self.assertIn("红九路", text)
        self.assertIn("红一路", text)
        self.assertIn("九宫", text)
        self.assertIn("河", text)

    def test_black_files_one_to_nine(self):
        text = file_legend("b")
        self.assertIn("黑一路", text)
        self.assertIn("黑九路", text)


class TestThreats(unittest.TestCase):
    def test_hanging_major(self):
        text = threats(Board(HANGING_ROOK), "w")
        self.assertIn("车", text)
        self.assertIn("无根", text)
        self.assertIn("可吃", text)
        self.assertNotIn("可踩", text)
        _assert_no_score_language(text)

    def test_start_threats_do_not_alarm_both_sides_horses(self):
        text = threats(Board(START_FEN), "w")
        self.assertNotIn("可踩", text)
        self.assertNotIn("己方红马", text)

    def test_rooted_capture_still_listed(self):
        # Black rook on h8 is defended by the black king-adjacent rook? Use
        # black horse defended by black rook, still legally capturable by red.
        board = Board("3k5/7r1/7n1/9/7P1/9/9/7C1/9/5K3 w")
        text = threats(board, "w")
        self.assertIn("马", text)
        self.assertIn("有根", text)
        self.assertIn("可吃", text)

    def test_empty_cannon_motif(self):
        text = threats(Board(EMPTY_CANNON), "w")
        self.assertIn("空头", text)

    def test_in_check(self):
        text = threats(Board(CHECK_FEN), "b")
        self.assertIn("将", text)

    def test_quiet_start_has_no_score_words(self):
        text = threats(Board(START_FEN), "w")
        _assert_no_score_language(text)


class TestLegalMovesAnnotated(unittest.TestCase):
    def test_grouped_with_chinese_and_iccs(self):
        text = legal_moves_annotated(Board(START_FEN))
        self.assertIn("h2e2", text)
        self.assertIn("炮二平五", text)
        self.assertIn("红炮", text)
        _assert_no_score_language(text)

    def test_check_flag_on_checking_move(self):
        text = legal_moves_annotated(Board("4k4/9/9/9/9/9/9/9/4R4/4K4 w"))
        self.assertIn("[将]", text)

    def test_second_cycle_move_is_flagged(self):
        board = Board("5k3/9/9/9/9/9/9/9/R8/4K4 w")
        for move in ("e0d0", "f9f8", "d0e0", "f8f9"):
            board.make_move(move)
        text = legal_moves_annotated(board)
        self.assertIn("e0d0", text)
        self.assertIn("[循环警告]", text)
        _assert_no_score_language(text)

    def test_filter_by_square_and_piece(self):
        board = Board(START_FEN)
        by_sq = legal_moves_annotated(board, squares="h2")
        self.assertIn("h2e2", by_sq)
        self.assertNotIn("红马 b0", by_sq)
        by_piece = legal_moves_annotated(board, pieces=["炮"])
        self.assertIn("红炮", by_piece)
        self.assertNotIn("红马", by_piece)
        empty = legal_moves_annotated(board, squares="e4")
        self.assertIn("以下为工作盘轮走方=红", empty)
        self.assertIn("无匹配合法着。", empty)


class TestOwnPieces(unittest.TestCase):
    def test_red_list_omits_black(self):
        text = own_pieces(Board(START_FEN), "w")
        self.assertIn("红炮 h2", text)
        self.assertNotIn("黑", text)


class TestThreatsCycle(unittest.TestCase):
    def test_threats_mentions_position_already_seen(self):
        board = Board("5k3/9/9/9/9/9/9/9/R8/4K4 w")
        for move in ("e0d0", "f9f8", "d0e0", "f8f9"):
            board.make_move(move)
        text = threats(board, "w")
        self.assertIn("已出现 2 次", text)
        _assert_no_score_language(text)


if __name__ == "__main__":
    unittest.main()
