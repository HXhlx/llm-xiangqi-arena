import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from xiangqi import Board


KINGS_ADVISORS_BISHOPS_ONLY_FEN = "2bakab2/9/9/9/9/9/9/9/9/2BAKAB2 w"
ROOK_STILL_ON_BOARD_FEN = "2bakab2/9/9/9/4R4/9/9/9/9/2BAKAB2 b"
START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class XiangqiRuleTests(unittest.TestCase):
    def test_draw_when_only_kings_advisors_and_bishops_remain(self):
        board = Board(KINGS_ADVISORS_BISHOPS_ONLY_FEN)

        is_over, winner, reason = board.is_game_over()

        self.assertTrue(is_over)
        self.assertEqual(winner, "draw")
        self.assertEqual(reason, "draw - only kings, advisors, and bishops remain")

    def test_not_draw_when_other_material_still_exists(self):
        board = Board(ROOK_STILL_ON_BOARD_FEN)

        is_over, winner, reason = board.is_game_over()

        self.assertFalse(is_over)
        self.assertIsNone(winner)
        self.assertEqual(reason, "")

    def test_rook_defends_adjacent_own_horse(self):
        board = Board("3k5/7r1/7n1/9/7P1/9/9/7C1/9/5K3 w")
        self.assertTrue(board._is_attacked_by(7, 7, "b"))

    def test_cannon_protects_own_piece_beyond_screen(self):
        board = Board("3k5/9/1c7/9/1p7/9/1r7/9/9/4K4 w")
        self.assertEqual(board.get_piece(1, 7), "c")
        self.assertEqual(board.get_piece(1, 5), "p")
        self.assertEqual(board.get_piece(1, 3), "r")
        self.assertTrue(board._is_attacked_by(1, 3, "b"))

    def test_horse_protects_own_piece_with_open_leg(self):
        board = Board("3k5/9/1n7/9/2p6/9/9/9/9/4K4 w")
        self.assertEqual(board.get_piece(1, 7), "n")
        self.assertIsNone(board.get_piece(1, 6))
        self.assertEqual(board.get_piece(2, 5), "p")
        self.assertTrue(board._is_attacked_by(2, 5, "b"))

    # --- Chinese notation disambiguation tests ---

    def test_single_file_two_rooks_uses_prefix_only(self):
        """2 rooks on same file, no other rooks → prefix only (e.g. 前车进一)."""
        board = Board("4k4/9/9/9/9/9/9/4R4/9/4R4 w")
        front = board.to_chinese_move("e2e1")  # front rook retreats
        rear = board.to_chinese_move("e0e1")   # rear rook advances
        self.assertEqual(front, "前车退一")
        self.assertEqual(rear, "后车进一")
        # No file number in the string (prefix replaces it)
        self.assertNotIn("五", front)
        self.assertNotIn("五", rear)

    def test_multi_file_each_two_pawns_uses_prefix_and_file(self):
        """2 files each with 2 same-type pawns → prefix + file num (e.g. 前兵七进一)."""
        # Red pawns at c4,c6 (file 七) and g4,g6 (file 三)
        board = Board("4k4/9/9/2P3P2/9/2P3P2/9/9/9/4K4 w")
        front_c = board.to_chinese_move("c6c7")  # front pawn on file 七
        rear_c = board.to_chinese_move("c4c5")   # rear pawn on file 七
        front_g = board.to_chinese_move("g6g7")  # front pawn on file 三
        rear_g = board.to_chinese_move("g4g5")   # rear pawn on file 三
        # Must include file number to disambiguate across files
        self.assertEqual(front_c, "前兵七进一")
        self.assertEqual(rear_c, "后兵七进一")
        self.assertEqual(front_g, "前兵三进一")
        self.assertEqual(rear_g, "后兵三进一")
        # The four move_zh strings must all be distinct (no collision)
        all_zh = {front_c, rear_c, front_g, rear_g}
        self.assertEqual(len(all_zh), 4, "move_zh strings must not collide")

    def test_multi_file_rooks_uses_prefix_and_file(self):
        """2 files each with 2 rooks → prefix + file num (e.g. 前车九进一)."""
        # Red rooks at a0,a2 (file 九) and e0,e2 (file 五)
        board = Board("4k4/9/9/9/9/9/9/R3R4/9/R3R4 w")
        front_a = board.to_chinese_move("a2a3")
        rear_a = board.to_chinese_move("a0a1")
        front_e = board.to_chinese_move("e2e3")
        rear_e = board.to_chinese_move("e0e1")
        self.assertEqual(front_a, "前车九进一")
        self.assertEqual(rear_a, "后车九进一")
        self.assertEqual(front_e, "前车五进一")
        self.assertEqual(rear_e, "后车五进一")
        self.assertEqual(len({front_a, rear_a, front_e, rear_e}), 4)

    def test_one_file_two_other_file_one_uses_prefix_only(self):
        """2 rooks on file A, 1 rook on file B → prefix only for A's rooks
        (prefix is unambiguous because only one file has ≥2)."""
        # a0=R (file 九, single), e0=R and e2=R (file 五, two rooks)
        board = Board("4k4/9/9/9/9/9/9/4R4/9/R3R4 w")
        front_e = board.to_chinese_move("e2e1")
        rear_e = board.to_chinese_move("e0e1")
        self.assertEqual(front_e, "前车退一")
        self.assertEqual(rear_e, "后车进一")
        # Single rook on file 九 still uses plain file notation
        rook_a = board.to_chinese_move("a0a1")
        self.assertEqual(rook_a, "车九进一")

    def test_black_opening_moves_match_iccs(self):
        board = Board(START_FEN)
        board.make_move("h2e2")
        self.assertEqual(board.to_chinese_move("h9g7"), "马8进7")
        self.assertEqual(board.to_chinese_move("b9c7"), "马2进3")
        self.assertEqual(board.to_chinese_move("i9h9"), "车9平8")

    def test_black_center_pawn_and_king_advance(self):
        board = Board(START_FEN)
        for move in (
            "h2e2", "h9g7", "h0g2", "b9c7", "i0h0", "i9h9",
            "b0c2", "g6g5", "h0h6", "h7i7", "h6g6", "i7i8",
            "g6g5", "i8g8", "g5e5",
        ):
            board.make_move(move)
        self.assertEqual(board.to_chinese_move("e6e5"), "卒5进1")
        board.make_move("e6e5")
        board.make_move("e2e5")
        self.assertEqual(board.to_chinese_move("e9e8"), "将5进1")

    def test_red_black_same_idea_uses_matching_file_numbers(self):
        start = Board(START_FEN)
        self.assertEqual(start.to_chinese_move("h2e2"), "炮二平五")
        self.assertEqual(start.to_chinese_move("b0c2"), "马八进七")
        start.make_move("h2e2")
        self.assertEqual(start.to_chinese_move("h9g7"), "马8进7")
        self.assertEqual(start.to_chinese_move("b9c7"), "马2进3")


if __name__ == "__main__":
    unittest.main()
