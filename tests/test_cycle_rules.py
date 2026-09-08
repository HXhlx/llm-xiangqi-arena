import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cycle_rules import classify_cycle, count_after, is_threefold_after, position_key
from llm_client import execute_tool
from xiangqi import Board


# Kings stay inside palaces and off the same file. Rook keeps K/A/B draw off.
KING_SHUTTLE_FEN = "5k3/9/9/9/9/9/9/9/R8/4K4 w"
SHUTTLE = ["e0d0", "f9f8", "d0e0", "f8f9"]


def _play(board: Board, moves: list[str]) -> None:
    for move in moves:
        board.make_move(move)


class PositionKeyTests(unittest.TestCase):
    def test_position_key_is_board_fen(self):
        board = Board(KING_SHUTTLE_FEN)
        self.assertEqual(position_key(board), board.to_fen())
        self.assertEqual(position_key(board), KING_SHUTTLE_FEN)

    def test_count_after_counts_existing_plus_one(self):
        keys = ["A w", "B b", "A w"]
        self.assertEqual(count_after(keys, "A w"), 3)
        self.assertEqual(count_after(keys, "C w"), 1)

    def test_threefold_after_at_three(self):
        keys = ["A w", "B b", "A w"]
        self.assertTrue(is_threefold_after(keys, "A w"))
        self.assertFalse(is_threefold_after(keys, "B b"))


class ThreefoldBanTests(unittest.TestCase):
    def test_new_board_records_starting_position_key(self):
        board = Board(KING_SHUTTLE_FEN)
        self.assertEqual(board.position_keys, [KING_SHUTTLE_FEN])

    def test_second_occurrence_is_still_legal(self):
        board = Board(KING_SHUTTLE_FEN)
        _play(board, SHUTTLE)
        self.assertEqual(board.to_fen(), KING_SHUTTLE_FEN)
        self.assertEqual(board.position_keys.count(KING_SHUTTLE_FEN), 2)
        self.assertIn("e0d0", board.get_legal_moves())
        self.assertTrue(board.is_valid_move("e0d0"))

    def test_third_occurrence_is_removed_from_legal_moves(self):
        board = Board(KING_SHUTTLE_FEN)
        _play(board, SHUTTLE + ["e0d0", "f9f8", "d0e0"])
        self.assertNotIn("f8f9", board.get_legal_moves())
        self.assertFalse(board.is_valid_move("f8f9"))
        with self.assertRaises(ValueError):
            board.make_move("f8f9")

    def test_copy_and_undo_keep_position_keys_in_sync(self):
        board = Board(KING_SHUTTLE_FEN)
        board.make_move("e0d0")
        cloned = board.copy()
        self.assertEqual(cloned.position_keys, board.position_keys)
        cloned.make_move("f9f8")
        self.assertNotEqual(cloned.position_keys, board.position_keys)
        cloned.undo_move()
        self.assertEqual(cloned.position_keys, board.position_keys)
        board.undo_move()
        self.assertEqual(board.position_keys, [KING_SHUTTLE_FEN])


class CycleExhaustedTests(unittest.TestCase):
    def test_no_raw_moves_without_check_is_still_stalemate(self):
        board = Board("3k5/4R4/9/9/9/9/9/9/9/4K4 b")
        self.assertEqual(board._legal_moves_ignoring_cycle(), [])
        self.assertFalse(board._is_in_check("b"))
        is_over, winner, reason = board.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, "red")
        self.assertEqual(reason, "stalemate - red wins")
        self.assertTrue(board.is_stalemate())

    def test_filtered_only_moves_draw_by_cycle_exhausted(self):
        board = Board(KING_SHUTTLE_FEN)
        extras = []
        for move in board._legal_moves_ignoring_cycle():
            extras.extend([board._trial_fen_after(move), board._trial_fen_after(move)])
        board.position_keys = [KING_SHUTTLE_FEN] + extras
        self.assertEqual(board.get_legal_moves(), [])
        self.assertFalse(board._is_in_check("w"))
        self.assertTrue(board._legal_moves_ignoring_cycle())
        is_over, winner, reason = board.is_game_over()
        self.assertTrue(is_over)
        self.assertEqual(winner, "draw")
        self.assertEqual(reason, "draw - cycle exhausted")
        self.assertFalse(board.is_stalemate())


class ClassifyCycleTests(unittest.TestCase):
    def test_no_repeat_is_undecided(self):
        board = Board(KING_SHUTTLE_FEN)
        board.make_move("e0d0")
        self.assertIsNone(classify_cycle(board))

    def test_example1_both_idle(self):
        board = Board("4k4/9/2c2an2/4c4/6R2/9/9/4B4/4A4/3K1AB2 w")
        _play(board, ["g5e5", "f7e8", "e5g5", "e8f7"])
        result = classify_cycle(board)
        self.assertIsNotNone(result)
        self.assertEqual(result.red, "idle")
        self.assertEqual(result.black, "idle")
        self.assertEqual(result.red_level, 0)
        self.assertEqual(result.black_level, 0)

    def test_example4_black_chases_rook(self):
        board = Board("4k3c/9/4bn2n/8c/6R2/6P2/9/9/9/3K5 w")
        _play(board, ["g5i5", "i6f6", "i5g5", "f6i6"])
        result = classify_cycle(board)
        self.assertIsNotNone(result)
        self.assertEqual(result.red, "idle")
        self.assertEqual(result.black, "chase")
        self.assertEqual(result.red_level, 0)
        self.assertEqual(result.black_level, 1)

    def test_example11_red_perpetual_check_blocks_chase(self):
        board = Board("4k4/9/9/9/4C4/9/4r4/4C4/9/4K1B2 w")
        _play(board, ["e5f5", "e3f3", "f5e5", "f3e3"])
        result = classify_cycle(board)
        self.assertIsNotNone(result)
        self.assertEqual(result.red, "check")
        self.assertEqual(result.black, "idle")
        self.assertEqual(result.red_level, 2)
        self.assertEqual(result.black_level, 0)

    def test_example12_red_checks_king(self):
        board = Board("3k5/2R6/9/9/9/9/9/9/6r2/4K1N2 w")
        _play(board, ["c8c9", "d9d8", "c9c8", "d8d9"])
        result = classify_cycle(board)
        self.assertIsNotNone(result)
        self.assertEqual(result.red, "check")
        self.assertEqual(result.black, "idle")
        self.assertEqual(result.red_level, 2)


class CycleRecordTests(unittest.TestCase):
    def test_make_move_reports_second_occurrence(self):
        board = Board(KING_SHUTTLE_FEN)
        _play(board, SHUTTLE)
        result = board.make_move("e0d0")
        self.assertEqual(result["cycle"]["count"], 2)
        self.assertIn(result["cycle"]["class"], ("idle", "check", "chase"))

    def test_tool_explains_threefold_ban(self):
        board = Board(KING_SHUTTLE_FEN)
        _play(board, SHUTTLE + ["e0d0", "f9f8", "d0e0"])
        text = execute_tool(board, "make_move", {"move": "f8f9"})
        self.assertIn("Illegal move", text)
        self.assertIn("third occurrence", text)
        self.assertNotIn("请从下列", text)
        self.assertNotIn("Legal live moves", text)


if __name__ == "__main__":
    unittest.main()
