# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server import parse_game_log


SAMPLE = """============================================================
Game: e4bf585b | 2026-08-20 00:49:44
Red: Spark-X2-Agent · LLM (spark-x2-agent)  vs  Black: SenseNova 6.8 Flash Lite · LLM (sensenova-6.8-flash-lite)
Initial FEN: rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w
Result: red wins - checkmate - red wins
Moves (3):
  #1 red: h2e2 (炮二平五)
    summary:
    我现在是这盘棋里的红方，正看着棋盘。
  #2 black: h7e7 (炮8平5)
    summary:
    红方刚才走了炮二平五。
  #3 red: e2e6 (炮五进四) xp
    summary:
    这手棋挺有意思的。
Turn thoughts JSONL: turns/e4bf585b.jsonl
Final FEN: rnbakabnr/9/1c5c1/p1p1C1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b
"""


class ParseGameLogTests(unittest.TestCase):
    def test_keeps_all_moves_when_summary_blocks_follow_each_ply(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "game.log"
            path.write_text(SAMPLE, encoding="utf-8")
            data = parse_game_log(str(path), include_moves=True)
        self.assertEqual(data["move_count"], 3)
        moves = data["moves"]
        self.assertEqual([m["move"] for m in moves], ["h2e2", "h7e7", "e2e6"])
        self.assertEqual(moves[2]["captured"], "p")


if __name__ == "__main__":
    unittest.main()

