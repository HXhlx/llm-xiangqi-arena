"""Rebuild interrupted-game snapshots from turns JSONL."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.snapshot_from_turns import rebuild_snapshot_from_turns
from server import load_game_runtime_snapshot
from xiangqi import Board

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class RebuildSnapshotFromTurnsTests(unittest.TestCase):
    def test_replays_moves_and_omits_thought_blobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            turns = Path(tmp) / "turns.jsonl"
            rows = [
                {
                    "ply": 1,
                    "side": "red",
                    "move": "h2e2",
                    "move_zh": "炮二平五",
                    "fen_before": START_FEN,
                    "fen_after": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b",
                    "reasoning": "do not persist this blob",
                    "thinking": "also omit",
                },
                {
                    "ply": 2,
                    "side": "black",
                    "move": "h9g7",
                    "move_zh": "马8进7",
                    "fen_before": "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b",
                    "fen_after": "rnbakab1r/9/1c4nc1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR w",
                },
            ]
            turns.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            out = Path(tmp) / "snap.json"
            payload = rebuild_snapshot_from_turns(
                turns,
                game_id="abc123",
                red_preset="grok46",
                black_preset="flash_preview",
                dest=out,
            )
            self.assertEqual(payload["game_id"], "abc123")
            self.assertEqual(payload["status"], "interrupted")
            self.assertEqual(len(payload["move_history"]), 2)
            self.assertEqual(payload["move_history"][0]["move"], "h2e2")
            self.assertEqual(payload["red"]["preset"], "grok46")
            self.assertEqual(payload["black"]["preset"], "flash_preview")
            raw = out.read_text(encoding="utf-8")
            self.assertNotIn("do not persist this blob", raw)
            restored = load_game_runtime_snapshot(str(out))
            board = Board(START_FEN)
            board.make_move("h2e2")
            board.make_move("h9g7")
            self.assertEqual(restored.board.to_fen(), board.to_fen())
            self.assertEqual(restored.status, "interrupted")
            self.assertEqual(len(restored.move_history), 2)


if __name__ == "__main__":
    unittest.main()
