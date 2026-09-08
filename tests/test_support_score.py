import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.support_score import score_phases, score_turn, score_turns


def _turn(**kwargs):
    base = {
        "ply": 1,
        "side": "red",
        "move": "h2e2",
        "move_zh": "炮二平五",
        "reasoning": "先开中炮",
        "thinking": "",
        "tool_rounds": [
            {
                "tool_calls": [{"name": "get_threats", "args": {}}],
                "tool_results": [
                    {
                        "name": "get_threats",
                        "content": "以下为工作盘轮走方=红\n无明显威胁事实。",
                    }
                ],
            },
            {
                "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                "tool_results": [
                    {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                ],
            },
        ],
    }
    base.update(kwargs)
    return base


class TestSupportScore(unittest.TestCase):
    def test_pass_when_query_then_legal_submit(self):
        report = score_turns([_turn()])
        self.assertTrue(report["s4_ok"])
        self.assertEqual(report["live_illegal_make_move"], 0)

    def test_fail_blind_guess_without_query(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                    ],
                }
            ]
        )
        row = score_turn(turn)
        self.assertFalse(row["used_query_tool"])
        self.assertFalse(row["ok"])

    def test_preview_move_alias_counts_as_query_tool(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "preview_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "preview_move", "content": "preview h2e2 炮二平五\n吃：无 | 将：否"}
                    ],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                    ],
                },
            ]
        )
        row = score_turn(turn)
        self.assertTrue(row["used_query_tool"])
        self.assertTrue(row["ok"])

    def test_reset_alone_does_not_count_as_query(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "preview_reset", "args": {}}],
                    "tool_results": [
                        {"name": "preview_reset", "content": "工作盘已回到 live。"}
                    ],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                    ],
                },
            ]
        )
        row = score_turn(turn)
        self.assertFalse(row["used_query_tool"])
        self.assertFalse(row["ok"])

    def test_count_live_illegal_make_move(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "get_board", "args": {}}],
                    "tool_results": [{"name": "get_board", "content": "以下为工作盘轮走方=红"}],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "a0a9"}}],
                    "tool_results": [
                        {
                            "name": "make_move",
                            "content": "Illegal move: 'a0a9' (live, not work). 不是当前真实盘合法着，不要再交同一着。",
                        }
                    ],
                },
            ]
        )
        row = score_turn(turn)
        self.assertEqual(row["live_illegal_make_move"], 1)
        self.assertTrue(row["unrecovered_live_illegal"])
        self.assertFalse(row["ok"])

    def test_recovered_live_illegal_is_support_success(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "get_legal_moves", "args": {}}],
                    "tool_results": [
                        {"name": "get_legal_moves", "content": "以下为工作盘轮走方=黑\n  马2进3 b9c7"}
                    ],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "c9e7"}}],
                    "tool_results": [
                        {
                            "name": "make_move",
                            "content": "Illegal move: 'c9e7' (live, not work). 不是当前真实盘合法着，不要再交同一着。",
                        }
                    ],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "b9c7"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move b9c7 is valid and will be played."}
                    ],
                },
            ]
        )
        row = score_turn(turn)
        self.assertEqual(row["live_illegal_make_move"], 1)
        self.assertFalse(row["unrecovered_live_illegal"])
        self.assertTrue(row["ok"])

    def test_flag_known_bad_black_chinese(self):
        row = score_turn(_turn(reasoning="准备马9进8"))
        self.assertEqual(row["bad_chinese"], ["马9进8"])
        self.assertFalse(row["ok"])

    def test_flag_threat_misread(self):
        turn = _turn(
            reasoning="红车有根所以安全，我平中",
            tool_rounds=[
                {
                    "tool_calls": [{"name": "get_threats", "args": {}}],
                    "tool_results": [
                        {
                            "name": "get_threats",
                            "content": "轮走方=黑 可吃：红车 e5（有根=被保护，仍可被吃）",
                        }
                    ],
                }
            ],
        )
        row = score_turn(turn)
        self.assertTrue(row["threat_misread"])
        self.assertFalse(row["ok"])

    def test_preview_illegal_then_reset_is_success(self):
        turn = _turn(
            tool_rounds=[
                {
                    "tool_calls": [{"name": "preview", "args": {"move": "h2e2"}}],
                    "tool_results": [{"name": "preview", "content": "preview h2e2 炮二平五"}],
                },
                {
                    "tool_calls": [{"name": "preview", "args": {"move": "b0c2"}}],
                    "tool_results": [
                        {
                            "name": "preview",
                            "content": "Illegal move: 'b0c2'. 工作盘已轮到对方。",
                        }
                    ],
                },
                {
                    "tool_calls": [{"name": "preview_reset", "args": {}}],
                    "tool_results": [{"name": "preview_reset", "content": "工作盘已回到 live。"}],
                },
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                    ],
                },
            ]
        )
        row = score_turn(turn)
        self.assertTrue(row["ok"])
        self.assertEqual(row["live_illegal_make_move"], 0)

    def test_phase_split_does_not_let_opening_hide_middle(self):
        opening = [_turn(ply=n) for n in range(1, 4)]
        middle_blind = _turn(
            ply=25,
            tool_rounds=[
                {
                    "tool_calls": [{"name": "make_move", "args": {"move": "h2e2"}}],
                    "tool_results": [
                        {"name": "make_move", "content": "OK: Move h2e2 is valid and will be played."}
                    ],
                }
            ],
        )
        report = score_phases(opening + [middle_blind])
        self.assertTrue(report["phases"]["opening"]["s4_ok"])
        self.assertFalse(report["phases"]["middlegame"]["s4_ok"])
        self.assertEqual(report["phases"]["middlegame"]["missing_query_plies"], [25])
        self.assertTrue(report["phases"]["late"]["empty"])
        self.assertFalse(report["phases_ok"])


if __name__ == "__main__":
    unittest.main()
