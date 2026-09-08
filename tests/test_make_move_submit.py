"""make_move is required; text is not a submit. Missing-tool nudges cap at 3."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestMissingMakeMoveNudge(unittest.TestCase):
    def _fn(self):
        from agent.player import handle_missing_make_move

        return handle_missing_make_move

    def test_first_miss_returns_tool_error_not_done(self):
        out = self._fn()(0)
        self.assertFalse(out["done"])
        self.assertIsNone(out.get("move"))
        self.assertEqual(out["make_move_misses"], 1)
        text = out["error_text"]
        self.assertIn("make_move", text)
        self.assertIn("工具节点错误", text)
        self.assertIn("1/3", text)

    def test_third_miss_still_nudges(self):
        out = self._fn()(2)
        self.assertFalse(out["done"])
        self.assertEqual(out["make_move_misses"], 3)
        self.assertIn("3/3", out["error_text"])

    def test_fourth_miss_fails(self):
        out = self._fn()(3)
        self.assertTrue(out["done"])
        self.assertTrue(out.get("error"))
        self.assertIsNone(out.get("move"))
        self.assertIn("make_move", out["error"])

    def test_cap_is_three(self):
        from agent.player import MAX_MAKE_MOVE_NUDGES

        self.assertEqual(MAX_MAKE_MOVE_NUDGES, 3)

    def test_observe_round_is_not_a_miss(self):
        from agent.player import should_count_missing_make_move

        self.assertFalse(should_count_missing_make_move(["get_board", "get_legal_moves"]))
        self.assertFalse(should_count_missing_make_move(["preview"]))
        self.assertFalse(should_count_missing_make_move(["preview_reset", "get_threats"]))
        self.assertFalse(should_count_missing_make_move(["make_move"]))

    def test_plain_text_round_is_a_miss(self):
        from agent.player import should_count_missing_make_move

        self.assertTrue(should_count_missing_make_move([]))
        self.assertTrue(should_count_missing_make_move(None))


class TestNoTextFallback(unittest.TestCase):
    def test_langgraph_player_has_no_text_extract(self):
        from agent.player import LangGraphPlayer

        self.assertFalse(hasattr(LangGraphPlayer, "_extract_move_from_text"))


class TestMoveGateError(unittest.TestCase):
    """走子阶段闸门纯函数：非 playing 一律禁用。"""

    def _fn(self):
        from agent.player import move_gate_error

        return move_gate_error

    def test_playing_or_empty_passes(self):
        for st in ("playing", None, "", "  "):
            self.assertIsNone(self._fn()(st), st)

    def test_paused_disabled_with_phase_hint(self):
        err = self._fn()("paused")
        self.assertIn("make_move 已禁用", err)
        self.assertIn("paused", err)
        self.assertIn("复盘/暂停", err)
        self.assertIn("不会被受理", err)

    def test_waiting_finished_interrupted_disabled(self):
        self.assertIn("预盘/开局前", self._fn()("waiting"))
        self.assertIn("终局复盘", self._fn()("finished"))
        self.assertIn("中断", self._fn()("interrupted"))

    def test_case_and_whitespace_insensitive(self):
        self.assertIn("已禁用", self._fn()("PAUSED"))
        self.assertIn("已禁用", self._fn()(" Paused "))
        self.assertIn("已禁用", self._fn()("Finished"))

    def test_unknown_status_still_disabled(self):
        err = self._fn()("weird")
        self.assertIn("已禁用", err)
        self.assertIn("weird", err)


class TestStatusProbeWiring(unittest.TestCase):
    def test_player_accepts_status_probe(self):
        from agent.player import LangGraphPlayer

        p = LangGraphPlayer(
            "http://example.invalid/v1", "k", "m", game_id="g1", side_name="red"
        )
        self.assertIsNone(p._status_probe)
        p2 = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="g1",
            side_name="red",
            status_probe=lambda: "paused",
        )
        self.assertEqual(p2._status_probe(), "paused")


class TestMoveGateGraph(unittest.IsolatedAsyncioTestCase):
    """graph 级闸门：status_probe 非 playing 时拦截 make_move 并 abort 回合。"""

    def _player(self, status: str):
        from agent.player import LangGraphPlayer

        return LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="gate-unittest",
            side_name="red",
            status_probe=lambda: status,
        )

    async def test_paused_make_move_gated_and_aborted(self):
        import json as _json
        from unittest.mock import AsyncMock, patch

        from langgraph.checkpoint.memory import MemorySaver
        from xiangqi import Board

        player = self._player("paused")

        async def fake_stream(_msgs):
            return (
                "",
                [
                    {
                        "id": "c1",
                        "name": "make_move",
                        "args": _json.dumps({"move": "h2e2"}),
                    }
                ],
                "tool_calls",
                "",
            )

        executed: list[tuple] = []

        def fake_execute_tool(session, name, args):
            executed.append((name, args))
            return "OK: Move h2e2 is valid and will be played."

        with patch(
            "agent.player.mem.get_checkpointer", AsyncMock(return_value=MemorySaver())
        ), patch(
            "agent.player.mem.prune_thread_to_latest", AsyncMock()
        ), patch.object(
            player, "_stream_completion", fake_stream
        ), patch(
            "agent.player.execute_tool", fake_execute_tool
        ), patch(
            "agent.player.load_last_turn", lambda *_a, **_k: None
        ):
            events = []
            async for ev in player.request_move(Board(), "w"):
                events.append(ev)

        # make_move 被闸门拦截：execute_tool 未收到 make_move
        self.assertFalse(any(n == "make_move" for n, _ in executed), executed)
        # 回合以 aborted 事件终止（非 move、非 error）
        self.assertEqual(events[-1]["type"], "aborted")
        self.assertEqual(events[-1]["reason"], "game_not_playing")
        # trace 记录了禁用错误
        trace = player.last_turn_trace or {}
        rounds = trace.get("tool_rounds") or []
        self.assertTrue(rounds)
        contents = [r.get("content") for rs in rounds for r in rs.get("tool_results", [])]
        self.assertTrue(any("make_move 已禁用" in (c or "") for c in contents), contents)

    async def test_playing_make_move_untouched(self):
        import json as _json
        from unittest.mock import AsyncMock, patch

        from langgraph.checkpoint.memory import MemorySaver
        from xiangqi import Board

        player = self._player("playing")

        async def fake_stream(_msgs):
            return (
                "",
                [
                    {
                        "id": "c1",
                        "name": "make_move",
                        "args": _json.dumps({"move": "h2e2"}),
                    }
                ],
                "tool_calls",
                "",
            )

        def fake_execute_tool(session, name, args):
            return "OK: Move h2e2 is valid and will be played."

        with patch(
            "agent.player.mem.get_checkpointer", AsyncMock(return_value=MemorySaver())
        ), patch(
            "agent.player.mem.prune_thread_to_latest", AsyncMock()
        ), patch.object(
            player, "_stream_completion", fake_stream
        ), patch(
            "agent.player.execute_tool", fake_execute_tool
        ), patch(
            "agent.player.load_last_turn", lambda *_a, **_k: None
        ):
            events = []
            async for ev in player.request_move(Board(), "w"):
                events.append(ev)

        self.assertEqual(events[-1], {"type": "move", "move": "h2e2"})


if __name__ == "__main__":
    unittest.main()
