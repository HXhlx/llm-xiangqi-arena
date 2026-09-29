"""Per-ply counter reset and plain-llm key checks before game_loop."""

from __future__ import annotations

import json
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
ENV = "XIANGQI_MAX_ILLEGAL_MOVES"


def _stream_for(script, calls):
    async def fake_stream(_msgs):
        calls["n"] += 1
        item = script[min(calls["n"] - 1, len(script) - 1)]
        if isinstance(item, tuple):
            name, args = item
            return (
                "",
                [{"id": f"c{calls['n']}", "name": name, "args": args}],
                "tool_calls",
                "",
            )
        return (
            "",
            [
                {
                    "id": f"c{calls['n']}",
                    "name": "make_move",
                    "args": json.dumps({"move": item}),
                }
            ],
            "tool_calls",
            "",
        )

    return fake_stream


async def _ply(player, saver, script, board, side="w"):
    from xiangqi import Board

    if not isinstance(board, Board):
        board = Board(board)
    calls = {"n": 0}
    with patch(
        "agent.player.mem.get_checkpointer", AsyncMock(return_value=saver)
    ), patch(
        "agent.player.mem.prune_thread_to_latest", AsyncMock()
    ), patch(
        "agent.player.load_last_turn", lambda *_a, **_k: None
    ), patch.object(player, "_stream_completion", _stream_for(script, calls)):
        events = []
        async for ev in player.request_move(board, side):
            events.append(ev)
    return calls["n"], events


class CounterResetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev = os.environ.get(ENV)
        self._lang = os.environ.get("XIANGQI_LANG")
        os.environ[ENV] = "3"
        os.environ.pop("XIANGQI_LANG", None)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._prev
        if self._lang is None:
            os.environ.pop("XIANGQI_LANG", None)
        else:
            os.environ["XIANGQI_LANG"] = self._lang

    def _player(self, **kwargs):
        from agent.player import LangGraphPlayer

        params = dict(
            api_base="http://example.invalid/v1",
            api_key="sk-test",
            model="example-model",
            game_id="reset-ply",
            side_name="red",
            # Above the three model calls in the illegal-move script, so a
            # leaked illegal counter is what interrupts ply 2, not the round cap.
            max_tool_rounds=8,
        )
        params.update(kwargs)
        return LangGraphPlayer(**params)

    async def test_illegal_counter_resets_on_the_next_ply(self):
        """Two rejections then a legal move, twice, on one checkpoint thread.

        The cap is 3, so a counter that survives the first ply interrupts the
        second ply on its first rejection.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from xiangqi import Board

        player = self._player()
        saver = MemorySaver()
        board = Board(START)
        calls, events = await _ply(
            player, saver, ["a0a9", "a0a9", "h2e2"], board
        )
        self.assertEqual(calls, 3)
        self.assertEqual(events[-1]["type"], "move")
        self.assertEqual(events[-1]["move"], "h2e2")

        board.make_move("h2e2")
        board.make_move("a6a5")
        calls, events = await _ply(
            player, saver, ["a0a9", "a0a9", "b0c2"], board
        )
        self.assertEqual(calls, 3, events[-1])
        self.assertEqual(events[-1]["type"], "move", events[-1])
        self.assertEqual(events[-1]["move"], "b0c2")
        self.assertNotIn("rejected make_move", events[-1].get("message", ""))

    async def test_tool_round_counter_resets_on_the_next_ply(self):
        """Two non-move tool rounds then a legal move, twice.

        max_tool_rounds is 3, so a counter left at 3 interrupts the next ply
        before the model is called again.
        """
        from langgraph.checkpoint.memory import MemorySaver
        from xiangqi import Board

        player = self._player(max_tool_rounds=3)
        self.assertEqual(player.max_tool_rounds, 3)
        saver = MemorySaver()
        board = Board(START)
        ply1 = [("get_board", "{}"), ("get_board", "{}"), "h2e2"]
        calls, events = await _ply(player, saver, ply1, board)
        self.assertEqual(calls, 3)
        self.assertEqual(events[-1]["type"], "move")
        self.assertEqual(events[-1]["move"], "h2e2")

        board.make_move("h2e2")
        board.make_move("a6a5")
        ply2 = [("get_board", "{}"), ("get_board", "{}"), "b0c2"]
        calls, events = await _ply(player, saver, ply2, board)
        self.assertEqual(calls, 3, events[-1])
        self.assertEqual(events[-1]["type"], "move", events[-1])
        self.assertEqual(events[-1]["move"], "b0c2")
        self.assertNotIn("rounds", events[-1].get("message", ""))


class LlmSeatMissingKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_seat_without_a_key_is_400_before_the_loop(self):
        import server

        fen = START
        game = server.GameSession(
            "llm-nokey",
            fen,
            server.PlayerConfig(
                type="llm",
                model="example-model",
                api_base="https://llm.example/v1",
                api_key=None,
            ),
            server.PlayerConfig(type="random"),
        )
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        with self.assertRaises(server.HTTPException) as ctx:
            server._ensure_seat_credentials(game.red_config)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("api_key is missing", str(ctx.exception.detail))
        self.assertIn("without an API key", str(ctx.exception.detail))

        for blank in ("", "   ", "sk-xxxxxxxx"):
            seat = game.red_config.model_copy(update={"api_key": blank})
            with self.assertRaises(server.HTTPException) as ctx:
                server._ensure_seat_credentials(seat)
            self.assertEqual(ctx.exception.status_code, 400, blank)

        started = {"n": 0}

        async def _loop(_game):
            started["n"] += 1

        with patch.object(server, "game_loop", _loop):
            with self.assertRaises(server.HTTPException) as ctx:
                await server.start_game(game.id)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("api_key is missing", str(ctx.exception.detail))
        self.assertEqual(started["n"], 0)
        self.assertEqual(game.status, "waiting")
        self.assertIsNone(game.task)

        game.status = "interrupted"
        with patch.object(server, "game_loop", _loop):
            with self.assertRaises(server.HTTPException) as resumed:
                await server.resume_game(game.id)
        self.assertEqual(resumed.exception.status_code, 400)
        self.assertEqual(game.status, "interrupted")
        self.assertEqual(started["n"], 0)

        with self.assertRaises(server.HTTPException) as restored:
            await server.restore_game(game.id)
        self.assertEqual(restored.exception.status_code, 400)
        self.assertEqual(game.status, "interrupted")

        kept = server._ensure_seat_credentials(
            server.PlayerConfig(
                type="llm",
                model="example-model",
                api_base="https://llm.example/v1",
                api_key="sk-real",
            )
        )
        self.assertEqual(kept.api_key, "sk-real")


if __name__ == "__main__":
    unittest.main()
