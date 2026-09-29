"""Rejected make_move calls stop a ply before the proxy is called unboundedly."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.player import (
    DEFAULT_MAX_ILLEGAL_MOVES,
    handle_illegal_make_move,
    max_illegal_moves,
)
from llm_client import DEFAULT_MAX_TOOL_ROUNDS, resolve_max_tool_rounds

ENV = "XIANGQI_MAX_ILLEGAL_MOVES"


class IllegalMoveCapTests(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get(ENV)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._prev

    def test_default_is_three(self):
        os.environ.pop(ENV, None)
        self.assertEqual(DEFAULT_MAX_ILLEGAL_MOVES, 3)
        self.assertEqual(max_illegal_moves(), 3)
        self.assertGreaterEqual(DEFAULT_MAX_TOOL_ROUNDS, 8)
        self.assertLessEqual(DEFAULT_MAX_TOOL_ROUNDS, 64)

    def test_third_rejection_interrupts_without_a_winner_flag(self):
        os.environ.pop(ENV, None)
        first = handle_illegal_make_move(0)
        self.assertFalse(first["done"])
        second = handle_illegal_make_move(first["illegal_make_moves"])
        self.assertFalse(second["done"])
        third = handle_illegal_make_move(second["illegal_make_moves"])
        self.assertTrue(third["done"])
        self.assertIn("rejected make_move", third["error"])
        self.assertIn("limit 3", third["error"])
        self.assertIsNone(third.get("move"))
        self.assertNotIn("forfeit", third["error"].lower())

    def test_env_override(self):
        os.environ[ENV] = "2"
        self.assertEqual(max_illegal_moves(), 2)
        self.assertTrue(handle_illegal_make_move(1)["done"])
        os.environ[ENV] = "nope"
        self.assertEqual(max_illegal_moves(), 3)
        os.environ[ENV] = "0"
        self.assertEqual(max_illegal_moves(), 1)
        os.environ[ENV] = "100"
        self.assertEqual(max_illegal_moves(), 20)


class ToolRoundCapTests(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("XIANGQI_MAX_TOOL_ROUNDS")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("XIANGQI_MAX_TOOL_ROUNDS", None)
        else:
            os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = self._prev

    def test_env_default_and_clamp(self):
        os.environ.pop("XIANGQI_MAX_TOOL_ROUNDS", None)
        self.assertEqual(DEFAULT_MAX_TOOL_ROUNDS, 32)
        self.assertEqual(resolve_max_tool_rounds(), 32)
        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "nope"
        self.assertEqual(resolve_max_tool_rounds(), 32)
        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "2"
        self.assertEqual(resolve_max_tool_rounds(), 4)
        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "999"
        self.assertEqual(resolve_max_tool_rounds(), 128)
        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "16"
        self.assertEqual(resolve_max_tool_rounds(), 16)

    def test_explicit_constructor_ignores_env(self):
        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "16"
        player = _player(max_tool_rounds=3)
        self.assertEqual(player.max_tool_rounds, 3)


class ServerToolRoundEnvTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev = os.environ.get("XIANGQI_MAX_TOOL_ROUNDS")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("XIANGQI_MAX_TOOL_ROUNDS", None)
        else:
            os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = self._prev

    async def test_server_passes_env_cap_into_the_player(self):
        import server

        os.environ["XIANGQI_MAX_TOOL_ROUNDS"] = "9"
        seen: dict = {}

        class Capture:
            def __init__(self, *args, **kwargs):
                seen["max_tool_rounds"] = kwargs.get("max_tool_rounds")

            async def request_move(self, board, side):
                if False:
                    yield {}

        start = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
        game = server.GameSession(
            "tool-rounds-env",
            start,
            server.PlayerConfig(
                type="llm",
                model="example-model",
                api_base="https://llm.example/v1",
                api_key="sk-test",
                context_window=256000,
            ),
            server.PlayerConfig(type="random"),
        )
        with patch.object(server, "LangGraphPlayer", Capture):
            await server._request_llm_move(game, "w", "red", game.red_config)
        self.assertEqual(seen["max_tool_rounds"], 9)


def _player(**kwargs):
    from agent.player import LangGraphPlayer

    params = dict(
        api_base="http://example.invalid/v1",
        api_key="sk-test",
        model="example-model",
        game_id="illegal-ply",
        side_name="red",
        max_tool_rounds=8,
    )
    params.update(kwargs)
    return LangGraphPlayer(**params)


async def _play(player, script):
    """Drive request_move. ``script`` items are make_move ICCS, None (plain text),
    or a raw tool-call args string."""
    from langgraph.checkpoint.memory import MemorySaver
    from xiangqi import Board

    start = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

    calls = {"n": 0}

    async def fake_stream(_msgs):
        calls["n"] += 1
        item = script[min(calls["n"] - 1, len(script) - 1)]
        if item is None:
            return ("I think h2e2 is good but I won't call tools.", None, "stop", "")
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

    with patch(
        "agent.player.mem.get_checkpointer", AsyncMock(return_value=MemorySaver())
    ), patch(
        "agent.player.mem.prune_thread_to_latest", AsyncMock()
    ), patch(
        "agent.player.load_last_turn", lambda *_a, **_k: None
    ), patch.object(player, "_stream_completion", fake_stream):
        events = []
        async for ev in player.request_move(Board(start), "w"):
            events.append(ev)
    return calls["n"], events


class IllegalMoveGraphTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_repeated_illegal_move_interrupts_within_the_limit(self):
        calls, events = await _play(_player(), ["a0a9"])
        self.assertLessEqual(calls, 3)
        self.assertEqual(calls, 3)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("rejected make_move", events[-1]["message"])
        self.assertNotIn("forfeit", events[-1]["message"].lower())

    async def test_legal_move_after_illegal_ones_is_played(self):
        calls, events = await _play(_player(), ["a0a9", "a0a9", "h2e2"])
        self.assertEqual(calls, 3)
        self.assertEqual(events[-1]["type"], "move")
        self.assertEqual(events[-1]["move"], "h2e2")

    async def test_env_limit_stops_sooner(self):
        os.environ[ENV] = "2"
        calls, events = await _play(_player(), ["a0a9"])
        self.assertEqual(calls, 2)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("limit 2", events[-1]["message"])

    async def test_plain_text_garbage_stops_at_the_text_strike_cap(self):
        calls, events = await _play(_player(), [None])
        self.assertLessEqual(calls, 4)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("make_move", events[-1]["message"])
        self.assertNotIn("rejected make_move", events[-1]["message"])

    async def test_non_json_make_move_counts_as_rejected(self):
        calls, events = await _play(_player(), [("make_move", "not-json")])
        self.assertEqual(calls, 3)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("rejected make_move", events[-1]["message"])

    async def test_malformed_tool_call_hits_the_round_cap(self):
        player = _player(max_tool_rounds=4)
        calls, events = await _play(player, [("", "{")])
        self.assertLessEqual(calls, 4)
        self.assertGreaterEqual(calls, 1)
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("rounds", events[-1]["message"])


class GameLoopIllegalInterruptTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev = os.environ.get(ENV)
        os.environ[ENV] = "3"

    def tearDown(self):
        if self._prev is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._prev

    async def test_game_loop_interrupts_with_no_winner(self):
        import server
        from langgraph.checkpoint.memory import MemorySaver

        calls = {"n": 0}

        async def fake_stream(_self, _msgs):
            calls["n"] += 1
            return (
                "",
                [{"id": "c", "name": "make_move", "args": json.dumps({"move": "a0a9"})}],
                "tool_calls",
                "",
            )

        game = server.GameSession(
            "loop-illegal",
            "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w",
            server.PlayerConfig(
                type="llm",
                api_base="https://llm.example/v1",
                api_key="sk-test",
                model="example-model",
                context_window=256000,
            ),
            server.PlayerConfig(type="random"),
        )
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        with patch(
            "agent.player.mem.get_checkpointer", AsyncMock(return_value=MemorySaver())
        ), patch(
            "agent.player.mem.prune_thread_to_latest", AsyncMock()
        ), patch(
            "agent.player.load_last_turn", lambda *_a, **_k: None
        ), patch.object(
            server.LangGraphPlayer, "_stream_completion", fake_stream
        ), patch.object(
            server, "_persist_failed_llm_turn", new=AsyncMock()
        ):
            task = asyncio.create_task(server.game_loop(game))
            try:
                deadline = time.time() + 5
                while time.time() < deadline:
                    if game.status == "interrupted":
                        break
                    await asyncio.sleep(0.02)
                else:
                    self.fail(f"status stayed {game.status}")
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self.assertEqual(game.status, "interrupted")
        self.assertIsNone(game.winner)
        self.assertLessEqual(calls["n"], 3)
        self.assertIn("rejected make_move", game.reason or "")
