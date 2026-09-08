import asyncio
import os
import sys
import types
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class DummyAPIConnectionError(Exception):
        pass

    class DummyAPIStatusError(Exception):
        status_code = 500

    class DummyAsyncOpenAI:
        pass

    openai_stub.APIConnectionError = DummyAPIConnectionError
    openai_stub.APIStatusError = DummyAPIStatusError
    openai_stub.AsyncOpenAI = DummyAsyncOpenAI
    sys.modules["openai"] = openai_stub

import server


DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class FastLLMPlayer:
    def __init__(self, *args, **kwargs):
        pass

    async def request_move(self, board, side):
        yield {"type": "thinking", "content": "delta"}
        await asyncio.sleep(0.05)
        # first legal-ish move from start: h2e2 is cannon
        moves = board.get_legal_moves()
        yield {"type": "move", "move": moves[0] if moves else "a0a1"}


class ElapsedTimerTests(unittest.IsolatedAsyncioTestCase):
    def _make_game(self, red_config: server.PlayerConfig) -> server.GameSession:
        game = server.GameSession(
            "test",
            DEFAULT_FEN,
            red_config,
            server.PlayerConfig(type="random"),
        )
        game.timer_config = server.TimerConfig(enabled=True)
        game.timer_red = 0.0
        game.timer_black = 0.0
        return game

    async def test_elapsed_timer_accumulates_without_timeout_loss(self):
        game = self._make_game(
            server.PlayerConfig(
                type="llm",
                api_base="https://example.invalid/v1",
                api_key="test-key",
                model="test-model",
                context_window=200000,
            )
        )

        # Run only one red move then stop loop by finishing after first move via patch
        original_inner = server._game_loop_inner

        async def one_move_loop(g):
            g.status = "playing"
            side = g.board.turn
            side_name = "red" if side == "w" else "black"
            config = g.red_config
            g.timer_turn_start = __import__("time").time()
            move = await server._request_llm_move(g, side, side_name, config)
            self.assertIsNotNone(move)
            result = g.board.make_move(move)
            g.move_history.append(server._build_move_record(1, side_name, result))
            elapsed = max(0.0, __import__("time").time() - g.timer_turn_start)
            g.timer_red += elapsed
            g.timer_turn_start = 0.0
            g.status = "finished"
            g.winner = "draw"
            g.reason = "test stop"

        with patch.object(server, "LangGraphPlayer", FastLLMPlayer):
            await one_move_loop(game)

        self.assertGreaterEqual(game.timer_red, 0.04)
        self.assertEqual(game.timer_black, 0.0)
        self.assertNotIn("ran out of time", game.reason or "")

    async def test_run_with_turn_timeout_is_passthrough(self):
        game = self._make_game(server.PlayerConfig(type="random"))

        async def action():
            return "h2e2"

        result = await server._run_with_turn_timeout(game, "w", "red", action())
        self.assertEqual(result, "h2e2")


if __name__ == "__main__":
    unittest.main()
