import asyncio
import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

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
from xiangqi import Board

DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class FailThenMovePlayer:
    def __init__(self, *args, **kwargs):
        self.last_turn_trace = {
            "reasoning": "looked at board",
            "thinking": "will submit later",
            "assistant_content": "will submit later",
            "tool_rounds": [
                {
                    "tool_calls": [{"name": "get_board", "args": {}}],
                    "tool_results": [{"name": "get_board", "content": "FEN"}],
                }
            ],
        }

    async def request_move(self, board, side):
        if getattr(FailThenMovePlayer, "fail_once", True):
            FailThenMovePlayer.fail_once = False
            yield {"type": "error", "message": "make_move was not called after 3 prompts."}
            return
        yield {"type": "move", "move": "h2e2"}


class GameSnapshotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.snap_dir = Path(self.tmp.name) / "snapshots"
        self.snap_dir.mkdir()
        p = patch.object(server, "SNAPSHOT_DIR", str(self.snap_dir))
        p.start()
        self.addCleanup(p.stop)
        FailThenMovePlayer.fail_once = True

    def _make_game(self) -> server.GameSession:
        game = server.GameSession(
            "snap1",
            DEFAULT_FEN,
            server.PlayerConfig(
                type="llm",
                api_base="https://example.invalid/v1",
                api_key="k",
                model="m",
                context_window=200000,
            ),
            server.PlayerConfig(type="random"),
        )
        return game

    async def test_llm_error_pauses_with_snapshot_instead_of_finishing(self):
        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            await server._game_loop_inner(game)

        self.assertEqual(game.status, "interrupted")
        self.assertIsNone(game.winner)
        self.assertTrue(game.reason)
        self.assertEqual(len(game.move_history), 0)
        self.assertEqual(game.board.to_fen().split()[0], DEFAULT_FEN.split()[0])
        snap_path = Path(server.SNAPSHOT_DIR) / f"{game.id}.json"
        self.assertTrue(snap_path.is_file())

    async def test_interrupt_writes_game_log(self):
        log_dir = Path(self.tmp.name) / "logs"
        log_dir.mkdir()
        p = patch.object(server, "LOG_DIR", str(log_dir))
        p.start()
        self.addCleanup(p.stop)

        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            await server.game_loop(game)

        self.assertEqual(game.status, "interrupted")
        logs = list(log_dir.glob(f"*_{game.id}.log"))
        self.assertEqual(len(logs), 1)
        parsed = server.parse_game_log(str(logs[0]), include_moves=True)
        self.assertEqual(parsed["game_id"], game.id)
        self.assertIsNone(parsed.get("winner"))
        self.assertTrue(parsed.get("reason") or parsed.get("result"))

    async def test_llm_error_writes_failed_turn_jsonl(self):
        from agent import turn_store

        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        turns_dir = Path(self.tmp.name) / "turns"
        turns_dir.mkdir()
        p = patch.object(turn_store, "TURNS_DIR", turns_dir)
        p.start()
        self.addCleanup(p.stop)

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            await server._game_loop_inner(game)

        rows = turn_store.iter_turns(game.id)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["side"], "red")
        self.assertEqual(row.get("move") or "", "")
        self.assertTrue(row.get("error"))
        self.assertIn("make_move was not called", row["error"])
        self.assertTrue(row.get("tool_rounds"))
        self.assertEqual(row.get("fen_before"), game.board.to_fen())

    async def test_resume_interrupted_game_continues_from_same_position(self):
        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        FailThenMovePlayer.fail_once = True

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            await server._game_loop_inner(game)
            self.assertEqual(game.status, "interrupted")
            game.status = "playing"
            game.pause_event.set()
            move = await server._request_llm_move(
                game, game.board.turn, "red", game.red_config
            )
            self.assertEqual(move, "h2e2")
            result = game.board.make_move(move)
            game.move_history.append(server._build_move_record(1, "red", result))

        self.assertEqual(game.move_history[0]["move"], "h2e2")
        self.assertNotEqual(game.board.to_fen().split()[0], DEFAULT_FEN.split()[0])

    def test_load_snapshot_restores_board_and_history(self):
        game = self._make_game()
        board = Board(DEFAULT_FEN)
        result = board.make_move("h2e2")
        game.board = board
        game.move_history.append(server._build_move_record(1, "red", result))
        game.status = "interrupted"
        game.reason = "provider down"
        path = server.write_game_snapshot(game, reason=game.reason)
        restored = server.load_game_runtime_snapshot(path)
        self.assertEqual(restored.id, game.id)
        self.assertEqual(restored.status, "interrupted")
        self.assertEqual(len(restored.move_history), 1)
        self.assertEqual(restored.board.to_fen(), game.board.to_fen())
        self.assertEqual(restored.red_config.type, "llm")

    def test_snapshot_does_not_persist_api_key(self):
        dummy = "TEST-PLACEHOLDER-DO-NOT-PERSIST"
        game = server.GameSession(
            "snap-secret",
            DEFAULT_FEN,
            server.PlayerConfig(
                type="llm",
                preset="flash_official",
                api_base="https://example.invalid/v1",
                api_key=dummy,
                model="m",
            ),
            server.PlayerConfig(type="random"),
        )
        path = server.write_game_snapshot(game, reason="provider down")
        raw = Path(path).read_text(encoding="utf-8")
        self.assertNotIn(dummy, raw)
        with patch.object(server, "load_model_presets", return_value=[]):
            restored = server.load_game_runtime_snapshot(path)
        self.assertNotEqual(restored.red_config.api_key, dummy)
        self.assertFalse(restored.red_config.api_key)
        self.assertEqual(restored.red_config.preset, "flash_official")
        self.assertEqual(restored.red_config.type, "llm")

    def test_load_snapshot_rehydrates_preset_credentials(self):
        live_dummy = "TEST-PLACEHOLDER-NEVER-ON-DISK"
        preset_dummy = "TEST-PLACEHOLDER-FROM-PRESET"
        game = server.GameSession(
            "snap-rehydrate",
            DEFAULT_FEN,
            server.PlayerConfig(
                type="llm",
                preset="flash_official",
                api_base="https://example.invalid/v1",
                api_key=live_dummy,
                model="stale-model",
            ),
            server.PlayerConfig(type="random"),
        )
        path = server.write_game_snapshot(game, reason="provider down")
        self.assertNotIn(live_dummy, Path(path).read_text(encoding="utf-8"))
        presets = [
            {
                "name": "flash_official",
                "api_base": "https://preset.example/v1",
                "api_key": preset_dummy,
                "model": "preset-model",
                "prompt_name": "zh",
                "enable_thinking": True,
                "max_completion_tokens": 8192,
                "context_window": 256000,
                "compress_threshold_ratio": 0.9,
            }
        ]
        with patch.object(server, "load_model_presets", return_value=presets):
            restored = server.load_game_runtime_snapshot(path)
        self.assertEqual(restored.red_config.api_key, preset_dummy)
        self.assertEqual(restored.red_config.api_base, "https://preset.example/v1")
        self.assertEqual(restored.red_config.model, "preset-model")
        self.assertNotEqual(restored.red_config.api_key, live_dummy)

    async def test_finished_game_clears_redis_threads(self):
        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        async def finish_now(g):
            g.status = "finished"
            g.winner = "red"
            g.reason = "checkmate"

        with patch.object(server, "_game_loop_inner", finish_now):
            with patch.object(server, "clear_game_threads", AsyncMock()) as clear:
                await server.game_loop(game)
        clear.assert_awaited_once_with(game.id)

    async def test_interrupted_game_keeps_redis_threads(self):
        game = self._make_game()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            with patch.object(server, "clear_game_threads", AsyncMock()) as clear:
                await server.game_loop(game)
        self.assertEqual(game.status, "interrupted")
        clear.assert_not_awaited()

    async def test_interrupted_game_releases_eval_engine(self):
        game = self._make_game()
        engine = AsyncMock()
        game.pikafish = engine
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        with patch.object(server, "LangGraphPlayer", FailThenMovePlayer):
            with patch.object(server, "clear_game_threads", AsyncMock()) as clear:
                await server.game_loop(game)
        self.assertEqual(game.status, "interrupted")
        engine.shutdown.assert_awaited()
        self.assertIsNone(game.pikafish)
        clear.assert_not_awaited()

    async def test_reset_awaits_cancelled_task_before_clearing_redis(self):
        game = self._make_game()
        order: list[str] = []

        async def lingering():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append("cancelled")
                raise

        async def clear(_gid):
            order.append("clear")

        game.task = asyncio.create_task(lingering())
        await asyncio.sleep(0)
        server.games[game.id] = game
        with patch.object(server, "clear_game_threads", clear):
            await server.reset_game(game.id)
        self.assertEqual(order, ["cancelled", "clear"])
        self.assertNotIn(game.id, server.games)

    async def test_finish_without_winner_does_not_drop_session(self):
        game = self._make_game()
        game.status = "playing"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        req = server.FinishGameRequest(moves=[], winner=None, reason="")

        with patch.object(server, "clear_game_threads", AsyncMock()) as clear:
            with patch.object(server.asyncio, "get_event_loop") as loop:
                later = loop.return_value.call_later
                result = await server.finish_game(game.id, req)

        self.assertEqual(result["status"], "finished")
        self.assertIsNone(result["winner"])
        self.assertEqual(game.status, "playing")
        self.assertIn(game.id, server.games)
        clear.assert_not_awaited()
        later.assert_not_called()

    async def test_pause_writes_snapshot(self):
        game = self._make_game()
        game.status = "playing"

        async def hang():
            await asyncio.sleep(3600)

        game.task = asyncio.create_task(hang())
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))

        result = await server.pause_game(game.id)
        self.assertEqual(result["status"], "paused")
        self.assertEqual(game.status, "paused")
        snap_path = Path(server.SNAPSHOT_DIR) / f"{game.id}.json"
        self.assertTrue(snap_path.is_file())
        payload = json.loads(snap_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "paused")
        self.assertEqual(payload["game_id"], game.id)
        self.assertEqual(payload["fen"], game.board.to_fen())


class PresetDisplayNameTests(unittest.TestCase):
    def _flash_preview_preset(self) -> dict:
        return {
            "name": "flash_preview",
            "display_name": "DeepSeek V4 Flash",
            "api_base": "https://example.invalid/v1",
            "api_key": "k",
            "model": "xopdeepseekv4flash",
            "prompt_name": "zh",
            "enable_thinking": True,
            "max_completion_tokens": 8192,
            "context_window": 256000,
        }

    def test_resolve_preset_uses_yaml_display_name(self):
        with patch.object(server, "load_model_presets", return_value=[self._flash_preview_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="llm", preset="flash_preview", name="flash_preview")
            )
        self.assertEqual(cfg.name, "DeepSeek V4 Flash")
        self.assertEqual(
            server._player_label(cfg),
            "DeepSeek V4 Flash · LLM (flash_preview)",
        )

    def test_resolve_preset_keeps_custom_name(self):
        with patch.object(server, "load_model_presets", return_value=[self._flash_preview_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="llm", preset="flash_preview", name="自定义红")
            )
        self.assertEqual(cfg.name, "自定义红")


if __name__ == "__main__":
    unittest.main()


