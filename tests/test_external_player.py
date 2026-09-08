"""Tests for external (MCP) player type: /move + /resign endpoints, game_loop
waiting, pause/resume semantics, start_game local flag, external preset seats."""

import asyncio
import contextlib
import json
import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import server
import external_mcp

DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

EXTERNAL_RED = server.PlayerConfig(type="external", name="外部挑战者")
EXTERNAL_BLACK = server.PlayerConfig(type="external", name="外部挑战者B")
RANDOM_BLACK = server.PlayerConfig(type="random")


def _ext_vs_random(gid="exttest1") -> server.GameSession:
    return server.GameSession(gid, DEFAULT_FEN, EXTERNAL_RED, RANDOM_BLACK)


def _ext_vs_ext(gid="exttest2") -> server.GameSession:
    return server.GameSession(gid, DEFAULT_FEN, EXTERNAL_RED, EXTERNAL_BLACK)


def _claim(game: server.GameSession, side: str = "red") -> str:
    return external_mcp.issue_seat_token(game, side)


async def _move(game_id: str, side: str, move: str, token: str, **kwargs):
    note = kwargs.pop("note", None)
    tool = kwargs.pop("tool", "submit_move")
    return await server.submit_external_move(
        game_id,
        server.SubmitMoveRequest(side=side, move=move, note=note),
        x_xiangqi_seat_token=token,
        x_xiangqi_mcp_tool=tool,
    )


async def _wait(game_id: str, side: str, token: str, timeout_sec: float = 2.0):
    return await server.wait_turn(
        game_id,
        server.WaitTurnRequest(side=side, timeout_sec=timeout_sec),
        x_xiangqi_seat_token=token,
        x_xiangqi_mcp_tool="wait_my_turn",
    )


async def _resign(game_id: str, token: str, side: Optional[str] = "red"):
    return await server.resign_game(
        game_id,
        server.ResignRequest(side=side),
        x_xiangqi_seat_token=token,
        x_xiangqi_mcp_tool="resign",
    )


async def _wait_for(cond, timeout: float = 5.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


async def _start_loop(game: server.GameSession) -> asyncio.Task:
    task = asyncio.create_task(server.game_loop(game))
    await _wait_for(lambda: game.status == "playing")
    return task


async def _stop_loop(task: asyncio.Task) -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class ExternalMoveEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_illegal_move_rejected_with_retry_info(self):
        game = _ext_vs_random()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        task = await _start_loop(game)
        try:
            with self.assertRaises(server.HTTPException) as ctx:
                await _move(game.id, "red", "zzzz", tok)
            detail = ctx.exception.detail
            self.assertIsInstance(detail, dict)
            self.assertIn("h2e2", detail["legal_moves"])
            self.assertEqual(detail.get("error_class"), "rules")
            self.assertEqual(detail.get("reason_code"), "invalid_format")
            self.assertIn("ICCS", detail["reason"])
            self.assertTrue(detail["board"])
            self.assertTrue(detail["fen"])
            res = await _move(game.id, "red", "h2e2", tok)
            self.assertTrue(res["ok"])
            await _wait_for(lambda: len(game.move_history) >= 1)
            self.assertEqual(game.move_history[0]["move"], "h2e2")
        finally:
            await _stop_loop(task)

    async def test_illegal_move_reason_codes(self):
        game = _ext_vs_random("illreason")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        task = await _start_loop(game)
        try:
            # 马 b0 走 d1 是日字，但 c0 有相（蹩马腿）
            with self.assertRaises(server.HTTPException) as ctx:
                await _move(game.id, "red", "b0d1", tok)
            self.assertEqual(ctx.exception.detail.get("reason_code"), "knight_leg_blocked")
            self.assertIn("蹩马腿", ctx.exception.detail["reason"])
            # 马 b0 直走 b1（非日字）
            with self.assertRaises(server.HTTPException) as ctx:
                await _move(game.id, "red", "b0b1", tok)
            self.assertEqual(ctx.exception.detail.get("reason_code"), "piece_pattern")
            # 炮 h2 吃 h7 黑炮：之间（h3-h6）无炮架，不能隔空吃
            with self.assertRaises(server.HTTPException) as ctx:
                await _move(game.id, "red", "h2h7", tok)
            self.assertEqual(ctx.exception.detail.get("reason_code"), "cannon_no_screen")
            self.assertIn("炮架", ctx.exception.detail["reason"])
        finally:
            await _stop_loop(task)

    async def test_external_move_applied_and_random_replies(self):
        game = _ext_vs_random()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        task = await _start_loop(game)
        try:
            res = await _move(game.id, "red", "h2e2", tok, note="当头炮")
            self.assertTrue(res["ok"])
            await _wait_for(lambda: len(game.move_history) >= 2)
            self.assertEqual(game.move_history[0]["note"], "当头炮")
            self.assertEqual(game.move_history[1]["side"], "black")
            legal = game.board.get_legal_moves()
            await _move(game.id, "red", legal[0], tok)
            await _wait_for(lambda: len(game.move_history) >= 3)
        finally:
            await _stop_loop(task)

    async def test_move_requires_seat_token(self):
        game = _ext_vs_random("auth1")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        game.status = "playing"
        with self.assertRaises(server.HTTPException) as ctx:
            await server.submit_external_move(
                game.id, server.SubmitMoveRequest(side="red", move="h2e2")
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail.get("error_class"), "auth")

    async def test_wrong_side_token_rejected(self):
        game = _ext_vs_ext("auth2")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        game.status = "playing"
        red_tok = _claim(game, "red")
        with self.assertRaises(server.HTTPException) as ctx:
            await _move(game.id, "black", "h7e7", red_tok)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("not black", str(ctx.exception.detail).lower())

    async def test_turn_type_and_pending_guards(self):
        game = _ext_vs_random()
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        game.status = "playing"
        red_tok = _claim(game, "red")
        # non-external side cannot be claimed
        with self.assertRaises(server.HTTPException):
            _claim(game, "black")

        game2 = _ext_vs_ext()
        server.games[game2.id] = game2
        self.addCleanup(lambda: server.games.pop(game2.id, None))
        game2.status = "playing"
        red2 = _claim(game2, "red")
        black2 = _claim(game2, "black")
        with self.assertRaises(server.HTTPException) as ctx2:
            await _move(game2.id, "black", "h2e2", black2)
        self.assertIn("Not black's turn", str(ctx2.exception.detail))
        res = await _move(game2.id, "red", "h2e2", red2)
        self.assertTrue(res["ok"])
        with self.assertRaises(server.HTTPException) as ctx3:
            await _move(game2.id, "red", "b2e2", red2)
        self.assertIn("pending move", str(ctx3.exception.detail))
        del red_tok  # silence unused in first branch

    async def test_paused_game_rejects_move_and_resume_consumes_pending(self):
        game = _ext_vs_random("exttest3")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        game.status = "paused"
        with self.assertRaises(server.HTTPException) as ctx:
            await _move(game.id, "red", "h2e2", tok)
        self.assertIn("not playing", str(ctx.exception.detail))
        # 复盘/预盘阶段禁用走子：错误附阶段 status 与中文指引
        detail = ctx.exception.detail
        self.assertEqual(detail.get("error_class"), "state")
        self.assertEqual(detail.get("status"), "paused")
        self.assertIn("复盘/预盘", detail.get("error", ""))
        game.external_moves["red"].put_nowait({"move": "h2e2", "note": ""})
        await server.resume_game(game.id)
        try:
            await _wait_for(lambda: len(game.move_history) >= 2)
            self.assertEqual(game.move_history[0]["move"], "h2e2")
            self.assertEqual(game.move_history[1]["side"], "black")
        finally:
            if game.task:
                await _stop_loop(game.task)

    async def test_wait_external_move_normalizes_queue_payload(self):
        game = _ext_vs_random("exttest4")
        game.external_moves["red"].put_nowait({"move": "  H2E2  ", "note": " 中炮 "})
        queued = await server._wait_external_move(game, "red")
        self.assertEqual(str(queued.get("move") or "").strip().lower(), "h2e2")
        self.assertEqual(str(queued.get("note") or "").strip(), "中炮")

    async def test_non_playing_after_dequeue_requeues_move(self):
        game = _ext_vs_ext("requeue1")
        payload = {"move": "h2e2", "note": "keep"}
        game.external_pending["red"] = dict(payload)
        game.external_moves["red"].put_nowait(dict(payload))
        queued = await server._wait_external_move(game, "red")
        game.status = "paused"
        server._requeue_external_move(game, "red", queued)
        self.assertEqual(game.external_moves["red"].qsize(), 1)
        self.assertEqual(game.external_pending["red"]["move"], "h2e2")
        self.assertEqual(game.external_pending["red"]["note"], "keep")
        self.assertNotIn("red", game.external_notes)

    async def test_pause_cancel_requeues_in_flight_external_move(self):
        """pause mid-wait must not drop an already-dequeued /move."""
        game = _ext_vs_ext("requeue2")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        task = await _start_loop(game)
        try:
            await _move(game.id, "red", "h2e2", tok, note="race")
            await asyncio.sleep(0.05)
            await server.pause_game(game.id)
            self.assertEqual(game.status, "paused")
            applied = any(m.get("move") == "h2e2" for m in game.move_history)
            pending = game.external_pending.get("red")
            queued = game.external_moves["red"].qsize()
            self.assertTrue(
                applied or (pending and pending.get("move") == "h2e2") or queued >= 1,
                f"move lost: history={game.move_history} pending={pending} qsize={queued}",
            )
            if not applied:
                await server.resume_game(game.id)
                await _wait_for(lambda: any(m.get("move") == "h2e2" for m in game.move_history))
                self.assertEqual(game.move_history[0].get("note"), "race")
        finally:
            if game.task:
                await _stop_loop(game.task)

    async def test_move_on_missing_game_404(self):
        with self.assertRaises(server.HTTPException) as ctx:
            await server.submit_external_move(
                "nope", server.SubmitMoveRequest(side="red", move="h2e2")
            )
        self.assertEqual(ctx.exception.status_code, 404)


class ResignTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for attr in ("SNAPSHOT_DIR", "LOG_DIR"):
            p = patch.object(server, attr, str(Path(self.tmp.name) / attr.lower()))
            p.start()
            self.addCleanup(p.stop)
        (Path(self.tmp.name) / "snapshots").mkdir()
        (Path(self.tmp.name) / "log_dir").mkdir()

    async def test_resign_finishes_game_and_writes_log(self):
        game = _ext_vs_random("exttest5")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        task = await _start_loop(game)
        game.task = task
        with patch.object(server, "clear_game_threads", AsyncMock()):
            result = await _resign(game.id, tok, side="red")
        self.assertEqual(result["winner"], "black")
        self.assertEqual(result["reason"], "resignation")
        self.assertEqual(game.status, "finished")
        self.assertEqual(game.winner, "black")
        self.assertTrue(task.done())
        logs = list(Path(server.LOG_DIR).glob(f"*_{game.id}.log"))
        self.assertEqual(len(logs), 1)

    async def test_resign_defaults_to_side_to_move(self):
        game = _ext_vs_random("exttest6")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        game.status = "playing"
        result = await _resign(game.id, tok, side=None)
        self.assertEqual(result["winner"], "black")  # 开局轮红，缺省红认输
        self.assertEqual(game.status, "finished")

    async def test_resign_rejected_when_not_playing(self):
        game = _ext_vs_random("exttest7")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        game.status = "paused"
        with self.assertRaises(server.HTTPException):
            await _resign(game.id, tok, side="red")

    async def test_resign_requires_token(self):
        game = _ext_vs_random("exttest7b")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        game.status = "playing"
        with self.assertRaises(server.HTTPException) as ctx:
            await server.resign_game(game.id, server.ResignRequest(side="red"))
        self.assertEqual(ctx.exception.status_code, 401)


class StartAndPresetTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_game_local_flag(self):
        for red, black, expect_local in (
            (server.PlayerConfig(type="random"), server.PlayerConfig(type="random"), True),
            (EXTERNAL_RED, EXTERNAL_BLACK, False),
            (EXTERNAL_RED, RANDOM_BLACK, False),
        ):
            gid = f"start_{red.type}_{black.type}"
            game = server.GameSession(gid, DEFAULT_FEN, red, black)
            server.games[gid] = game
            self.addCleanup(lambda g=gid: server.games.pop(g, None))
            result = await server.start_game(gid)
            self.assertEqual(result["local"], expect_local, (red.type, black.type))
            if game.task:
                await _stop_loop(game.task)
                game.task = None

    def test_validate_player_type(self):
        server._validate_player_type(server.PlayerConfig(type="external"))
        with self.assertRaises(server.HTTPException):
            server._validate_player_type(server.PlayerConfig(type="human"))

    async def test_state_includes_sides(self):
        game = _ext_vs_random("exttest8")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        st = await server.get_game_state(game.id)
        self.assertEqual(st["sides"]["red"]["type"], "external")
        self.assertEqual(st["sides"]["red"]["name"], "外部挑战者")
        self.assertEqual(st["sides"]["black"]["type"], "random")

    def test_player_label_external(self):
        self.assertEqual(server._player_label(EXTERNAL_RED), "外部挑战者 · External agent")
        seat = server.PlayerConfig(type="external", name="老外", preset="ext_seat")
        self.assertEqual(server._player_label(seat), "老外 · External (ext_seat)")


class ExternalPresetTests(unittest.TestCase):
    def _external_preset(self) -> dict:
        return {"name": "ext_seat", "type": "external", "display_name": "外部挑战者A"}

    def _llm_preset(self) -> dict:
        return {
            "name": "glm53",
            "api_base": "https://example.invalid/v1",
            "api_key": "k",
            "model": "glm-5.3",
            "context_window": 256000,
        }

    def test_resolve_preset_external_seat(self):
        with patch.object(server, "load_model_presets", return_value=[self._external_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="external", preset="ext_seat")
            )
            self.assertEqual(cfg.type, "external")
            self.assertEqual(cfg.name, "外部挑战者A")
            # runner 旧载荷（type=llm）引用 external 席也映射为 external
            cfg2 = server.resolve_preset(
                server.PlayerConfig(type="llm", preset="ext_seat")
            )
            self.assertEqual(cfg2.type, "external")
            # llm 玩家引用 llm preset 不受影响
            with patch.object(server, "load_model_presets", return_value=[self._llm_preset()]):
                cfg3 = server.resolve_preset(
                    server.PlayerConfig(type="llm", preset="glm53")
                )
            self.assertEqual(cfg3.type, "llm")
            self.assertEqual(cfg3.model, "glm-5.3")

    def test_external_seat_cannot_be_used_by_llm_preset(self):
        with patch.object(server, "load_model_presets", return_value=[self._llm_preset()]):
            with self.assertRaises(server.HTTPException):
                server.resolve_preset(
                    server.PlayerConfig(type="external", preset="glm53")
                )

    def test_unknown_llm_preset_still_400(self):
        with patch.object(server, "load_model_presets", return_value=[]):
            with self.assertRaises(server.HTTPException):
                server.resolve_preset(
                    server.PlayerConfig(type="llm", preset="ghost")
                )


class SeekClearsExternalPendingTests(unittest.IsolatedAsyncioTestCase):
    async def test_seek_drops_stale_pending_move(self):
        game = _ext_vs_ext("seekext1")
        game.status = "paused"
        game.external_moves["red"].put_nowait({"move": "h2e2", "note": "stale"})
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        await server.seek_game(game.id, server.SeekGameRequest(ply=0))
        self.assertEqual(game.external_moves["red"].qsize(), 0)
        self.assertEqual(game.external_moves["black"].qsize(), 0)


class SnapshotExternalPendingTests(unittest.TestCase):
    def test_snapshot_roundtrips_pending_external_move(self):
        game = _ext_vs_random("snappext1")
        game.status = "interrupted"
        game.reason = "paused"
        payload = {"move": "h2e2", "note": "当头炮"}
        game.external_pending["red"] = dict(payload)
        game.external_moves["red"].put_nowait(dict(payload))
        with TemporaryDirectory() as tmp:
            with patch.object(server, "SNAPSHOT_DIR", tmp):
                path = server.write_game_snapshot(game, reason="paused")
                restored = server.load_game_runtime_snapshot(path)
        self.assertEqual(restored.external_moves["red"].qsize(), 1)
        self.assertEqual(restored.external_pending["red"]["move"], "h2e2")
        pending = restored.external_moves["red"].get_nowait()
        self.assertEqual(pending["move"], "h2e2")
        self.assertEqual(pending["note"], "当头炮")


class LegalMovesEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_legal_moves_uses_live_board(self):
        game = _ext_vs_random("legal1")
        game.status = "playing"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        out = await server.get_legal_moves(game.id)
        self.assertEqual(out["game_id"], game.id)
        self.assertGreaterEqual(out["legal_count"], 40)
        self.assertIn("h2e2", out["legal_moves"])
        self.assertEqual(out["legal_moves"], game.board.get_legal_moves())


class WaitTurnEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_wait_turn_immediate_when_already_my_turn(self):
        game = _ext_vs_ext("wait1")
        game.status = "playing"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        out = await _wait(game.id, "red", tok, timeout_sec=2.0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["waiting_done_reason"], "my_turn")
        self.assertEqual(out["turn"], "red")

    async def test_wait_turn_wakes_after_opponent_move(self):
        game = _ext_vs_ext("wait2")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        red_tok = _claim(game, "red")
        black_tok = _claim(game, "black")
        task = await _start_loop(game)
        try:
            await _move(game.id, "red", "h2e2", red_tok)
            await _wait_for(lambda: len(game.move_history) >= 1)
            waiter = asyncio.create_task(_wait(game.id, "red", red_tok, timeout_sec=5.0))
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            await _move(game.id, "black", "h7e7", black_tok)
            out = await asyncio.wait_for(waiter, timeout=3.0)
            self.assertTrue(out["ok"])
            self.assertEqual(out["waiting_done_reason"], "my_turn")
            self.assertEqual(out["turn"], "red")
            self.assertEqual(out["last_move"]["move"], "h7e7")
        finally:
            await _stop_loop(task)

    async def test_wait_turn_returns_finished_on_resign(self):
        game = _ext_vs_ext("wait3")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        red_tok = _claim(game, "red")
        black_tok = _claim(game, "black")
        task = await _start_loop(game)
        try:
            waiter = asyncio.create_task(_wait(game.id, "black", black_tok, timeout_sec=5.0))
            await asyncio.sleep(0.05)
            await _resign(game.id, red_tok, side="red")
            out = await asyncio.wait_for(waiter, timeout=3.0)
            self.assertEqual(out["waiting_done_reason"], "finished")
            self.assertEqual(out["winner"], "black")
        finally:
            await _stop_loop(task)

    async def test_wait_turn_timeout(self):
        game = _ext_vs_ext("wait4")
        game.status = "playing"
        game.board.turn = "b"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        out = await _wait(game.id, "red", tok, timeout_sec=0.15)
        # 超时是正常结果：ok=true、无 error_class，附建议重试间隔
        self.assertTrue(out["ok"])
        self.assertEqual(out["waiting_done_reason"], "timeout")
        self.assertNotIn("error_class", out)
        self.assertIsInstance(out.get("suggested_retry_sec"), int)

    async def test_wait_turn_blocks_until_start(self):
        game = _ext_vs_ext("wait5")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        waiter = asyncio.create_task(_wait(game.id, "red", tok, timeout_sec=3.0))
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        await server.start_game(game.id)
        out = await asyncio.wait_for(waiter, timeout=3.0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["waiting_done_reason"], "my_turn")
        if game.task and not game.task.done():
            await _stop_loop(game.task)


class ClaimSeatContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_seat_embeds_contract_and_marks_claimed(self):
        red = server.PlayerConfig(type="external", name="外部挑战者")
        black = server.PlayerConfig(type="external", name="外部挑战者B")
        game = server.GameSession("claim1", DEFAULT_FEN, red, black)
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        out = await server.claim_seat(
            game.id, server.ClaimSeatRequest(side="red", name="赤练")
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["side"], "red")
        self.assertTrue(out["seat_token"])
        self.assertEqual(out["contract_version"], "mcp-agent-v1")
        self.assertIn("player_brief", out)
        self.assertIn("forbid", out["rules"])
        self.assertEqual(red.name, "赤练")
        state = await server.get_game_state(game.id)
        self.assertTrue(state["sides"]["red"]["claimed"])
        self.assertFalse(state["sides"]["black"]["claimed"])
        listed = await server.list_active_games()
        entry = next(g for g in listed["games"] if g["id"] == game.id)
        red_seat = next(s for s in entry["external_seats"] if s["side"] == "red")
        black_seat = next(s for s in entry["external_seats"] if s["side"] == "black")
        self.assertTrue(red_seat["claimed"])
        self.assertFalse(red_seat["joinable"])
        self.assertTrue(black_seat["joinable"])


class PreviewAndAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_does_not_change_live_and_resets_after_submit(self):
        game = _ext_vs_random("prev1")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        tok = _claim(game, "red")
        live_fen = game.board.to_fen()
        out = await server.preview_move(
            game.id,
            server.PreviewMoveRequest(move="h2e2"),
            x_xiangqi_seat_token=tok,
            x_xiangqi_mcp_tool="preview",
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["path"], ["h2e2"])
        self.assertEqual(game.board.to_fen(), live_fen)
        self.assertNotEqual(out["fen_work"], live_fen)
        task = await _start_loop(game)
        try:
            await _move(game.id, "red", "h2e2", tok)
            await _wait_for(lambda: len(game.move_history) >= 1)
            obs = external_mcp.get_observe(game, "red")
            self.assertEqual(obs.path, [])
            self.assertEqual(obs.work.to_fen(), game.board.to_fen())
        finally:
            await _stop_loop(task)

    async def test_tool_audit_jsonl_and_events(self):
        game = _ext_vs_random("audit1")
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        with TemporaryDirectory() as tmp:
            with patch.object(server, "LOG_DIR", tmp):
                tok = _claim(game, "red")
                await server.get_game_state(
                    game.id,
                    x_xiangqi_seat_token=tok,
                    x_xiangqi_mcp_tool="get_board",
                )
                await server.preview_move(
                    game.id,
                    server.PreviewMoveRequest(move="h2e2"),
                    x_xiangqi_seat_token=tok,
                    x_xiangqi_mcp_tool="preview",
                )
                task = await _start_loop(game)
                try:
                    await _move(game.id, "red", "h2e2", tok, tool="submit_move")
                    await _wait_for(lambda: len(game.move_history) >= 1)
                finally:
                    await _stop_loop(task)
                path = external_mcp.mcp_jsonl_path(tmp, game.id)
                self.assertTrue(os.path.exists(path))
                lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
                self.assertGreaterEqual(len(lines), 4)
                tools = {json.loads(line)["tool"] for line in lines}
                self.assertIn("get_board", tools)
                self.assertIn("preview", tools)
                self.assertIn("submit_move", tools)
                types = {e["type"] for e in game.events}
                self.assertIn("tool_call", types)
                self.assertIn("tool_result", types)


if __name__ == "__main__":
    unittest.main()
