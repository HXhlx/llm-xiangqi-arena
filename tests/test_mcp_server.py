"""Tests for the MCP tool layer (mcp_server) with a mocked duel-server REST."""

import os
import sys
import unittest
from typing import Any, Callable

import httpx

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mcp_server

DEFAULT_FEN = mcp_server.DEFAULT_FEN
SEAT_TOKEN_FIXTURE = "test-seat-token-red"


def _state(**over) -> dict:
    base = {
        "game_id": "g1",
        "fen": DEFAULT_FEN,
        "turn": "red",
        "status": "playing",
        "winner": None,
        "reason": None,
        "move_count": 0,
        "move_history": [],
        "sides": {
            "red": {"type": "external", "name": "外部挑战者", "label": "外部挑战者 · External agent"},
            "black": {"type": "llm", "name": "GLM-5.3", "label": "GLM-5.3 · LLM (glm53)"},
        },
    }
    base.update(over)
    return base


def _json_response(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


class McpToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.routes: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            fn = self.routes.get(f"{request.method} {request.url.path}")
            if fn is None:
                return _json_response({"detail": "Not Found"}, 404)
            return fn(request)

        self._prev_transport = mcp_server._TEST_TRANSPORT
        mcp_server._TEST_TRANSPORT = httpx.MockTransport(handler)
        self.addCleanup(setattr, mcp_server, "_TEST_TRANSPORT", self._prev_transport)

    def route(self, method: str, path: str, fn: Callable[[httpx.Request], httpx.Response]):
        self.routes[f"{method} {path}"] = fn

    def _request_body(self, method: str, path: str) -> Any:
        for req in self.requests:
            if req.method == method and req.url.path == path:
                return _parse_body(req)
        raise AssertionError(f"{method} {path} was not requested")

    def _request(self, method: str, path: str) -> httpx.Request:
        for req in self.requests:
            if req.method == method and req.url.path == path:
                return req
        raise AssertionError(f"{method} {path} was not requested")


def _parse_body(request: httpx.Request) -> Any:
    import json

    return json.loads(request.content.decode("utf-8")) if request.content else None


class GetBoardTests(McpToolTests):
    async def test_get_board_renders_local_board_and_hint(self):
        self.route(
            "GET",
            "/api/game/g1/state",
            lambda req: _json_response(_state(move_count=1, move_history=[
                {"number": 1, "side": "red", "move": "h2e2", "move_zh": "炮二平五", "captured": None},
            ], seat="red", board="Side to move: Red\nC")),
        )
        out = await mcp_server.get_board("g1", SEAT_TOKEN_FIXTURE)
        self.assertTrue(out["ok"])
        self.assertEqual(out["fen"], DEFAULT_FEN)
        self.assertTrue(out["board"])
        self.assertEqual(out["turn"], "red")
        self.assertEqual(out["last_moves"][0]["move_zh"], "炮二平五")
        self.assertIn("submit_move", out["hint"])
        req = self._request("GET", "/api/game/g1/state")
        self.assertEqual(req.headers.get("x-xiangqi-seat-token"), SEAT_TOKEN_FIXTURE)
        self.assertEqual(req.headers.get("x-xiangqi-mcp-tool"), "get_board")

    async def test_get_board_missing_token(self):
        out = await mcp_server.get_board("g1", "")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_class"], "auth")

    async def test_get_board_missing_game(self):
        out = await mcp_server.get_board("nope", SEAT_TOKEN_FIXTURE)
        self.assertFalse(out["ok"])
        self.assertIn("Not Found", out["error"])
        self.assertEqual(out["error_class"], "rules")

    async def test_get_legal_moves_from_server(self):
        self.route(
            "GET",
            "/api/game/g1/legal_moves",
            lambda req: _json_response({
                "game_id": "g1",
                "turn": "red",
                "fen": DEFAULT_FEN,
                "board": "board-text",
                "legal_moves": ["h2e2", "b2e2", "a0a1"],
                "legal_count": 3,
                "status": "playing",
                "annotated": "annotated",
                "path": [],
            }),
        )
        out = await mcp_server.get_legal_moves("g1", SEAT_TOKEN_FIXTURE)
        self.assertTrue(out["ok"])
        self.assertEqual(out["legal_count"], 3)
        self.assertIn("h2e2", out["legal_moves"])
        self.assertEqual(out["board"], "board-text")
        req = self._request("GET", "/api/game/g1/legal_moves")
        self.assertEqual(req.headers.get("x-xiangqi-seat-token"), SEAT_TOKEN_FIXTURE)


class SubmitMoveTests(McpToolTests):
    async def test_submit_move_success_includes_state_after(self):
        calls = {"n": 0}

        def state_handler(_req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response(_state(move_count=0, move_history=[]))
            return _json_response(_state(
                move_count=1,
                turn="black",
                move_history=[{
                    "number": 1, "side": "red", "move": "h2e2",
                    "move_zh": "炮二平五", "captured": None,
                }],
            ))

        self.route("GET", "/api/game/g1/state", state_handler)
        self.route("POST", "/api/game/g1/move", lambda req: _json_response({
            "ok": True, "accepted": "h2e2", "turn": "red", "pending": 1,
        }))
        out = await mcp_server.submit_move(
            "g1", "h2e2", SEAT_TOKEN_FIXTURE, side="red", note="当头炮"
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["accepted"], "h2e2")
        self.assertEqual(out["pending"], 1)
        body = self._request_body("POST", "/api/game/g1/move")
        self.assertEqual(body, {"side": "red", "move": "h2e2", "note": "当头炮"})
        move_req = self._request("POST", "/api/game/g1/move")
        self.assertEqual(move_req.headers.get("x-xiangqi-seat-token"), SEAT_TOKEN_FIXTURE)
        self.assertEqual(move_req.headers.get("x-xiangqi-mcp-tool"), "submit_move")
        self.assertEqual(out["state_after"]["move_count"], 1)
        self.assertEqual(out["state_after"]["turn"], "black")
        self.assertTrue(out["applied"])

    async def test_submit_move_uses_turn_side_by_default(self):
        calls = {"n": 0}

        def state_handler(_req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response(_state(turn="black", move_count=0, move_history=[]))
            return _json_response(_state(
                turn="red",
                move_count=1,
                move_history=[{
                    "number": 1, "side": "black", "move": "b2e2",
                    "move_zh": "炮八平五", "captured": None,
                }],
            ))

        self.route("GET", "/api/game/g1/state", state_handler)
        self.route("POST", "/api/game/g1/move", lambda req: _json_response({"ok": True, "pending": 1}))
        out = await mcp_server.submit_move("g1", "b2e2", SEAT_TOKEN_FIXTURE)
        self.assertTrue(out["ok"])
        body = self._request_body("POST", "/api/game/g1/move")
        self.assertEqual(body["side"], "black")

    async def test_submit_move_not_applied_returns_ok_false(self):
        # 走子前 playing，落子后轮询发现已被 pause（竞态场景）
        calls = {"n": 0}

        def state_handler(_req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response(_state(move_count=0, move_history=[]))
            return _json_response(_state(move_count=0, move_history=[], status="paused"))

        self.route("GET", "/api/game/g1/state", state_handler)
        self.route("POST", "/api/game/g1/move", lambda req: _json_response({
            "ok": True, "accepted": "h2e2", "turn": "red", "pending": 1,
        }))
        out = await mcp_server.submit_move("g1", "h2e2", SEAT_TOKEN_FIXTURE, side="red")
        self.assertFalse(out["ok"])
        self.assertFalse(out["applied"])
        self.assertEqual(out["accepted"], "h2e2")
        self.assertIn("not yet applied", out["error"])
        self.assertEqual(out["error_class"], "state")

    async def _assert_move_disabled_in_phase(self, phase: str, want_hint: str):
        self.route("GET", "/api/game/g1/state", lambda req: _json_response(_state(status=phase)))

        def _no_move(_req):
            raise AssertionError(f"submit_move should be disabled in {phase}, POST /move fired")

        self.route("POST", "/api/game/g1/move", _no_move)
        out = await mcp_server.submit_move("g1", "h2e2", SEAT_TOKEN_FIXTURE, side="red")
        self.assertFalse(out["ok"])
        self.assertEqual(out["error_class"], "state")
        self.assertEqual(out["status"], phase)
        self.assertIn("disabled", out["error"])
        self.assertIn(want_hint, out["hint"])
        self.assertFalse(any(
            req.method == "POST" and req.url.path == "/api/game/g1/move"
            for req in self.requests
        ))

    async def test_submit_move_disabled_when_paused(self):
        # 复盘/暂停阶段：走子工具禁用，不发出 POST /move
        await self._assert_move_disabled_in_phase("paused", "wait_my_turn")

    async def test_submit_move_disabled_when_waiting(self):
        # 预盘/开局前：走子工具禁用
        await self._assert_move_disabled_in_phase("waiting", "Pre-game")

    async def test_submit_move_disabled_when_finished(self):
        # 终局复盘：走子工具禁用，指引改用 get_result
        await self._assert_move_disabled_in_phase("finished", "get_result")

    async def test_submit_move_disabled_when_interrupted(self):
        await self._assert_move_disabled_in_phase("interrupted", "wait_my_turn")

    async def test_submit_move_illegal_returns_retry_info(self):
        self.route("GET", "/api/game/g1/state", lambda req: _json_response(_state()))
        self.route("POST", "/api/game/g1/move", lambda req: _json_response({
            "detail": {
                "error": "Illegal move: 'zzzz'",
                "error_class": "rules",
                "legal_moves": ["h2e2", "b2e2"],
                "legal_count": 2,
                "board": "board-text",
                "fen": DEFAULT_FEN,
            }
        }, 400))
        out = await mcp_server.submit_move("g1", "zzzz", SEAT_TOKEN_FIXTURE, side="red")
        self.assertFalse(out["ok"])
        self.assertIn("Illegal move", out["error"])
        self.assertEqual(out["error_class"], "rules")
        self.assertEqual(out["legal_moves"], ["h2e2", "b2e2"])
        self.assertTrue(out["board"])


class WaitMyTurnTests(McpToolTests):
    async def test_wait_my_turn_my_turn(self):
        self.route(
            "POST",
            "/api/game/g1/wait-turn",
            lambda req: _json_response({
                "ok": True,
                "game_id": "g1",
                "waiting_done_reason": "my_turn",
                "status": "playing",
                "turn": "red",
                "fen": DEFAULT_FEN,
                "board": "board-text",
                "move_count": 0,
                "last_move": None,
            }),
        )
        out = await mcp_server.wait_my_turn("g1", "red", SEAT_TOKEN_FIXTURE, timeout_sec=10)
        self.assertTrue(out["ok"])
        self.assertEqual(out["waiting_done_reason"], "my_turn")
        self.assertIn("Your turn", out["hint"])
        body = self._request_body("POST", "/api/game/g1/wait-turn")
        self.assertEqual(body["side"], "red")
        self.assertEqual(body["timeout_sec"], 10)
        req = self._request("POST", "/api/game/g1/wait-turn")
        self.assertEqual(req.headers.get("x-xiangqi-seat-token"), SEAT_TOKEN_FIXTURE)

    async def test_wait_my_turn_paused_hints_move_disabled(self):
        self.route(
            "POST",
            "/api/game/g1/wait-turn",
            lambda req: _json_response({
                "ok": True,
                "game_id": "g1",
                "waiting_done_reason": "paused",
                "status": "paused",
                "turn": "red",
                "fen": DEFAULT_FEN,
                "board": "board-text",
                "move_count": 0,
                "last_move": None,
            }),
        )
        out = await mcp_server.wait_my_turn("g1", "red", SEAT_TOKEN_FIXTURE, timeout_sec=10)
        self.assertTrue(out["ok"])
        self.assertEqual(out["waiting_done_reason"], "paused")
        self.assertIn("submit_move is disabled", out["hint"])
        self.assertIn("wait_my_turn", out["hint"])

    async def test_wait_my_turn_finished_hints_move_disabled(self):
        self.route(
            "POST",
            "/api/game/g1/wait-turn",
            lambda req: _json_response({
                "ok": True,
                "game_id": "g1",
                "waiting_done_reason": "finished",
                "status": "finished",
                "turn": "red",
                "winner": "black",
                "fen": DEFAULT_FEN,
                "board": "board-text",
                "move_count": 30,
                "last_move": None,
            }),
        )
        out = await mcp_server.wait_my_turn("g1", "red", SEAT_TOKEN_FIXTURE, timeout_sec=10)
        self.assertEqual(out["waiting_done_reason"], "finished")
        self.assertIn("submit_move is disabled", out["hint"])
        self.assertIn("get_result", out["hint"])

    async def test_submit_move_wait_opponent(self):
        calls = {"n": 0}

        def state_handler(_req):
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response(_state(move_count=0, move_history=[]))
            return _json_response(_state(
                move_count=1,
                turn="black",
                move_history=[{
                    "number": 1, "side": "red", "move": "h2e2",
                    "move_zh": "炮二平五", "captured": None,
                }],
            ))

        self.route("GET", "/api/game/g1/state", state_handler)
        self.route("POST", "/api/game/g1/move", lambda req: _json_response({
            "ok": True, "accepted": "h2e2", "turn": "red", "pending": 1,
        }))
        self.route(
            "POST",
            "/api/game/g1/wait-turn",
            lambda req: _json_response({
                "ok": True,
                "game_id": "g1",
                "waiting_done_reason": "my_turn",
                "status": "playing",
                "turn": "red",
                "fen": DEFAULT_FEN,
                "move_count": 2,
                "last_move": {
                    "number": 2, "side": "black", "move": "h7e7",
                },
            }),
        )
        out = await mcp_server.submit_move(
            "g1", "h2e2", SEAT_TOKEN_FIXTURE, side="red", wait_opponent=True, wait_timeout_sec=30
        )
        self.assertTrue(out["ok"])
        self.assertTrue(out["applied"])
        self.assertEqual(out["wait_after"]["waiting_done_reason"], "my_turn")
        self.assertIn("your turn again", out["hint"])


class DuelLifecycleTests(McpToolTests):
    async def test_create_duel_auto_claims_and_returns_token(self):
        self.route("POST", "/api/game/create", lambda req: _json_response({
            "game_id": "g9", "fen": DEFAULT_FEN,
        }))
        self.route("POST", "/api/game/g9/start", lambda req: _json_response({
            "status": "started", "local": False,
        }))
        self.route("POST", "/api/game/g9/claim-seat", lambda req: _json_response({
            "ok": True, "game_id": "g9", "side": "red", "seat_token": SEAT_TOKEN_FIXTURE,
        }))
        out = await mcp_server.create_duel(
            {"type": "external", "name": "外部挑战者"},
            {"type": "preset", "preset": "glm53"},
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["game_id"], "g9")
        self.assertEqual(out["seat_token"], SEAT_TOKEN_FIXTURE)
        self.assertEqual(out["my_side"], "red")
        body = self._request_body("POST", "/api/game/create")
        self.assertEqual(body["red"], {"type": "external", "name": "外部挑战者"})
        self.assertEqual(body["black"], {"type": "llm", "preset": "glm53"})
        claim = self._request_body("POST", "/api/game/g9/claim-seat")
        self.assertEqual(claim["side"], "red")

    async def test_challenge_preset_black_side_swaps(self):
        self.route("GET", "/api/presets", lambda req: _json_response({
            "presets": [{"name": "glm53", "type": "llm", "display_name": "GLM-5.3"}],
        }))
        self.route("POST", "/api/game/create", lambda req: _json_response({
            "game_id": "g8", "fen": DEFAULT_FEN,
        }))
        self.route("POST", "/api/game/g8/start", lambda req: _json_response({"status": "started"}))
        self.route("POST", "/api/game/g8/claim-seat", lambda req: _json_response({
            "ok": True, "side": "black", "seat_token": "black-tok",
        }))
        out = await mcp_server.challenge_preset("glm53", my_side="black", my_name="老外")
        self.assertTrue(out["ok"])
        self.assertEqual(out["seat_token"], "black-tok")
        self.assertEqual(out["my_side"], "black")
        body = self._request_body("POST", "/api/game/create")
        self.assertEqual(body["red"], {"type": "llm", "preset": "glm53"})
        self.assertEqual(body["black"], {"type": "external", "name": "老外"})

    async def test_challenge_preset_rejects_external_seat(self):
        self.route("GET", "/api/presets", lambda req: _json_response({
            "presets": [{"name": "ext_seat", "type": "external"}],
        }))
        out = await mcp_server.challenge_preset("ext_seat")
        self.assertFalse(out["ok"])
        self.assertIn("type=llm", out["error"])
        self.assertFalse(any(r.url.path == "/api/game/create" for r in self.requests))

    async def test_list_games_marks_open_seats(self):
        self.route("GET", "/api/games", lambda req: _json_response({
            "games": [{"id": "g1", "status": "playing", "move_count": 3}],
        }))
        self.route("GET", "/api/game/g1/state", lambda req: _json_response(_state(
            turn="red",
            sides={
                "red": {"type": "external", "name": "外部", "claimed": False},
                "black": {"type": "llm", "name": "GLM", "claimed": None},
            },
        )))
        out = await mcp_server.list_games()
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["open_seats"]), 1)
        seat = out["open_seats"][0]
        self.assertEqual(seat["game_id"], "g1")
        self.assertEqual(seat["side"], "red")
        self.assertTrue(seat["joinable"])
        self.assertTrue(seat["waiting_for_side"])
        self.assertEqual(len(out["joinable_seats"]), 1)

    async def test_resign_tool(self):
        self.route("POST", "/api/game/g1/resign", lambda req: _json_response({
            "status": "finished", "winner": "black", "reason": "resignation",
        }))
        out = await mcp_server.resign("g1", SEAT_TOKEN_FIXTURE, side="red")
        self.assertTrue(out["ok"])
        self.assertEqual(out["winner"], "black")
        body = self._request_body("POST", "/api/game/g1/resign")
        self.assertEqual(body, {"side": "red"})
        req = self._request("POST", "/api/game/g1/resign")
        self.assertEqual(req.headers.get("x-xiangqi-seat-token"), SEAT_TOKEN_FIXTURE)

    async def test_get_result_includes_moves_and_note(self):
        self.route("GET", "/api/game/g1/state", lambda req: _json_response(_state(
            status="finished", winner="black", reason="resignation",
            move_count=1,
            move_history=[{
                "number": 1, "side": "red", "move": "h2e2",
                "move_zh": "炮二平五", "captured": None, "note": "当头炮",
            }],
        )))
        out = await mcp_server.get_result("g1", SEAT_TOKEN_FIXTURE)
        self.assertTrue(out["ok"])
        self.assertEqual(out["winner"], "black")
        self.assertEqual(out["moves"][0]["move_zh"], "炮二平五")
        self.assertEqual(out["moves"][0]["note"], "当头炮")

    async def test_list_presets_passthrough(self):
        self.route("GET", "/api/presets", lambda req: _json_response({
            "presets": [
                {"name": "glm53", "type": "llm", "display_name": "GLM-5.3"},
                {"name": "ext_seat", "type": "external", "display_name": "外部挑战者A"},
            ],
        }))
        out = await mcp_server.list_presets()
        self.assertTrue(out["ok"])
        self.assertEqual(out["presets"][1]["type"], "external")

    async def test_preview_tool(self):
        self.route("POST", "/api/game/g1/preview", lambda req: _json_response({
            "ok": True, "path": ["h2e2"], "text": "preview ok", "fen_work": "x",
        }))
        out = await mcp_server.preview("g1", SEAT_TOKEN_FIXTURE, "h2e2")
        self.assertTrue(out["ok"])
        self.assertEqual(out["path"], ["h2e2"])
        req = self._request("POST", "/api/game/g1/preview")
        self.assertEqual(req.headers.get("x-xiangqi-mcp-tool"), "preview")

    async def test_get_player_brief(self):
        prev = os.environ.get("XIANGQI_LANG")
        os.environ.pop("XIANGQI_LANG", None)
        try:
            out = await mcp_server.get_player_brief()
            self.assertTrue(out["ok"])
            self.assertEqual(out["contract_version"], "mcp-agent-v1")
            self.assertEqual(out["play_mode"], "llm_stepwise")
            self.assertIn("must", out["rules"])
            self.assertIn("forbid", out["rules"])
            self.assertIn("LLM-plays-the-pieces", out["player_brief"])
        finally:
            if prev is None:
                os.environ.pop("XIANGQI_LANG", None)
            else:
                os.environ["XIANGQI_LANG"] = prev

    async def test_claim_seat_attaches_contract(self):
        self.route("POST", "/api/game/g1/claim-seat", lambda req: _json_response({
            "ok": True, "game_id": "g1", "side": "red", "seat_token": SEAT_TOKEN_FIXTURE,
        }))
        out = await mcp_server.claim_seat("g1", "red", name="赤练")
        self.assertTrue(out["ok"])
        self.assertEqual(out["seat_token"], SEAT_TOKEN_FIXTURE)
        self.assertEqual(out["contract_version"], "mcp-agent-v1")
        self.assertTrue(out["player_brief"])
        self.assertIn("VAL", " ".join(out["rules"]["forbid"]))

    async def test_create_duel_auto_claim_false_skips_tokens(self):
        self.route("POST", "/api/game/create", lambda req: _json_response({
            "game_id": "g7", "fen": DEFAULT_FEN,
        }))
        self.route("POST", "/api/game/g7/start", lambda req: _json_response({
            "status": "started", "local": False,
        }))
        out = await mcp_server.create_duel(
            {"type": "external", "name": "赤练"},
            {"type": "external", "name": "玄戈"},
            auto_claim=False,
        )
        self.assertTrue(out["ok"])
        self.assertFalse(out["auto_claim"])
        self.assertNotIn("red_seat_token", out)
        self.assertNotIn("black_seat_token", out)
        self.assertEqual(len(out["open_seats"]), 2)
        self.assertTrue(all(s["joinable"] for s in out["open_seats"]))
        self.assertFalse(any(r.url.path.endswith("/claim-seat") for r in self.requests))

    async def test_create_duel_dual_external_defaults_to_no_auto_claim(self):
        self.route("POST", "/api/game/create", lambda req: _json_response({
            "game_id": "g6", "fen": DEFAULT_FEN,
        }))
        self.route("POST", "/api/game/g6/start", lambda req: _json_response({
            "status": "started", "local": False,
        }))
        out = await mcp_server.create_duel(
            {"type": "external", "name": "赤练"},
            {"type": "external", "name": "玄戈"},
        )
        self.assertTrue(out["ok"])
        self.assertFalse(out["auto_claim"])
        self.assertEqual(out["join_playbook"], "referee_open__agents_claim")
        self.assertNotIn("red_seat_token", out)
        self.assertFalse(any(r.url.path.endswith("/claim-seat") for r in self.requests))

    async def test_wait_my_turn_includes_rules_reminder(self):
        self.route("POST", "/api/game/g1/wait-turn", lambda req: _json_response({
            "ok": True,
            "game_id": "g1",
            "waiting_done_reason": "my_turn",
            "status": "playing",
            "turn": "red",
            "fen": DEFAULT_FEN,
            "move_count": 0,
        }))
        out = await mcp_server.wait_my_turn("g1", "red", SEAT_TOKEN_FIXTURE, timeout_sec=5)
        self.assertTrue(out["ok"])
        self.assertIn("rules_reminder", out)
        self.assertIn("submit_move", out["rules_reminder"])


if __name__ == "__main__":
    unittest.main()
