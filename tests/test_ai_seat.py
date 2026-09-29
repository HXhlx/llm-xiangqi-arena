"""AI seat: LiteLLM env/preset resolution, game_loop, and the finish rule."""

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import server

DEFAULT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
_LITELLM_KEYS = ("LITELLM_API_BASE", "LITELLM_API_KEY", "LITELLM_MODEL")


class _OneMovePlayer:
    """Stand-in for LangGraphPlayer. Records the endpoint it was given."""

    seen: list[dict] = []

    def __init__(self, *args, **kwargs):
        self.last_turn_trace = None
        self.kwargs = kwargs
        _OneMovePlayer.seen.append(kwargs)

    async def request_move(self, board, side):
        yield {"type": "move", "move": "h2e2"}


def _preset(**over) -> dict:
    base = {
        "name": "example-model",
        "display_name": "Example seat",
        "type": "llm",
        "api_base": "https://other.example/v1",
        "api_key": "preset-key",
        "model": "preset-model",
        "context_window": 256000,
        "prompt_name": "en",
    }
    base.update(over)
    return base


async def _wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met within timeout")


async def _stop(task: asyncio.Task | None) -> None:
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class AiSeatResolutionTests(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _LITELLM_KEYS}
        self._saved_allowed = os.environ.pop("LITELLM_ALLOWED_MODELS", None)
        for key in _LITELLM_KEYS:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._saved_allowed is None:
            os.environ.pop("LITELLM_ALLOWED_MODELS", None)
        else:
            os.environ["LITELLM_ALLOWED_MODELS"] = self._saved_allowed

    def _set_env(self):
        os.environ["LITELLM_API_BASE"] = "https://llm.example/v1/"
        os.environ["LITELLM_API_KEY"] = "  sk-test  "
        os.environ["LITELLM_MODEL"] = "example-model"

    def test_env_resolves_ai_to_llm(self):
        self._set_env()
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000, name="Red AI")
        )
        self.assertEqual(cfg.type, "llm")
        self.assertEqual(cfg.api_base, "https://llm.example/v1")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertEqual(cfg.model, "example-model")
        self.assertEqual(cfg.name, "Red AI")
        self.assertEqual(cfg.context_window, 256000)

    def test_preset_model_overrides_env_and_ignores_preset_endpoint(self):
        self._set_env()
        with patch.object(server, "load_model_presets", return_value=[_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="ai", preset="example-model")
            )
        self.assertEqual(cfg.type, "llm")
        self.assertEqual(cfg.model, "preset-model")
        self.assertEqual(cfg.api_base, "https://llm.example/v1")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertNotEqual(cfg.api_key, "preset-key")
        self.assertEqual(cfg.name, "Example seat")
        self.assertEqual(cfg.preset, "example-model")

    def test_placeholder_key_is_rejected_as_missing(self):
        os.environ["LITELLM_API_BASE"] = "https://llm.example/v1"
        os.environ["LITELLM_API_KEY"] = "sk-xxxxxxxx"
        os.environ["LITELLM_MODEL"] = "example-model"
        with self.assertRaises(server.HTTPException) as ctx:
            server.resolve_preset(
                server.PlayerConfig(type="ai", context_window=256000)
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("LITELLM_API_KEY", str(ctx.exception.detail))

    def test_env_example_does_not_set_litellm_values(self):
        with open(os.path.join(PROJECT_ROOT, ".env.example"), encoding="utf-8") as fh:
            text = fh.read()
        for line in text.splitlines():
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith("LITELLM_"),
                stripped,
            )
        self.assertIn("# LITELLM_API_KEY=sk-xxxxxxxx", text)

    def test_missing_litellm_config_is_400(self):
        with self.assertRaises(server.HTTPException) as ctx:
            server.resolve_preset(server.PlayerConfig(type="ai"))
        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail)
        for name in _LITELLM_KEYS:
            self.assertIn(name, detail)

    def test_snapshot_rehydrate_reapplies_litellm_not_preset_endpoint(self):
        self._set_env()
        with patch.object(server, "load_model_presets", return_value=[_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="ai", preset="example-model")
            )
        self.assertTrue(cfg.litellm)
        raw = server._snapshot_player_config(cfg)
        self.assertIsNone(raw["api_key"])
        self.assertNotIn("sk-test", str(raw))
        os.environ["LITELLM_API_KEY"] = "sk-restored"
        with patch.object(server, "load_model_presets", return_value=[_preset()]):
            revived = server._rehydrate_snapshot_player(raw)
        self.assertEqual(revived.type, "llm")
        self.assertTrue(revived.litellm)
        self.assertEqual(revived.api_base, "https://llm.example/v1")
        self.assertEqual(revived.api_key, "sk-restored")
        self.assertEqual(revived.model, "preset-model")
        self.assertNotEqual(revived.api_base, "https://other.example/v1")

    def test_plain_llm_snapshot_keeps_preset_endpoint(self):
        self._set_env()
        with patch.object(server, "load_model_presets", return_value=[_preset()]):
            cfg = server.resolve_preset(
                server.PlayerConfig(type="llm", preset="example-model")
            )
            self.assertFalse(cfg.litellm)
            raw = server._snapshot_player_config(cfg)
            revived = server._rehydrate_snapshot_player(raw)
        self.assertEqual(revived.api_base, "https://other.example/v1")
        self.assertEqual(revived.api_key, "preset-key")
        self.assertFalse(revived.litellm)

    def test_request_endpoint_cannot_override_litellm_env(self):
        self._set_env()
        cfg = server.resolve_preset(
            server.PlayerConfig(
                type="ai",
                api_base="https://evil.example/v1",
                api_key="sk-attacker",
                model="example-model",
                context_window=256000,
            )
        )
        self.assertEqual(cfg.api_base, "https://llm.example/v1")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertNotEqual(cfg.api_key, "sk-attacker")
        self.assertNotIn("evil.example", cfg.api_base or "")

    def test_allowed_models_rejects_names_outside_the_list(self):
        self._set_env()
        os.environ["LITELLM_ALLOWED_MODELS"] = " example-model, other-model "
        cfg = server.resolve_preset(
            server.PlayerConfig(
                type="ai", model="other-model", context_window=256000
            )
        )
        self.assertEqual(cfg.model, "other-model")
        self.assertEqual(cfg.api_key, "sk-test")
        self.assertEqual(cfg.api_base, "https://llm.example/v1")
        with self.assertRaises(server.HTTPException) as ctx:
            server.resolve_preset(
                server.PlayerConfig(
                    type="ai",
                    api_base="https://evil.example/v1",
                    api_key="sk-attacker",
                    model="spend-anything",
                    context_window=256000,
                )
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("LITELLM_ALLOWED_MODELS", str(ctx.exception.detail))

    def test_unset_allowlist_still_ignores_request_endpoint(self):
        self._set_env()
        os.environ.pop("LITELLM_ALLOWED_MODELS", None)
        cfg = server.resolve_preset(
            server.PlayerConfig(
                type="ai",
                api_base="https://evil.example/v1",
                api_key="sk-attacker",
                model="any-model",
                context_window=256000,
            )
        )
        self.assertEqual(cfg.model, "any-model")
        self.assertEqual(cfg.api_base, "https://llm.example/v1")
        self.assertEqual(cfg.api_key, "sk-test")

    def test_snapshot_rehydrate_fails_when_litellm_env_missing(self):
        self._set_env()
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000)
        )
        raw = server._snapshot_player_config(cfg)
        self.assertIsNone(raw["api_key"])
        self.assertEqual(raw["api_base"], "https://llm.example/v1")
        for key in _LITELLM_KEYS:
            os.environ.pop(key, None)
        with self.assertRaises(ValueError) as ctx:
            server._rehydrate_snapshot_player(raw)
        message = str(ctx.exception)
        self.assertIn("Cannot restore AI seat", message)
        self.assertIn("LITELLM_API_KEY", message)
        self.assertIn("API key", message)

    def test_external_preset_cannot_back_an_ai_seat(self):
        self._set_env()
        external = {"name": "ext_seat", "type": "external", "display_name": "External"}
        with patch.object(server, "load_model_presets", return_value=[external]):
            with self.assertRaises(server.HTTPException) as ctx:
                server.resolve_preset(server.PlayerConfig(type="ai", preset="ext_seat"))
        self.assertEqual(ctx.exception.status_code, 400)


class AiSeatLoopTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _LITELLM_KEYS}
        self._saved_allowed = os.environ.pop("LITELLM_ALLOWED_MODELS", None)
        os.environ["LITELLM_API_BASE"] = "https://llm.example/v1"
        os.environ["LITELLM_API_KEY"] = "sk-test"
        os.environ["LITELLM_MODEL"] = "example-model"
        _OneMovePlayer.seen = []

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._saved_allowed is None:
            os.environ.pop("LITELLM_ALLOWED_MODELS", None)
        else:
            os.environ["LITELLM_ALLOWED_MODELS"] = self._saved_allowed

    async def test_game_loop_drives_created_ai_seat(self):
        req = server.CreateGameRequest(
            red=server.PlayerConfig(type="ai", context_window=256000),
            black=server.PlayerConfig(type="random"),
        )
        with patch.object(server, "LangGraphPlayer", _OneMovePlayer), patch.object(
            server, "_persist_llm_turn", new=AsyncMock()
        ):
            created = await server.create_game(req)
            gid = created["game_id"]
            self.addCleanup(lambda: server.games.pop(gid, None))
            game = server.games[gid]
            self.assertEqual(game.red_config.type, "llm")
            self.assertEqual(game.red_config.model, "example-model")
            self.assertEqual(game.red_config.api_base, "https://llm.example/v1")
            self.assertFalse(server._is_local_game(game))
            started = await server.start_game(gid)
            self.assertFalse(started["local"])
            try:
                await _wait_for(lambda: len(game.move_history) >= 1)
            finally:
                await _stop(game.task)
        self.assertEqual(game.move_history[0]["move"], "h2e2")
        self.assertEqual(game.move_history[0]["side"], "red")
        self.assertNotEqual(game.status, "interrupted")
        self.assertTrue(_OneMovePlayer.seen)
        self.assertEqual(_OneMovePlayer.seen[0]["api_base"], "https://llm.example/v1")
        self.assertEqual(_OneMovePlayer.seen[0]["api_key"], "sk-test")
        self.assertEqual(_OneMovePlayer.seen[0]["model"], "example-model")

    async def test_game_loop_drives_raw_ai_type(self):
        game = server.GameSession(
            "rawai",
            DEFAULT_FEN,
            server.PlayerConfig(
                type="ai",
                api_base="https://llm.example/v1",
                api_key="sk-test",
                model="example-model",
                context_window=256000,
            ),
            server.PlayerConfig(type="random"),
        )
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        with patch.object(server, "LangGraphPlayer", _OneMovePlayer), patch.object(
            server, "_persist_llm_turn", new=AsyncMock()
        ):
            task = asyncio.create_task(server.game_loop(game))
            try:
                await _wait_for(lambda: len(game.move_history) >= 1)
            finally:
                await _stop(task)
        self.assertEqual(game.move_history[0]["side"], "red")
        self.assertEqual(game.move_history[0]["move"], "h2e2")
        self.assertNotEqual(game.status, "interrupted")

    async def test_finish_rejects_resolved_ai_game(self):
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000)
        )
        game = server.GameSession(
            "aires",
            DEFAULT_FEN,
            cfg,
            server.PlayerConfig(type="random"),
        )
        game.status = "playing"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        with self.assertRaises(server.HTTPException) as ctx:
            await server.finish_game(
                game.id,
                server.FinishGameRequest(moves=["h2e2"], winner="red", reason="forged"),
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("llm", str(ctx.exception.detail))
        self.assertEqual(game.move_history, [])
        self.assertEqual(game.status, "playing")

    async def test_game_loop_ignores_malicious_request_endpoint(self):
        _OneMovePlayer.seen = []
        req = server.CreateGameRequest(
            red=server.PlayerConfig(
                type="ai",
                api_base="https://evil.example/v1",
                api_key="sk-attacker",
                context_window=256000,
            ),
            black=server.PlayerConfig(type="random"),
        )
        with patch.object(server, "LangGraphPlayer", _OneMovePlayer), patch.object(
            server, "_persist_llm_turn", new=AsyncMock()
        ):
            created = await server.create_game(req)
            gid = created["game_id"]
            self.addCleanup(lambda: server.games.pop(gid, None))
            game = server.games[gid]
            self.assertEqual(game.red_config.api_base, "https://llm.example/v1")
            self.assertEqual(game.red_config.api_key, "sk-test")
            await server.start_game(gid)
            try:
                await _wait_for(lambda: len(game.move_history) >= 1)
            finally:
                await _stop(game.task)
        self.assertEqual(_OneMovePlayer.seen[0]["api_base"], "https://llm.example/v1")
        self.assertEqual(_OneMovePlayer.seen[0]["api_key"], "sk-test")
        self.assertNotEqual(_OneMovePlayer.seen[0]["api_key"], "sk-attacker")

    async def test_restore_rejects_ai_seat_when_litellm_env_missing(self):
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000)
        )
        game = server.GameSession(
            "ai-nokey",
            DEFAULT_FEN,
            cfg,
            server.PlayerConfig(type="random"),
        )
        game.status = "interrupted"
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(server, "SNAPSHOT_DIR", tmp):
                server.write_game_snapshot(game, reason="interrupted")
                for key in _LITELLM_KEYS:
                    os.environ.pop(key, None)
                server.games.pop(game.id, None)
                with self.assertRaises(server.HTTPException) as ctx:
                    await server.restore_game(game.id)
        self.assertEqual(ctx.exception.status_code, 400)
        detail = str(ctx.exception.detail)
        self.assertIn("Cannot restore AI seat", detail)
        self.assertIn("LITELLM_API_KEY", detail)
        self.assertIn("API key", detail)
        self.assertNotIn(game.id, server.games)

    async def test_resume_and_start_reject_ai_seat_without_key(self):
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000)
        )
        game = server.GameSession(
            "ai-nokey-resume",
            DEFAULT_FEN,
            cfg.model_copy(update={"api_key": None}),
            server.PlayerConfig(type="random"),
        )
        game.status = "interrupted"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        for key in _LITELLM_KEYS:
            os.environ.pop(key, None)
        with self.assertRaises(server.HTTPException) as ctx:
            await server.resume_game(game.id)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("API key", str(ctx.exception.detail))
        self.assertEqual(game.status, "interrupted")
        self.assertIsNone(game.task)
        with self.assertRaises(server.HTTPException) as restored:
            await server.restore_game(game.id)
        self.assertEqual(restored.exception.status_code, 400)
        self.assertEqual(game.status, "interrupted")
        game.status = "waiting"
        game.red_config.api_key = None
        with self.assertRaises(server.HTTPException) as started:
            await server.start_game(game.id)
        self.assertEqual(started.exception.status_code, 400)
        self.assertEqual(game.status, "waiting")
        self.assertIsNone(game.task)

    async def test_resume_refills_ai_key_from_env_before_loop(self):
        cfg = server.resolve_preset(
            server.PlayerConfig(type="ai", context_window=256000)
        )
        game = server.GameSession(
            "ai-refill",
            DEFAULT_FEN,
            cfg.model_copy(update={"api_key": None}),
            server.PlayerConfig(type="random"),
        )
        game.status = "interrupted"
        server.games[game.id] = game
        self.addCleanup(lambda: server.games.pop(game.id, None))
        _OneMovePlayer.seen = []
        with patch.object(server, "LangGraphPlayer", _OneMovePlayer), patch.object(
            server, "_persist_llm_turn", new=AsyncMock()
        ):
            resumed = await server.resume_game(game.id)
            try:
                await _wait_for(lambda: len(game.move_history) >= 1)
            finally:
                await _stop(game.task)
        self.assertEqual(resumed["status"], "resumed")
        self.assertEqual(game.red_config.api_key, "sk-test")
        self.assertEqual(_OneMovePlayer.seen[0]["api_key"], "sk-test")


def _tool_stream(move: str = "h2e2") -> bytes:
    chunks = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "example-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "make_move", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "example-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": json.dumps({"move": move})},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "example-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    return (body + "data: [DONE]\n\n").encode()


def _openai_http_module():
    """HTTP stack bound by the installed OpenAI client module.

    openai 1.x and 2.x import ``httpx``. openai 3.x imports ``httpx2``.
    Both packages can be installed at once (langsmith/mcp pull in httpx2),
    so the mock follows the name bound on the client module. openai 3.x
    later assigns ``openai._client = None``, which hides that module from
    ``import openai._client``; the loaded module is read from ``sys.modules``.
    """
    import sys

    import openai  # noqa: F401  -- loads openai._client before the name is shadowed

    client_mod = sys.modules.get("openai._client")
    if client_mod is None:
        raise RuntimeError("openai client module was not loaded")
    for name in ("httpx2", "httpx"):
        mod = getattr(client_mod, name, None)
        if (
            mod is not None
            and hasattr(mod, "AsyncClient")
            and hasattr(mod, "MockTransport")
        ):
            return mod
    raise RuntimeError("openai client did not import an HTTP stack")


class AiSeatRealClientTests(unittest.IsolatedAsyncioTestCase):
    """Drive LangGraphPlayer through the real AsyncOpenAI client.

    The mock transport matches the HTTP package ``openai._client`` imported
    (``httpx`` before 3.x, ``httpx2`` from 3.x).
    """

    def setUp(self):
        self._saved = {key: os.environ.get(key) for key in _LITELLM_KEYS}
        self._saved_openai = os.environ.get("OPENAI_API_KEY")
        os.environ["LITELLM_API_BASE"] = "https://llm.example/v1/"
        os.environ["LITELLM_API_KEY"] = "sk-env"
        os.environ["LITELLM_MODEL"] = "example-model"
        os.environ["OPENAI_API_KEY"] = "sk-openai-fallback"

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._saved_openai is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = self._saved_openai

    async def test_real_client_sends_env_key_not_request_or_openai_key(self):
        http = _openai_http_module()
        from openai import AsyncOpenAI

        from agent.player import LangGraphPlayer

        cfg = server.resolve_preset(
            server.PlayerConfig(
                type="ai",
                api_base="https://evil.example/v1",
                api_key="sk-attacker",
                model="example-model",
                context_window=256000,
            )
        )
        seen: dict = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            return http.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_tool_stream(),
            )

        transport = http.MockTransport(handler)
        real_client = AsyncOpenAI

        def factory(*args, **kwargs):
            seen["ctor_api_key"] = kwargs.get("api_key")
            seen["ctor_base_url"] = kwargs.get("base_url")
            kwargs["http_client"] = http.AsyncClient(transport=transport)
            kwargs["max_retries"] = 0
            return real_client(*args, **kwargs)

        player = LangGraphPlayer(
            api_base=cfg.api_base,
            api_key=cfg.api_key,
            model=cfg.model,
            game_id="real-ai",
            side_name="red",
            context_window=256000,
            timeout=5,
        )
        with patch("agent.player.AsyncOpenAI", factory):
            content, tool_calls, finish, _reasoning = await player._stream_completion(
                [{"role": "user", "content": "your move"}]
            )
        self.assertEqual(content, "")
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(tool_calls[0]["name"], "make_move")
        self.assertIn("h2e2", tool_calls[0]["args"])
        self.assertEqual(seen["ctor_api_key"], "sk-env")
        self.assertEqual(seen["ctor_base_url"], "https://llm.example/v1")
        self.assertEqual(seen["url"], "https://llm.example/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer sk-env")
        self.assertNotIn("sk-attacker", seen["auth"])
        self.assertNotIn("sk-openai-fallback", seen["auth"])
        self.assertNotIn("evil.example", seen["url"])

    async def test_missing_key_never_constructs_async_openai(self):
        from agent.player import LangGraphPlayer, MissingApiKeyError

        constructed = {"n": 0}

        def factory(*_args, **_kwargs):
            constructed["n"] += 1
            raise AssertionError("AsyncOpenAI must not be constructed")

        player = LangGraphPlayer(
            api_base="https://llm.example/v1",
            api_key=None,
            model="example-model",
            game_id="nokey",
            side_name="red",
        )
        with patch("agent.player.AsyncOpenAI", factory):
            with self.assertRaises(MissingApiKeyError):
                await player._stream_completion(
                    [{"role": "user", "content": "your move"}]
                )
        self.assertEqual(constructed["n"], 0)


if __name__ == "__main__":
    unittest.main()
