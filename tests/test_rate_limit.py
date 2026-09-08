"""Shared RPM/TPM sliding-window limiter."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.rate_limit import (
    DEFAULT_MODEL_RPM,
    DEFAULT_MODEL_TPM,
    DEFAULT_SITE_RPM,
    DEFAULT_SITE_TPM,
    KEY_PREFIX,
    WINDOW_SEC,
    MemoryWindowStore,
    RateLimitConfig,
    RateLimiter,
    Reservation,
    acquire_outbound,
    load_rate_limit_config,
    parse_limit,
    reset_limiter,
    rpm_model_key,
    rpm_site_key,
    sanitize_preset,
    set_limiter,
    tpm_model_key,
    tpm_site_key,
)


class ParseAndConfigTests(unittest.TestCase):
    def test_defaults(self):
        cfg = load_rate_limit_config(env={}, yaml_data={})
        self.assertEqual(cfg.site_rpm, DEFAULT_SITE_RPM)
        self.assertEqual(cfg.site_tpm, DEFAULT_SITE_TPM)
        self.assertEqual(cfg.default_model_rpm, DEFAULT_MODEL_RPM)
        self.assertEqual(cfg.default_model_tpm, DEFAULT_MODEL_TPM)
        self.assertEqual(DEFAULT_SITE_RPM, 0)
        self.assertEqual(DEFAULT_MODEL_RPM, 300)
        self.assertEqual(DEFAULT_SITE_TPM, 0)
        self.assertEqual(DEFAULT_MODEL_TPM, 0)

    def test_env_overrides_yaml_site(self):
        cfg = load_rate_limit_config(
            env={"XIANGQI_RPM": "30", "XIANGQI_TPM": "1000", "XIANGQI_MODEL_RPM": "2"},
            yaml_data={"rate_limit": {"rpm": 90, "tpm": 9}},
        )
        self.assertEqual(cfg.site_rpm, 30)
        self.assertEqual(cfg.site_tpm, 1000)
        self.assertEqual(cfg.default_model_rpm, 2)

    def test_yaml_site_when_env_empty(self):
        cfg = load_rate_limit_config(
            env={},
            yaml_data={"rate_limit": {"rpm": 40, "tpm": 0}},
        )
        self.assertEqual(cfg.site_rpm, 40)
        self.assertEqual(cfg.site_tpm, 0)

    def test_per_model_yaml_overrides(self):
        cfg = load_rate_limit_config(
            env={},
            yaml_data={
                "models": [
                    {"name": "grok46", "rpm": 3, "tpm": 8000},
                    {"name": "glm"},
                ]
            },
        )
        self.assertEqual(cfg.model_rpm_for("grok46"), 3)
        self.assertEqual(cfg.model_tpm_for("grok46"), 8000)
        self.assertEqual(cfg.model_rpm_for("glm"), DEFAULT_MODEL_RPM)
        self.assertEqual(cfg.model_tpm_for("glm"), 0)

    def test_yaml_model_rpm_and_unlimited_site(self):
        cfg = load_rate_limit_config(
            env={},
            yaml_data={"rate_limit": {"rpm": 0, "model_rpm": 300, "tpm": 0}},
        )
        self.assertEqual(cfg.site_rpm, 0)
        self.assertEqual(cfg.default_model_rpm, 300)
        self.assertEqual(cfg.model_rpm_for("flash_preview"), 300)

    def test_parse_limit_zero_is_unlimited(self):
        self.assertEqual(parse_limit(0, 5), 0)
        self.assertEqual(parse_limit("0", 5), 0)
        self.assertEqual(parse_limit("", 5), 5)
        self.assertEqual(parse_limit("nope", 5), 5)

    def test_sanitize_and_keys(self):
        self.assertEqual(sanitize_preset("grok46"), "grok46")
        self.assertTrue(rpm_site_key().startswith(KEY_PREFIX))
        self.assertIn("grok46", rpm_model_key("grok46"))
        self.assertNotIn("xiangqi:lg:", rpm_model_key("grok46"))


class MemoryWindowTests(unittest.IsolatedAsyncioTestCase):
    async def test_sixth_model_rpm_waits(self):
        clock = [1_000.0]
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        limiter = RateLimiter(
            MemoryWindowStore(),
            RateLimitConfig(site_rpm=60, default_model_rpm=5),
            clock=lambda: clock[0],
        )
        with patch("asyncio.sleep", fake_sleep):
            for _ in range(5):
                await limiter.acquire("grok46")
            await limiter.acquire("grok46")
        self.assertTrue(sleeps)
        self.assertGreaterEqual(sleeps[0], 0.05)
        self.assertAlmostEqual(clock[0], 1_000.0 + WINDOW_SEC, delta=0.2)

    async def test_shared_store_counts_across_clients(self):
        store = MemoryWindowStore()
        cfg = RateLimitConfig(site_rpm=60, default_model_rpm=5)
        clock = lambda: 2_000.0
        first = RateLimiter(store, cfg, clock=clock)
        second = RateLimiter(store, cfg, clock=clock)
        for _ in range(3):
            await first.acquire("grok46")
        for _ in range(2):
            await second.acquire("grok46")
        used = window_weight(store, rpm_model_key("grok46"))
        self.assertEqual(used, 5)
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            raise asyncio_break()

        with patch("asyncio.sleep", fake_sleep):
            with self.assertRaises(asyncio_break):
                await first.acquire("grok46")
        self.assertTrue(sleeps)

    async def test_tpm_zero_skips_token_buckets(self):
        store = MemoryWindowStore()
        limiter = RateLimiter(
            store,
            RateLimitConfig(site_rpm=60, site_tpm=0, default_model_rpm=5, default_model_tpm=0),
            clock=lambda: 3_000.0,
        )
        await limiter.acquire("grok46", tokens=9999)
        await limiter.record_tokens("grok46", 4000)
        self.assertEqual(window_weight(store, tpm_site_key()), 0)
        self.assertEqual(window_weight(store, tpm_model_key("grok46")), 0)
        self.assertEqual(window_weight(store, rpm_model_key("grok46")), 1)

    async def test_tpm_records_when_enabled(self):
        store = MemoryWindowStore()
        limiter = RateLimiter(
            store,
            RateLimitConfig(site_rpm=60, site_tpm=10000, default_model_rpm=5, default_model_tpm=8000),
            clock=lambda: 4_000.0,
        )
        await limiter.acquire("flash_preview", tokens=100)
        await limiter.record_tokens("flash_preview", 50)
        self.assertEqual(window_weight(store, tpm_model_key("flash_preview")), 150)
        self.assertEqual(window_weight(store, tpm_site_key()), 150)

    async def test_site_rpm_blocks_before_model(self):
        store = MemoryWindowStore()
        limiter = RateLimiter(
            store,
            RateLimitConfig(site_rpm=2, default_model_rpm=5),
            clock=lambda: 5_000.0,
        )
        await limiter.acquire("a")
        await limiter.acquire("b")
        sleeps: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)
            raise asyncio_break()

        with patch("asyncio.sleep", fake_sleep):
            with self.assertRaises(asyncio_break):
                await limiter.acquire("c")
        self.assertTrue(sleeps)

    async def test_acquire_outbound_uses_injected_limiter(self):
        store = MemoryWindowStore()
        limiter = RateLimiter(
            store,
            RateLimitConfig(site_rpm=60, default_model_rpm=5),
            clock=lambda: 6_000.0,
        )
        set_limiter(limiter)
        self.addCleanup(reset_limiter)
        await acquire_outbound("glm", tokens=10)
        self.assertEqual(window_weight(store, rpm_model_key("glm")), 1)
        self.assertEqual(window_weight(store, rpm_site_key()), 1)

    async def test_site_rpm_zero_skips_site_bucket(self):
        store = MemoryWindowStore()
        limiter = RateLimiter(
            store,
            RateLimitConfig(site_rpm=0, default_model_rpm=300),
            clock=lambda: 7_000.0,
        )
        for _ in range(8):
            await limiter.acquire("flash_preview")
        self.assertEqual(window_weight(store, rpm_site_key()), 0)
        self.assertEqual(window_weight(store, rpm_model_key("flash_preview")), 8)


class asyncio_break(Exception):
    """Stop the wait loop once the first sleep is requested."""


def window_weight(store: MemoryWindowStore, key: str) -> int:
    return sum(entry.weight for entry in store.windows.get(key) or [])


class ReservationShapeTests(unittest.TestCase):
    def test_prefix_is_not_checkpoint_namespace(self):
        self.assertEqual(KEY_PREFIX, "xiangqi:rl:")
        item = Reservation(key=rpm_site_key(), limit=60, weight=1)
        self.assertTrue(item.key.startswith("xiangqi:rl:"))


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeChatCompletions:
    def __init__(self, content: str = ""):
        self.content = content

    async def create(self, **_kwargs):
        if self.content:
            choice = type("Choice", (), {})()
            choice.message = type("Msg", (), {"content": self.content})()
            resp = type("Resp", (), {})()
            resp.choices = [choice]
            return resp
        return _EmptyStream()


class _FakeAsyncOpenAI:
    def __init__(self, **_kwargs):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeChatCompletions()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class OutboundHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_completion_acquires(self):
        from agent.player import LangGraphPlayer

        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            preset="grok46",
            rpm=5,
            tpm=0,
        )
        seen: list[str] = []

        async def fake_acquire(preset, *, tokens=0, rpm=None, tpm=None):
            seen.append(preset)
            self.assertEqual(rpm, 5)

        with patch("agent.player.acquire_outbound", fake_acquire):
            with patch("agent.player.AsyncOpenAI", _FakeAsyncOpenAI):
                await player._stream_completion(
                    [{"role": "user", "content": "hi"}]
                )
        self.assertEqual(seen, ["grok46"])

    async def test_compress_messages_acquires(self):
        from agent.compress import compress_messages

        acquired: list[str] = []

        async def fake_acquire(preset, *, tokens=0, rpm=None, tpm=None):
            acquired.append(preset)

        async def fake_record(*_args, **_kwargs):
            return None

        client = _FakeAsyncOpenAI()
        client.chat.completions = _FakeChatCompletions(content="旧着已压缩")

        with patch("agent.compress.should_compress", return_value=True):
            with patch(
                "agent.compress._split_keep_recent",
                return_value=(
                    [{"role": "user", "content": "old"}],
                    [{"role": "user", "content": "new"}],
                ),
            ):
                with patch("agent.compress.acquire_outbound", fake_acquire):
                    with patch(
                        "agent.compress.record_outbound_tokens", fake_record
                    ):
                        with patch("agent.compress.AsyncOpenAI", lambda **k: client):
                            out = await compress_messages(
                                [
                                    {"role": "system", "content": "sys"},
                                    {"role": "user", "content": "old"},
                                    {"role": "user", "content": "new"},
                                ],
                                api_base="http://example.invalid/v1",
                                api_key="k",
                                model="m",
                                context_window=256000,
                                preset="grok46",
                            )
        self.assertEqual(acquired, ["grok46"])
        self.assertTrue(any("[对局历史摘要]" in (m.get("content") or "") for m in out))

    async def test_summarize_turn_acquires(self):
        from agent.summarize_turn import summarize_turn

        acquired: list[str] = []

        async def fake_acquire(preset, *, tokens=0, rpm=None, tpm=None):
            acquired.append(preset)

        client = _FakeAsyncOpenAI()
        client.chat.completions = _FakeChatCompletions(content="这波不亏")

        with patch("agent.summarize_turn.acquire_outbound", fake_acquire):
            with patch(
                "agent.summarize_turn.record_outbound_tokens",
                AsyncMock(),
            ):
                with patch("agent.summarize_turn.AsyncOpenAI", lambda **k: client):
                    result = await summarize_turn(
                        api_base="http://example.invalid/v1",
                        api_key="k",
                        model="m",
                        move="h2e2",
                        reasoning="把炮架到中路",
                        preset="glm",
                    )
        self.assertEqual(acquired, ["glm"])
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
