"""Each LLM ply drops older checkpoints, not only after compress."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.player import LangGraphPlayer
from xiangqi import Board


class FakeGraph:
    async def ainvoke(self, *_args, **_kwargs):
        return {"move": "h2e2", "error": None}


class FakeBuilder:
    def compile(self, checkpointer=None):
        del checkpointer
        return FakeGraph()


class RequestMovePruneTests(unittest.IsolatedAsyncioTestCase):
    async def test_prunes_latest_even_when_turn_did_not_compress(self):
        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="g1",
            side_name="red",
        )
        prune = AsyncMock()
        with patch("agent.player.mem.get_checkpointer", AsyncMock(return_value=object())):
            with patch("agent.player.mem.prune_thread_to_latest", prune):
                with patch.object(player, "_build_graph", return_value=FakeBuilder()):
                    events = []
                    async for event in player.request_move(Board(), "w"):
                        events.append(event)
        self.assertFalse(player._archived_this_turn)
        prune.assert_awaited_once_with(player.thread_id)
        self.assertEqual(events[-1], {"type": "move", "move": "h2e2"})

    async def test_hanging_prune_does_not_block_move(self):
        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="g1",
            side_name="red",
        )

        async def hang(_thread_id: str) -> None:
            await asyncio.sleep(60)

        real_wait_for = asyncio.wait_for

        async def short_wait(aw, timeout=None):
            del timeout
            return await real_wait_for(aw, 0.05)

        with patch("agent.player.mem.get_checkpointer", AsyncMock(return_value=object())):
            with patch("agent.player.mem.prune_thread_to_latest", hang):
                with patch.object(player, "_build_graph", return_value=FakeBuilder()):
                    with patch("agent.player.asyncio.wait_for", new=short_wait):
                        events = []
                        async for event in player.request_move(Board(), "w"):
                            events.append(event)
        self.assertEqual(events[-1], {"type": "move", "move": "h2e2"})

    async def test_cancelled_turn_skips_prune_and_propagates(self):
        """Pause/reset/shutdown must abort, not uncancel and wait on prune."""
        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="g1",
            side_name="red",
        )
        started = asyncio.Event()

        class HangingGraph:
            async def ainvoke(self, *_args, **_kwargs):
                started.set()
                await asyncio.sleep(60)
                return {"move": "h2e2", "error": None}

        class HangingBuilder:
            def compile(self, checkpointer=None):
                del checkpointer
                return HangingGraph()

        prune = AsyncMock()
        with patch("agent.player.mem.get_checkpointer", AsyncMock(return_value=object())):
            with patch("agent.player.mem.prune_thread_to_latest", prune):
                with patch.object(player, "_build_graph", return_value=HangingBuilder()):

                    async def consume():
                        async for _event in player.request_move(Board(), "w"):
                            pass

                    task = asyncio.create_task(consume())
                    await asyncio.wait_for(started.wait(), timeout=1)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=1)
        prune.assert_not_awaited()

    async def test_aclose_without_cancel_still_prunes(self):
        """Stopping the generator is not a task cancel; do not leak CancelledError."""
        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            game_id="g1",
            side_name="red",
        )
        started = asyncio.Event()

        class HangingGraph:
            async def ainvoke(self, *_args, **_kwargs):
                started.set()
                q = player._live_queue
                if q is not None:
                    await q.put(("event", {"type": "thought", "thinking": "x"}))
                await asyncio.sleep(60)
                return {"move": "h2e2", "error": None}

        class HangingBuilder:
            def compile(self, checkpointer=None):
                del checkpointer
                return HangingGraph()

        prune = AsyncMock()
        with patch("agent.player.mem.get_checkpointer", AsyncMock(return_value=object())):
            with patch("agent.player.mem.prune_thread_to_latest", prune):
                with patch.object(player, "_build_graph", return_value=HangingBuilder()):
                    agen = player.request_move(Board(), "w")
                    event = await asyncio.wait_for(agen.__anext__(), timeout=1)
                    self.assertEqual(event.get("type"), "thought")
                    await asyncio.wait_for(started.wait(), timeout=1)
                    await agen.aclose()
        prune.assert_awaited_once_with(player.thread_id)


class CompressBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_does_not_hard_trim_to_leave_room_for_completion(self):
        """max_completion_tokens must not shrink the compress budget to 2048."""
        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            context_window=500000,
            max_completion_tokens=500000,
            compress_threshold_ratio=0.9,
        )
        blob = "车马炮" * 4000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": blob},
            {"role": "assistant", "content": "ok"},
        ]
        out = await player._maybe_compress_openai_messages(msgs)
        self.assertEqual(out, msgs)

    async def test_max_input_tokens_drives_compress_budget(self):
        """Compress uses max_input_tokens, not the full context_window."""
        from unittest.mock import AsyncMock, patch

        player = LangGraphPlayer(
            "http://example.invalid/v1",
            "k",
            "m",
            context_window=262144,
            max_input_tokens=10000,
            compress_threshold_ratio=0.9,
        )
        self.assertEqual(player.max_input_tokens, 10000)
        blob = "车马炮兵" * 8000  # well above 9k tokens heuristically
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": blob},
        ]
        compressed = [{"role": "system", "content": "sys"}, {"role": "user", "content": "summary"}]
        with patch(
            "agent.player.should_compress", return_value=True
        ) as should:
            with patch(
                "agent.player.compress_messages",
                new=AsyncMock(return_value=compressed),
            ) as compress:
                out = await player._maybe_compress_openai_messages(msgs)
        self.assertEqual(out, compressed)
        self.assertEqual(should.call_args.args[1], 10000)
        self.assertEqual(compress.await_args.kwargs["context_window"], 10000)
