"""Pikafish subprocess must be reaped even after the UCI protocol dies."""

from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pikafish_manager import PikafishEvaluator


class FakeStdin:
    def __init__(self):
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


class FakeProcess:
    def __init__(self):
        self.stdin = FakeStdin()
        self.waited = False

    async def wait(self):
        self.waited = True
        return 0

    def kill(self):
        self.waited = True


class PikafishShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_reaps_process_when_protocol_already_dead(self):
        engine = PikafishEvaluator(engine_path="/nonexistent/pikafish")
        proc = FakeProcess()
        engine._process = proc
        engine._alive = False
        await engine.shutdown()
        self.assertIn(b"quit\n", proc.stdin.writes)
        self.assertTrue(proc.waited)
        self.assertFalse(engine._alive)
        self.assertIsNone(engine._process)


if __name__ == "__main__":
    unittest.main()
