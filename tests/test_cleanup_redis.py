"""cleanup_redis prefers XIANGQI_API_BASE over the legacy name."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.cleanup_redis import resolve_api_base


class ResolveApiBaseTests(unittest.TestCase):
    def test_api_base_wins_over_legacy_name(self):
        env = {
            "XIANGQI_API_BASE": "http://127.0.0.1:8000/",
            "XIANGQI_BASE": "http://127.0.0.1:9",
        }
        self.assertEqual(resolve_api_base(env), "http://127.0.0.1:8000")

    def test_legacy_name_warns_and_is_still_accepted(self):
        buf = io.StringIO()
        with redirect_stderr(buf):
            base = resolve_api_base({"XIANGQI_BASE": "http://127.0.0.1:9/"})
        self.assertEqual(base, "http://127.0.0.1:9")
        self.assertIn("deprecated", buf.getvalue())
        self.assertIn("XIANGQI_API_BASE", buf.getvalue())

    def test_default_is_local_arena(self):
        self.assertEqual(resolve_api_base({}), "http://127.0.0.1:8000")
