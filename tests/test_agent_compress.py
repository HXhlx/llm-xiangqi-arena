"""Unit tests for agent context compression helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.compress import estimate_text_tokens, estimate_tokens, hard_trim, should_compress
from agent.player import REASONING_EFFORTS, make_thread_id
from agent import memory as mem
from adapters import get_adapter
from adapters.default import DefaultAdapter
from adapters.spark import SparkAdapter


class TestCompressHelpers(unittest.TestCase):
    def test_estimate_tokens_positive(self):
        msgs = [{"role": "user", "content": "hello world " * 50}]
        self.assertGreater(estimate_tokens(msgs), 10)

    def test_estimate_text_tokens_empty_is_zero(self):
        self.assertEqual(estimate_text_tokens(""), 0)
        self.assertEqual(estimate_text_tokens("   "), 0)
        self.assertGreater(estimate_text_tokens("车进九路"), 0)

    def test_should_compress_threshold(self):
        msgs = [{"role": "user", "content": "hello world " * 5000}]
        tokens = estimate_tokens(msgs)
        self.assertTrue(should_compress(msgs, context_window=max(100, tokens // 2), threshold_ratio=0.9))
        self.assertFalse(should_compress(msgs, context_window=tokens * 10, threshold_ratio=0.9))

    def test_hard_trim_keeps_system(self):
        msgs = (
            [{"role": "system", "content": "sys"}]
            + [{"role": "user", "content": f"msg-{i} " * 200} for i in range(20)]
        )
        trimmed = hard_trim(msgs, context_window=2000, keep_ratio=0.3)
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertLess(len(trimmed), len(msgs))

    def test_thread_id(self):
        tid = make_thread_id("abc", "red")
        self.assertTrue(tid.endswith("abc:red"))
        self.assertIn(mem.redis_prefix(), tid)

    def test_reasoning_efforts_include_openai_set(self):
        for v in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            self.assertIn(v, REASONING_EFFORTS)


class TestAdapterReasoningEffort(unittest.TestCase):
    def test_default_effort(self):
        body = DefaultAdapter().extra_body(True, reasoning_effort="high")
        self.assertEqual(
            body,
            {"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
        )

    def test_spark_effort_merged(self):
        body = SparkAdapter().extra_body(True, reasoning_effort="medium")
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "medium")

    def test_openai_compat_no_effort(self):
        adapter = get_adapter("https://api.openai.com/v1", "gpt-4o")
        self.assertEqual(adapter.extra_body(False), {"thinking": {"type": "disabled"}})

    def test_minimax_drops_reasoning_effort(self):
        adapter = get_adapter("http://localhost:4000/v1", "minimax-m3")
        body = adapter.extra_body(True, reasoning_effort="max")
        self.assertEqual(body, {"thinking": {"type": "adaptive"}})


if __name__ == "__main__":
    unittest.main()
