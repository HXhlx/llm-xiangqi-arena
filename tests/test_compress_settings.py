"""Tests for compress_settings after context_window fallback removal.

The 5-level fallback chain (explicit -> env -> yaml -> tournament -> default)
was removed in PR-2. context_window now comes only from the model preset.
threshold_ratio still falls back: env -> yaml.compress.threshold_ratio -> 0.9.
max_input_tokens defaults to context_window when omitted.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.compress_settings import (
    DEFAULT_COMPRESS_THRESHOLD_RATIO,
    resolve_compress_threshold_ratio,
    resolve_context_window,
    resolve_max_input_tokens,
)


class TestResolveContextWindow(unittest.TestCase):
    def test_valid_value_returned(self):
        self.assertEqual(resolve_context_window(200000), 200000)

    def test_min_floor_4096(self):
        self.assertEqual(resolve_context_window(100), 4096)

    def test_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_context_window(None)

    def test_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            resolve_context_window("not_a_number")


class TestResolveMaxInputTokens(unittest.TestCase):
    def test_defaults_to_context_window(self):
        self.assertEqual(
            resolve_max_input_tokens(None, context_window=262144),
            262144,
        )

    def test_explicit_value(self):
        self.assertEqual(
            resolve_max_input_tokens(192000, context_window=262144),
            192000,
        )

    def test_clamped_to_context_window(self):
        self.assertEqual(
            resolve_max_input_tokens(999999, context_window=262144),
            262144,
        )

    def test_floor_4096(self):
        self.assertEqual(
            resolve_max_input_tokens(100, context_window=262144),
            4096,
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            resolve_max_input_tokens("bad", context_window=262144)


class TestResolveCompressThresholdRatio(unittest.TestCase):
    def test_explicit_takes_precedence(self):
        self.assertEqual(resolve_compress_threshold_ratio(0.85), 0.85)

    def test_explicit_clamped_to_05_99(self):
        self.assertEqual(resolve_compress_threshold_ratio(0.1), 0.5)
        self.assertEqual(resolve_compress_threshold_ratio(1.5), 0.99)

    def test_env_override(self):
        with patch.dict("os.environ", {"XIANGQI_COMPRESS_THRESHOLD_RATIO": "0.8"}):
            self.assertEqual(resolve_compress_threshold_ratio(None), 0.8)

    def test_default_when_no_env_no_yaml(self):
        # Ensure env doesn't leak from previous test.
        with patch.dict("os.environ", {"XIANGQI_COMPRESS_THRESHOLD_RATIO": ""}, clear=True):
            # Note: _yaml_compress may return something from config.yaml
            # (which is gitignored); the test just verifies it doesn't raise
            # and produces a value in the valid range.
            result = resolve_compress_threshold_ratio(None)
            self.assertGreaterEqual(result, 0.5)
            self.assertLessEqual(result, 0.99)
            self.assertEqual(DEFAULT_COMPRESS_THRESHOLD_RATIO, 0.9)


if __name__ == "__main__":
    unittest.main()
