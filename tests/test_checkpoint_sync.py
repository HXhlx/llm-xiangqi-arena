"""Compressed LLM context is written back to LangGraph / Redis state."""

from __future__ import annotations

import os
import sys
import unittest

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.checkpoint_sync import (
    merge_model_messages,
    openai_dicts_to_lc_messages,
    replace_checkpoint_messages,
)
from agent.compress import checkpoint_messages_need_sync, estimate_tokens, hard_trim, without_orphan_tools


class CheckpointSyncDecisionTests(unittest.TestCase):
    def test_same_list_is_not_a_sync(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.assertFalse(checkpoint_messages_need_sync(msgs, msgs))

    def test_rewritten_list_needs_sync(self):
        before = [{"role": "user", "content": "history " * 400}]
        after = [{"role": "system", "content": "[对局历史摘要]\n短"}]
        self.assertTrue(checkpoint_messages_need_sync(before, after))
        self.assertLess(estimate_tokens(after), estimate_tokens(before))

    def test_same_length_different_content_needs_sync(self):
        before = [{"role": "user", "content": "aaaaaaaaaa"}]
        after = [{"role": "user", "content": "bbbbbbbbbb"}]
        self.assertTrue(checkpoint_messages_need_sync(before, after))


class CheckpointReplaceTests(unittest.TestCase):
    def test_roundtrip_keeps_tool_pairing(self):
        oai = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "observe", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "ok",
                "tool_call_id": "c1",
                "name": "observe",
            },
        ]
        lc = openai_dicts_to_lc_messages(oai)
        self.assertIsInstance(lc[0], AIMessage)
        self.assertEqual(lc[0].tool_calls[0]["name"], "observe")
        self.assertIsInstance(lc[1], ToolMessage)
        self.assertEqual(lc[1].tool_call_id, "c1")

    def test_replace_clears_old_checkpoint_messages(self):
        old = [
            HumanMessage(content="old-1 " * 80),
            HumanMessage(content="keep me"),
        ]
        compressed = [
            {"role": "system", "content": "[对局历史摘要]\n开局"},
            {"role": "user", "content": "keep me"},
        ]
        update = merge_model_messages(
            [{"role": "user", "content": old[0].content}, {"role": "user", "content": "keep me"}],
            compressed,
            [AIMessage(content="ok")],
        )
        self.assertIsInstance(update[0], RemoveMessage)
        self.assertEqual(update[0].id, REMOVE_ALL_MESSAGES)
        merged = add_messages(old, update)
        self.assertEqual(len(merged), 3)
        self.assertIsInstance(merged[0], SystemMessage)
        self.assertIn("摘要", merged[0].content)
        self.assertEqual(merged[1].content, "keep me")
        self.assertEqual(merged[2].content, "ok")

    def test_no_compress_only_appends_new_model_messages(self):
        raw = [{"role": "user", "content": "hi"}]
        update = merge_model_messages(raw, raw, [AIMessage(content="ok")])
        self.assertEqual(len(update), 1)
        self.assertIsInstance(update[0], AIMessage)
        merged = add_messages([HumanMessage(content="hi")], update)
        self.assertEqual([m.content for m in merged], ["hi", "ok"])

    def test_orphan_tool_is_dropped_before_replace(self):
        before = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "observe", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "ok", "tool_call_id": "c1", "name": "observe"},
            {"role": "user", "content": "next"},
        ]
        cut = [before[1], before[2]]
        update = merge_model_messages(before, cut, [])
        merged = add_messages(
            [
                AIMessage(content="", tool_calls=[{"id": "c1", "name": "observe", "args": {}, "type": "tool_call"}]),
                ToolMessage(content="ok", tool_call_id="c1", name="observe"),
                HumanMessage(content="next"),
            ],
            update,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].content, "next")

    def test_hard_trim_drops_orphan_tool(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "content": "orphan", "tool_call_id": "old", "name": "observe"},
            {"role": "user", "content": "now"},
        ]
        trimmed = hard_trim(msgs, context_window=8000, keep_ratio=0.5)
        self.assertTrue(all(m.get("tool_call_id") != "old" for m in trimmed))
        self.assertEqual(without_orphan_tools(trimmed), trimmed)
        self.assertTrue(any(m.get("content") == "now" for m in trimmed))

    def test_compress_without_new_messages_still_replaces(self):
        before = [{"role": "user", "content": "long " * 200}]
        after = [{"role": "system", "content": "[对局历史摘要]\n短"}]
        update = merge_model_messages(before, after, [])
        self.assertEqual(len(replace_checkpoint_messages(after)), 2)
        self.assertIsInstance(update[0], RemoveMessage)
        merged = add_messages([HumanMessage(content=before[0]["content"])], update)
        self.assertEqual(len(merged), 1)
        self.assertIn("摘要", merged[0].content)


if __name__ == "__main__":
    unittest.main()
