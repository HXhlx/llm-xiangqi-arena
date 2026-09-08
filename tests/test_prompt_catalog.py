"""Unified prompt catalog: dotted keys, no hardcoded LLM copy in callers."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt_registry import (
    PROMPTS_DIR,
    _load_prompt_file,
    _prompt_file_path,
    get_prompt_profile,
    list_prompt_profiles,
    load_prompt,
    render_prompt,
)


REQUIRED_KEYS = (
    "player.zh",
    "player.en",
    "player.tools",
    "agent.compress",
    "agent.turn_summary",
    "agent.emotion_line",
)


class PromptCatalogTests(unittest.TestCase):
    def test_load_unknown_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_prompt("does.not.exist")
        self.assertIn("does.not.exist", str(ctx.exception))

    def test_required_catalog_keys_exist(self):
        for key in REQUIRED_KEYS:
            doc = load_prompt(key)
            self.assertIsInstance(doc, dict, key)
            self.assertTrue(doc, key)

    def test_player_en_is_default_profile(self):
        profile = get_prompt_profile("en")
        self.assertEqual(profile["name"], "en")
        self.assertTrue(profile["is_default"])
        self.assertIn("preview", profile["system_prompt"])
        self.assertIn("{fen}", profile["system_prompt"])
        self.assertIn("{own_pieces}", profile["system_prompt"])

    def test_player_zh_profile_still_has_required_fields(self):
        profile = get_prompt_profile("zh")
        self.assertEqual(profile["name"], "zh")
        self.assertFalse(profile["is_default"])
        self.assertIn("preview", profile["system_prompt"])
        self.assertIn("{fen}", profile["system_prompt"])
        self.assertIn("{own_pieces}", profile["system_prompt"])
        self.assertIn("{side_name_zh}", profile["turn_prompt"])

    def test_list_profiles_does_not_scan_non_player_yaml(self):
        names = [p["name"] for p in list_prompt_profiles()]
        self.assertEqual(set(names), {"zh", "en"})

    def test_render_prompt_formats_placeholders(self):
        text = render_prompt("agent.turn_summary", "user", move="h2e2", move_zh="炮二平五", thought_text="先开中炮")
        self.assertIn("h2e2", text)
        self.assertIn("炮二平五", text)
        self.assertIn("先开中炮", text)

    def test_player_tools_cover_observe_functions(self):
        tools = load_prompt("player.tools")
        for name in (
            "make_move",
            "preview",
            "preview_reset",
            "get_legal_moves",
            "get_threats",
            "get_board",
        ):
            self.assertIn(name, tools)
            self.assertTrue(str(tools[name].get("description") or "").strip())
        self.assertNotIn("preview_move", tools)
        self.assertNotIn("preview_undo", tools)
        self.assertNotIn("get_recent_moves", tools)
        self.assertIn("preview_move", str(tools["preview"]["description"]))

    def test_preview_is_signal_only(self):
        tools = load_prompt("player.tools")
        desc = str(tools["preview"]["description"])
        self.assertIn("working board", desc)
        self.assertNotIn("cannot preview", desc.lower())
        self.assertNotIn("cannot chain", desc.lower())

    def test_observe_copy_states_scope_not_workflow(self):
        tools = load_prompt("player.tools")
        preview = str(tools["preview"]["description"])
        threats = str(tools["get_threats"]["description"])
        zh = get_prompt_profile("zh")["system_prompt"]
        en = get_prompt_profile("en")["system_prompt"]
        zh_obs = "\n".join(
            line for line in zh.splitlines() if "get_threats" in line or "preview：" in line
        )
        en_obs = "\n".join(
            line
            for line in en.splitlines()
            if "get_threats" in line or line.strip().startswith("- preview:")
        )
        for text in (preview, threats, zh_obs, en_obs):
            lowered = text.lower()
            self.assertNotIn("you must", lowered)
            self.assertNotIn("always finish", lowered)
            self.assertNotIn("always call", lowered)
            self.assertNotIn("required loop", lowered)
        self.assertNotIn("必须", zh_obs)
        self.assertIn("side to move", threats.lower())
        self.assertIn("无明显威胁", threats)
        self.assertIn("轮走方", zh_obs)
        self.assertIn("无明显威胁", zh_obs)
        self.assertIn("推荐", zh)
        self.assertIn("side to move", en_obs.lower())
        self.assertIn("no obvious threat", en_obs.lower())
        self.assertIn("recommended", en.lower())

    def test_agent_compress_zh_fallback(self):
        from prompt_registry import PROMPTS_DIR, _load_prompt_file, _prompt_file_path

        zh_path = Path(PROMPTS_DIR) / "agent" / "compress.zh.yaml"
        self.assertTrue(zh_path.is_file())
        doc = _load_prompt_file(str(zh_path))
        self.assertTrue(doc)
        resolved = _prompt_file_path("agent", "compress", PROMPTS_DIR, lang="zh")
        self.assertTrue(resolved.endswith("compress.zh.yaml"))

    def test_legacy_prompt_paths_are_gone(self):
        root = Path(PROJECT_ROOT) / "prompts"
        leftovers = [
            root / "zh.yaml",
            root / "en.yaml",
            root / "turn_summary_zh.txt",
            root / "postmortem",
        ]
        missing = [str(p) for p in leftovers if p.exists()]
        self.assertEqual(missing, [])

    def test_callers_render_catalog_copy(self):
        from llm_client import TOOL_DEFINITIONS

        preview = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "preview")
        self.assertEqual(
            preview["function"]["description"],
            str(load_prompt("player.tools")["preview"]["description"]).strip(),
        )

    def test_load_prompt_reads_from_custom_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain = Path(tmp) / "agent"
            domain.mkdir()
            (domain / "sample.yaml").write_text(
                "system: hello\nuser: |\n  world {name}\n",
                encoding="utf-8",
            )
            doc = load_prompt("agent.sample", root=tmp)
            self.assertEqual(doc["system"], "hello")
            self.assertEqual(
                render_prompt("agent.sample", "user", name="红", root=tmp).rstrip("\n"),
                "world 红",
            )


if __name__ == "__main__":
    unittest.main()
