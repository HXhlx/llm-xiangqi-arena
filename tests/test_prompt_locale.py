"""Locale consistency: ZH MCP brief, observe labels, smoke-script shells."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.locale import get_lang
from mcp_contract import brief_path, load_player_brief


PLAYER_DIR = Path(PROJECT_ROOT) / "prompts" / "player"


class ChineseBriefTests(unittest.TestCase):
    def test_zh_brief_is_not_english_copy(self):
        en = (PLAYER_DIR / "mcp_external.md").read_text(encoding="utf-8")
        zh = (PLAYER_DIR / "mcp_external.zh.md").read_text(encoding="utf-8")
        self.assertNotEqual(en, zh)
        self.assertIn("大模型执子对决", zh)
        self.assertIn("统一入座剧本", zh)
        self.assertIn("走子阶段闸门", zh)
        self.assertIn("docs/mcp-agent-contract.zh.md", zh)
        self.assertNotIn("LLM-plays-the-pieces", zh)
        for text in (en, zh):
            self.assertIn("seat_token", text)
            self.assertIn("auto_claim=false", text)
        self.assertIn("store only your own `seat_token`", en)
        self.assertIn("Never write the opponent's token", en)
        self.assertIn("只保存本席 `seat_token`", zh)
        self.assertIn("不要把对方的 token 写进共享", zh)

    def test_brief_path_follows_lang(self):
        prev = os.environ.get("XIANGQI_LANG")
        try:
            os.environ.pop("XIANGQI_LANG", None)
            self.assertEqual(get_lang(), "en")
            self.assertTrue(str(brief_path()).endswith("mcp_external.md"))
            self.assertIn("LLM-plays-the-pieces", load_player_brief())

            os.environ["XIANGQI_LANG"] = "zh"
            self.assertEqual(get_lang(), "zh")
            self.assertTrue(str(brief_path()).endswith("mcp_external.zh.md"))
            self.assertIn("大模型执子对决", load_player_brief())
        finally:
            if prev is None:
                os.environ.pop("XIANGQI_LANG", None)
            else:
                os.environ["XIANGQI_LANG"] = prev


class SmokeScriptLocaleTests(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("XIANGQI_LANG")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("XIANGQI_LANG", None)
        else:
            os.environ["XIANGQI_LANG"] = self._prev

    def test_default_english_shell(self):
        os.environ.pop("XIANGQI_LANG", None)
        from scripts.mcp_llm_duel_smoke import _referee_readme, _seat_prompt

        text = _seat_prompt(
            name="RedSeat",
            side="red",
            game_id="g1",
            api_base="http://127.0.0.1:8000",
            max_own_moves=6,
            work_dir=Path("/tmp/duel"),
        )
        self.assertIn("LLM-plays-the-pieces", text)
        self.assertIn("claim this seat only", text)
        self.assertIn("store only the returned seat_token", text)
        self.assertIn("Do not read or use the opponent's token", text)
        self.assertIn("do not write tokens back into shared meta.json", text)
        self.assertIn("no cards dealt", text)
        self.assertNotIn("你是象棋选手", text)
        referee = _referee_readme(Path("/tmp/duel"), "g1", "http://127.0.0.1:8000")
        self.assertIn("Referee notes", referee)
        self.assertIn("intentionally omits** seat_token", referee)
        self.assertNotIn("裁判说明", referee)

    def test_zh_shell_when_lang_zh(self):
        os.environ["XIANGQI_LANG"] = "zh"
        from scripts.mcp_llm_duel_smoke import _referee_readme, _seat_prompt

        text = _seat_prompt(
            name="赤练",
            side="red",
            game_id="g1",
            api_base="http://127.0.0.1:8000",
            max_own_moves=6,
            work_dir=Path("/tmp/duel"),
        )
        self.assertIn("你是象棋选手", text)
        self.assertIn("大模型执子对决", text)
        self.assertIn("只入本席", text)
        self.assertIn("只保存返回的本席 seat_token", text)
        self.assertIn("不要读取或使用对方的 token", text)
        self.assertIn("不要把 token 写回共享 meta.json", text)
        self.assertIn("未发牌", text)
        self.assertNotIn("You are Xiangqi player", text)
        referee = _referee_readme(Path("/tmp/duel"), "g1", "http://127.0.0.1:8000")
        self.assertIn("裁判说明", referee)
        self.assertIn("故意不含** seat_token", referee)
        self.assertNotIn("Referee notes", referee)


if __name__ == "__main__":
    unittest.main()
