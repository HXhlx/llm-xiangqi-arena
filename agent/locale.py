"""Public-facing locale for player prompts and MCP briefs.

English is the default. Set ``XIANGQI_LANG=zh`` for Chinese player prompts
and MCP copy. Chinese glyphs on Xiangqi piece faces are artwork, not locale.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

Lang = Literal["en", "zh"]

ENV_LANG = "XIANGQI_LANG"
DEFAULT_LANG: Lang = "en"

_ZH_ALIASES = frozenset({"zh", "zh-cn", "zh_cn", "cn", "chinese"})
_EN_ALIASES = frozenset({"en", "en-us", "en_us", "english"})
# Unknown values already logged. One warning per raw value per process.
_warned_unknown_langs: set[str] = set()


def get_lang() -> Lang:
    """Active locale. Unknown ``XIANGQI_LANG`` values warn once and fall back to en.

    Blank and the English aliases (``en``, ``en-us``, ``english``) are English.
    ``zh-TW`` is not a Chinese alias and falls back to English. Each unknown
    raw value is logged at most once per process.
    """
    raw = (os.environ.get(ENV_LANG) or "").strip().lower()
    if not raw or raw in _EN_ALIASES:
        return "en"
    if raw in _ZH_ALIASES:
        return "zh"
    if raw not in _warned_unknown_langs:
        _warned_unknown_langs.add(raw)
        logger.warning("Unknown XIANGQI_LANG=%r; falling back to en", raw)
    return DEFAULT_LANG


def is_zh() -> bool:
    return get_lang() == "zh"


def brief_path_for(basename: str, *, root=None) -> str:
    """Return ``name.md`` or ``name.zh.md`` next to a default English file."""
    from pathlib import Path

    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    stem = Path(basename)
    if is_zh():
        zh = stem.with_name(f"{stem.stem}.zh{stem.suffix}")
        if zh.is_file():
            return str(zh)
    return str(stem if stem.is_absolute() else base / basename)
