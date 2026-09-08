"""
Prompt catalog for LLM Xiangqi Arena.

Each prompt lives at prompts/<domain>/<name>.yaml and is loaded by dotted key
(e.g. player.en, agent.compress). When XIANGQI_LANG=zh, agent keys prefer
``{name}.zh.yaml``. Player profiles (system/turn/retry) stay under
prompts/player/ so /api/prompts can list them.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from agent.locale import get_lang


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
PLAYER_PROMPTS_DIR = os.path.join(PROMPTS_DIR, "player")
LEGACY_PROMPT_NAME_MAP = {
    "zh": "zh",
    "en": "en",
}


def _load_prompt_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt file must contain a mapping: {path}")
    return data


def _prompt_file_path(
    domain: str,
    name: str,
    base: str,
    *,
    lang: str | None = None,
) -> str:
    """Prefer ``{name}.{lang}.yaml`` for non-English locales, else ``{name}.yaml``."""
    locale = lang if lang is not None else get_lang()
    if locale and locale != "en":
        localized = os.path.join(base, domain, f"{name}.{locale}.yaml")
        if os.path.isfile(localized):
            return localized
    return os.path.join(base, domain, f"{name}.yaml")


def load_prompt(key: str, *, root: str | os.PathLike | None = None) -> dict:
    parts = str(key).split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Prompt key must be 'domain.name', got '{key}'")
    domain, name = parts
    base = os.fspath(root) if root is not None else PROMPTS_DIR
    path = _prompt_file_path(domain, name, base)
    if not os.path.isfile(path):
        raise ValueError(f"Prompt '{key}' not found. Expected file: {path}")
    data = _load_prompt_file(path)
    if not data:
        raise ValueError(f"Prompt '{key}' is empty")
    return data


def render_prompt(
    key: str,
    field: str,
    *,
    root: str | os.PathLike | None = None,
    **kwargs: object,
) -> str:
    doc = load_prompt(key, root=root)
    if field not in doc:
        raise ValueError(f"Prompt '{key}' has no field '{field}'")
    text = str(doc[field])
    for name, value in kwargs.items():
        text = text.replace("{" + name + "}", str(value))
    return text


def list_prompt_profiles() -> list[dict]:
    profiles = []
    scan_dir = PLAYER_PROMPTS_DIR if os.path.isdir(PLAYER_PROMPTS_DIR) else PROMPTS_DIR
    if not os.path.isdir(scan_dir):
        return profiles

    for filename in sorted(os.listdir(scan_dir)):
        if not filename.lower().endswith((".yaml", ".yml")):
            continue
        path = os.path.join(scan_dir, filename)
        data = _load_prompt_file(path)
        if not str(data.get("system_prompt", "")).strip():
            continue
        name = str(data.get("name") or os.path.splitext(filename)[0]).strip()
        if not name:
            continue

        required_fields = ("system_prompt", "turn_prompt", "tool_retry_prompt")
        missing = [field for field in required_fields if not str(data.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Prompt '{name}' is missing required fields: {', '.join(missing)}")

        profiles.append({
            "name": name,
            "display_name": str(data.get("display_name") or name),
            "description": str(data.get("description") or ""),
            "system_prompt": str(data["system_prompt"]),
            "turn_prompt": str(data["turn_prompt"]),
            "tool_retry_prompt": str(data["tool_retry_prompt"]),
            "empty_legal_moves_text": str(data.get("empty_legal_moves_text") or "(none)"),
            "is_default": bool(data.get("default", False)),
        })

    return profiles


def get_default_prompt_name() -> str:
    profiles = list_prompt_profiles()
    for profile in profiles:
        if profile.get("is_default"):
            return profile["name"]
    preferred = "zh" if get_lang() == "zh" else "en"
    for profile in profiles:
        if profile["name"] == preferred:
            return profile["name"]
    for profile in profiles:
        if profile["name"] == "en":
            return profile["name"]
    return profiles[0]["name"] if profiles else "en"


def resolve_prompt_name(prompt_name: Optional[str] = None, prompt_lang: Optional[str] = None) -> str:
    if prompt_name:
        return prompt_name
    if prompt_lang:
        return LEGACY_PROMPT_NAME_MAP.get(prompt_lang, prompt_lang)
    return get_default_prompt_name()


def get_prompt_profile(prompt_name: Optional[str] = None, prompt_lang: Optional[str] = None) -> dict:
    resolved_name = resolve_prompt_name(prompt_name, prompt_lang)
    profiles = list_prompt_profiles()
    for profile in profiles:
        if profile["name"] == resolved_name:
            return profile
    available = ", ".join(profile["name"] for profile in profiles) or "(none)"
    raise ValueError(f"Prompt '{resolved_name}' not found. Available prompts: {available}")
