"""Global context-compression settings.

Precedence for each value:
  - context_window: only from config.yaml ``models[].context_window``
    (preset). resolve_context_window now requires the preset value;
    no env/yaml-global/defaults fallback. The model's own window is
    the only source of truth.
  - max_input_tokens: optional per-preset cap for compress/trim. When
    omitted, defaults to ``context_window``. Clamped to
    ``[4096, context_window]``.
  - threshold_ratio: env (XIANGQI_COMPRESS_THRESHOLD_RATIO) >
    config.yaml ``compress.threshold_ratio`` > 0.9.

compress_model is always the game model — not configured separately.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

DEFAULT_COMPRESS_THRESHOLD_RATIO = 0.9

ENV_COMPRESS_THRESHOLD_RATIO = "XIANGQI_COMPRESS_THRESHOLD_RATIO"

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_YAML = _ROOT / "config.yaml"


def _env_float(name: str) -> Optional[float]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _yaml_compress() -> Mapping[str, Any]:
    if not _CONFIG_YAML.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(_CONFIG_YAML.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    section = data.get("compress")
    return section if isinstance(section, Mapping) else {}


def resolve_context_window(explicit: Optional[int] = None) -> int:
    """Resolve context_window from the preset's per-model value.

    No fallback chain: caller (server.resolve_preset) must pass the
    preset's context_window. Raises ValueError if missing/invalid so
    misconfigured presets fail fast at game creation time.
    """
    if explicit is None:
        raise ValueError(
            "context_window is required from the model preset "
            "(config.yaml -> models[].context_window); "
            "no env/yaml-global fallback is available"
        )
    try:
        return max(4096, int(explicit))
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid context_window value: {explicit!r}") from e


def resolve_max_input_tokens(
    explicit: Optional[int] = None,
    *,
    context_window: int,
) -> int:
    """Input budget used by compress/trim (defaults to full context_window)."""
    cw = resolve_context_window(context_window)
    if explicit is None:
        return cw
    try:
        value = int(explicit)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid max_input_tokens value: {explicit!r}") from e
    return max(4096, min(cw, value))


def resolve_compress_threshold_ratio(
    explicit: Optional[float] = None,
    *,
    defaults: Optional[Mapping[str, Any]] = None,
) -> float:
    if explicit is not None:
        try:
            return min(0.99, max(0.5, float(explicit)))
        except (TypeError, ValueError):
            pass
    env_val = _env_float(ENV_COMPRESS_THRESHOLD_RATIO)
    if env_val is not None:
        return min(0.99, max(0.5, env_val))
    yaml_sec = _yaml_compress()
    yaml_val = yaml_sec.get("threshold_ratio")
    if yaml_val is None:
        yaml_val = yaml_sec.get("compress_threshold_ratio")
    if yaml_val is not None:
        try:
            return min(0.99, max(0.5, float(yaml_val)))
        except (TypeError, ValueError):
            pass
    if defaults:
        d = defaults.get("compress_threshold_ratio")
        if d is not None:
            try:
                return min(0.99, max(0.5, float(d)))
            except (TypeError, ValueError):
                pass
    return DEFAULT_COMPRESS_THRESHOLD_RATIO
