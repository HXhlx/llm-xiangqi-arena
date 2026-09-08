"""MCP external-player contract: brief text + structured rules (soft policy).

Shared by server.py (claim-seat / state) and mcp_server.py (tools).
Behavioral rules remain soft (prompt + tool results + audit); this module
makes them discoverable without host-side markdown paste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from agent.locale import get_lang

CONTRACT_VERSION = "mcp-agent-v1"
PLAY_MODE = "llm_stepwise"

_ROOT = Path(__file__).resolve().parent


def _brief_path() -> Path:
    if get_lang() == "zh":
        zh = _ROOT / "prompts" / "player" / "mcp_external.zh.md"
        if zh.is_file():
            return zh
    return _ROOT / "prompts" / "player" / "mcp_external.md"


RULES_EN: dict[str, Any] = {
    "must": [
        "Each ply: the model decides from the live position, then submit_move",
        "submit_move note is one spoken reason",
        "Turn: wait_my_turn → get_board/get_legal_moves (optional preview) → decide → submit_move",
        "Long opponent thinks (30+ minutes) are normal: wait_my_turn timeout is not an error and not a resignation — keep waiting; only status=finished ends the game; never quit because the other side is quiet",
    ],
    "forbid": [
        "One Shell/Python while/for loop that auto-plays many plies",
        "Hard-coded opening_seq / VAL material table / move_score greed",
        "Pikafish, other engines, MCTS, or alpha-beta picking the move",
    ],
    "turn_loop": (
        "wait_my_turn → get_legal_moves (optional preview/get_threats)"
        " → write reason+ICCS → submit_move(note=...)"
    ),
}

RULES_ZH: dict[str, Any] = {
    "must": [
        "每一手由大模型根据当前局面决定着法后再 submit_move",
        "submit_move 的 note 写一句人话理由",
        "回合：wait_my_turn → get_board/get_legal_moves（可选 preview）→ 决策 → submit_move",
        "对方长考（半小时甚至更久）属正常：wait_my_turn 超时不是错误也不是对方退出，继续续等即可；只有 status=finished 才是对局结束，任何时候都不要因对方沉默而主动退出或认输",
    ],
    "forbid": [
        "一条 Shell/Python while/for 循环自动下多手",
        "写死 opening_seq / VAL 子力表 / move_score 贪心代打",
        "Pikafish、其它引擎、MCTS、αβ 搜索代选着",
    ],
    "turn_loop": (
        "wait_my_turn → get_legal_moves（可选 preview/get_threats）"
        " → 写出理由+ICCS → submit_move(note=...)"
    ),
}

MY_TURN_REMINDER_EN = (
    "LLM duel: you decide this ply, then submit_move. "
    "No multi-move scripts, material-score greed, or engine play."
)
MY_TURN_REMINDER_ZH = (
    "大模型对决：本手由你决策后 submit_move；禁止多手脚本/子力分贪心/引擎代打。"
)
# Back-compat alias; live locale is applied in attach_contract / wait_my_turn.
MY_TURN_REMINDER = MY_TURN_REMINDER_EN

FALLBACK_BRIEF_EN = (
    "This project is an LLM-plays-the-pieces duel. Decide each ply, then "
    "submit_move. No heuristic scripts and no engine play. "
    "See docs/mcp-agent-contract.md."
)
FALLBACK_BRIEF_ZH = (
    "本项目定位为大模型执子对决。逐步决策后 submit_move；"
    "禁止启发式脚本与引擎代打。详见 docs/mcp-agent-contract.md。"
)


# Default public surface (English). Live locale: rules_for_lang().
RULES = RULES_EN


def rules_for_lang() -> dict[str, Any]:
    return RULES_ZH if get_lang() == "zh" else RULES_EN


def brief_path() -> Path:
    return _brief_path()


def load_player_brief() -> str:
    path = brief_path()
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return FALLBACK_BRIEF_ZH if get_lang() == "zh" else FALLBACK_BRIEF_EN


def contract_payload(*, include_brief: bool = True) -> dict[str, Any]:
    rules = rules_for_lang()
    out: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "play_mode": PLAY_MODE,
        "rules": {
            "must": list(rules["must"]),
            "forbid": list(rules["forbid"]),
            "turn_loop": rules["turn_loop"],
        },
    }
    if include_brief:
        out["player_brief"] = load_player_brief()
    return out


def attach_contract(
    payload: Optional[dict],
    *,
    include_brief: bool = True,
    reminder: bool = False,
) -> dict[str, Any]:
    """Merge contract fields into an ok-ish tool/API response dict."""
    out: dict[str, Any] = dict(payload or {})
    out.update(contract_payload(include_brief=include_brief))
    if reminder:
        out["rules_reminder"] = (
            MY_TURN_REMINDER_ZH if get_lang() == "zh" else MY_TURN_REMINDER_EN
        )
    return out


def seat_claimed(game, side: str) -> bool:
    tokens = getattr(game, "seat_tokens", None) or {}
    return bool(tokens.get(side))
