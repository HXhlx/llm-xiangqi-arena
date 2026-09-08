"""Score LLM-duel support usage. Chess strength is out of scope."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from agent.observe import SCORE_FORBIDDEN
from agent.turn_store import iter_turns, turns_path

# Substantive analysis only. preview_reset clears a path but does not itself
# inspect the position — a turn that is only reset+make_move still counts as
# blind. Legacy preview_move aliases to preview at dispatch and counts here.
QUERY_TOOLS = frozenset(
    {"get_board", "get_legal_moves", "get_threats", "preview", "preview_move"}
)
KNOWN_BAD_ZH = ("马9进8", "卒6进2", "将6进2")
THREAT_MISREAD = re.compile(
    r"(我的?车.{0,12}安全|有根.{0,8}安全|不怕被?吃|车安全所以.{0,16}平[中五])"
)
LIVE_ILLEGAL = re.compile(r"illegal move:.*live", re.IGNORECASE)


def _tool_calls(turn: dict[str, Any]) -> list[tuple[str, Any, str]]:
    rows: list[tuple[str, Any, str]] = []
    for rnd in turn.get("tool_rounds") or []:
        if not isinstance(rnd, dict):
            continue
        results = {
            (tr.get("name") or ""): str(tr.get("content") or "")
            for tr in rnd.get("tool_results") or []
            if isinstance(tr, dict)
        }
        leftover = list(rnd.get("tool_results") or [])
        for tc in rnd.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name") or ""
            args = tc.get("args") if "args" in tc else tc.get("arguments")
            content = results.get(name, "")
            if not content:
                for tr in leftover:
                    if isinstance(tr, dict) and (tr.get("name") or "") == name:
                        content = str(tr.get("content") or "")
                        leftover.remove(tr)
                        break
            rows.append((name, args, content))
    return rows


def _reasoning_text(turn: dict[str, Any]) -> str:
    parts = [
        str(turn.get("reasoning") or ""),
        str(turn.get("thinking") or ""),
        str(turn.get("assistant_content") or ""),
        str(turn.get("move_zh") or ""),
    ]
    return "\n".join(parts)


def _is_live_illegal_result(content: str) -> bool:
    return bool(LIVE_ILLEGAL.search(content)) or (
        "Illegal move" in content and "live, not work" in content
    )


def _is_live_ok_result(content: str) -> bool:
    return "OK: Move" in content or "is valid and will be played" in content


def _live_illegal_make_moves(turn: dict[str, Any]) -> int:
    n = 0
    for name, _args, content in _tool_calls(turn):
        if name == "make_move" and _is_live_illegal_result(content):
            n += 1
    return n


def _unrecovered_live_illegal(turn: dict[str, Any]) -> bool:
    """Rejected live submits that are later replaced by a legal make_move are support success."""
    saw_illegal = False
    recovered = False
    for name, _args, content in _tool_calls(turn):
        if name != "make_move":
            continue
        if _is_live_illegal_result(content):
            saw_illegal = True
            recovered = False
        elif _is_live_ok_result(content):
            recovered = True
    return saw_illegal and not recovered


def _used_query_tool(turn: dict[str, Any]) -> bool:
    return any(name in QUERY_TOOLS for name, _args, _content in _tool_calls(turn))


def _bad_chinese(text: str) -> list[str]:
    return [token for token in KNOWN_BAD_ZH if token in text]


def _score_leaks(text: str) -> list[str]:
    lowered = text.lower()
    return [token for token in SCORE_FORBIDDEN if token.lower() in lowered]


def _tool_results_text(turn: dict[str, Any]) -> str:
    return "\n".join(content for _n, _a, content in _tool_calls(turn))


def score_turn(turn: dict[str, Any]) -> dict[str, Any]:
    reasoning = _reasoning_text(turn)
    tools = _tool_results_text(turn)
    names = Counter(name for name, _a, _c in _tool_calls(turn))
    live_illegal = _live_illegal_make_moves(turn)
    unrecovered = _unrecovered_live_illegal(turn)
    used_query = _used_query_tool(turn)
    bad_zh = _bad_chinese(reasoning)
    leaks = _score_leaks(tools)
    threat_clear = "仍可被吃" in tools
    misread = bool(threat_clear and THREAT_MISREAD.search(reasoning))
    ok = (
        not unrecovered
        and used_query
        and not bad_zh
        and not leaks
        and not misread
    )
    return {
        "ply": turn.get("ply"),
        "side": turn.get("side"),
        "move": turn.get("move"),
        "move_zh": turn.get("move_zh"),
        "model": turn.get("model"),
        "ok": ok,
        "used_query_tool": used_query,
        "live_illegal_make_move": live_illegal,
        "unrecovered_live_illegal": unrecovered,
        "bad_chinese": bad_zh,
        "score_leaks": leaks,
        "threat_misread": misread,
        "tool_counts": dict(names),
    }


PHASES = (
    ("opening", 1, 20),
    ("middlegame", 21, 60),
    ("late", 61, 10_000),
)


def _ply_of(row: dict[str, Any]) -> int:
    try:
        return int(row.get("ply") or 0)
    except (TypeError, ValueError):
        return 0


def _phase_for_ply(ply: int) -> str:
    for name, lo, hi in PHASES:
        if lo <= ply <= hi:
            return name
    return "late"


def score_turns(turns: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [score_turn(t) for t in turns]
    live_illegal = sum(int(r["live_illegal_make_move"]) for r in rows)
    unrecovered = [r["ply"] for r in rows if r.get("unrecovered_live_illegal")]
    missing_query = [r["ply"] for r in rows if not r["used_query_tool"]]
    bad_zh = [r["ply"] for r in rows if r["bad_chinese"]]
    misread = [r["ply"] for r in rows if r["threat_misread"]]
    leaks = [r["ply"] for r in rows if r["score_leaks"]]
    s4_ok = (
        len(rows) >= 1
        and not unrecovered
        and not missing_query
        and not bad_zh
        and not misread
        and not leaks
    )
    for row in rows:
        row["phase"] = _phase_for_ply(_ply_of(row))
    return {
        "plies": len(rows),
        "s4_ok": s4_ok,
        "live_illegal_make_move": live_illegal,
        "unrecovered_live_illegal_plies": unrecovered,
        "missing_query_plies": missing_query,
        "bad_chinese_plies": bad_zh,
        "threat_misread_plies": misread,
        "score_leak_plies": leaks,
        "turns": rows,
    }


def score_phases(turns: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """S4 split so opening-pass cannot hide a mid/late support break."""
    scored = score_turns(turns)
    by_name: dict[str, list[dict[str, Any]]] = {name: [] for name, _lo, _hi in PHASES}
    for row in scored["turns"]:
        by_name[_phase_for_ply(_ply_of(row))].append(row)

    phases: dict[str, Any] = {}
    for name, _lo, _hi in PHASES:
        rows = by_name[name]
        if not rows:
            phases[name] = {"plies": 0, "s4_ok": None, "empty": True}
            continue
        missing_query = [r["ply"] for r in rows if not r["used_query_tool"]]
        unrecovered = [r["ply"] for r in rows if r.get("unrecovered_live_illegal")]
        bad_zh = [r["ply"] for r in rows if r["bad_chinese"]]
        misread = [r["ply"] for r in rows if r["threat_misread"]]
        leaks = [r["ply"] for r in rows if r["score_leaks"]]
        phases[name] = {
            "plies": len(rows),
            "empty": False,
            "s4_ok": not unrecovered and not missing_query and not bad_zh and not misread and not leaks,
            "live_illegal_make_move": sum(int(r["live_illegal_make_move"]) for r in rows),
            "unrecovered_live_illegal_plies": unrecovered,
            "missing_query_plies": missing_query,
            "bad_chinese_plies": bad_zh,
            "threat_misread_plies": misread,
            "score_leak_plies": leaks,
        }
    present = [name for name, block in phases.items() if not block.get("empty")]
    scored["phases"] = phases
    scored["phases_present"] = present
    scored["phases_ok"] = bool(present) and all(phases[name]["s4_ok"] for name in present)
    return scored


def score_game_file(game_id: str, min_plies: int = 20) -> dict[str, Any]:
    turns = list(iter_turns(game_id))
    report = score_phases(turns)
    report["game_id"] = game_id
    report["turns_path"] = str(turns_path(game_id))
    report["reached_min_plies"] = report["plies"] >= min_plies
    report["s4_ok"] = bool(report["s4_ok"] and report["reached_min_plies"])
    return report


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
