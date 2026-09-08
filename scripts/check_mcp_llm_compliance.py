#!/usr/bin/env python3
"""Post-hoc compliance check: heuristic/script-bot patterns in subagent transcripts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FORBIDDEN = [
    (re.compile(r"\bVAL\s*="), "VAL piece-value table"),
    (re.compile(r"opening_seq\s*="), "opening_seq list"),
    (re.compile(r"def\s+move_score\b"), "move_score heuristic"),
    (re.compile(r"while\s+True[\s\S]{0,800}?submit_move"), "while-loop multi-move submit"),
    (re.compile(r"MAX_PLY\s*="), "MAX_PLY auto-play loop"),
    (re.compile(r"pikafish|stockfish|mcts", re.I), "engine/MCTS mention in code path"),
]


def scan_shell_commands(transcript: Path) -> list[dict]:
    hits: list[dict] = []
    for i, line in enumerate(transcript.read_text(encoding="utf-8").splitlines(), 1):
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = o.get("message") or o
        content = msg.get("content") if isinstance(msg, dict) else o.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "tool_use":
                continue
            if c.get("name") not in {"Shell", "Bash"}:
                continue
            cmd = str((c.get("input") or {}).get("command") or "")
            for rx, label in FORBIDDEN:
                if rx.search(cmd):
                    hits.append({"line": i, "label": label, "cmd_len": len(cmd)})
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", type=Path)
    args = ap.parse_args()
    any_hit = False
    for p in args.transcripts:
        hits = scan_shell_commands(p) if p.exists() else [{"label": "missing_file"}]
        print(f"== {p.name} hits={len(hits)}")
        for h in hits:
            any_hit = True
            print(" ", h)
    return 1 if any_hit else 0


if __name__ == "__main__":
    raise SystemExit(main())
