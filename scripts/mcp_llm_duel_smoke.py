#!/usr/bin/env python3
"""Referee helper: open a dual-external LLM duel for any client host.

Unified join playbook (subagent / multi-session / remote MCP):

  1. This script (or any referee) creates the game with auto_claim=false
  2. Writes meta.json + per-seat prompts — NO seat tokens shared
  3. Each agent claim_seat(own side) then plays stepwise

Does not play moves. Optional --preclaim is debug-only (discouraged).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("XIANGQI_API_BASE", "http://127.0.0.1:18010")

import asyncio

import mcp_server

BRIEF = (ROOT / "prompts" / "player" / "mcp_external.md").read_text(encoding="utf-8")


def _seat_prompt(
    *,
    name: str,
    side: str,
    game_id: str,
    api_base: str,
    max_own_moves: int,
    work_dir: Path,
) -> str:
    side_zh = "红" if side == "red" else "黑"
    opp = "黑方" if side == "red" else "红方"
    return f"""你是象棋选手「{name}」，执{side_zh}（side={side}）。这是认真的**大模型对决**实测。

{BRIEF}

## 本局参数（裁判已开席，未发牌）

- API：`XIANGQI_API_BASE={api_base}`（调用前 export 或在进程环境设置）
- 仓库：`{ROOT}`
- game_id：`{game_id}`
- side：`{side}`（只入本席）
- 工作目录：`{work_dir}`（meta.json 只有 game_id/side/name，**没有**对方 token）

## 入座（与远程/多会话同一剧本）

1. （可选）`get_player_brief()`
2. `claim_seat(game_id, "{side}", name="{name}")` → **只保存返回的本席 seat_token**；阅读注入的 rules
3. 不要读取或使用对方的 token；不要把 token 写回共享 meta.json

## 任务

1. 用 `mcp_server`（仓库根、已设 `XIANGQI_API_BASE`）按契约**逐步**落子。
2. 你最多下 **{max_own_moves}** 手己方棋；达到后若未终局则 `resign`，并写短报告。
3. 每一手在提交前用自然语言说明理由；`submit_move` 的 `note` 写一句人话。
4. 终局或达上限后，把报告写到 `{work_dir}/{side}-report.txt`。

对手是另一名大模型选手（{opp}），用同一入座剧本。用 `wait_my_turn` 等真实落子。

立刻：先 claim_seat，再开下。
"""


def _referee_readme(work: Path, gid: str, api_base: str) -> str:
    return f"""# 裁判说明（统一入座剧本）

game_id: `{gid}`
api_base: `{api_base}`
work_dir: `{work}`

## 宿主无关

子智能体、两个 Cursor 会话、远程 MCP 客户端都走同一路径：

1. 裁判已 `create_duel(auto_claim=false)`（本脚本）
2. 红读 `red-prompt.txt` → `claim_seat(red)` → 逐步落子
3. 黑读 `black-prompt.txt` → `claim_seat(black)` → 逐步落子

`meta.json` **故意不含** seat_token。token 只存在各 agent 私有记忆里。
"""


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", default=os.environ.get("XIANGQI_API_BASE", "http://127.0.0.1:18010"))
    ap.add_argument("--work-dir", default="/tmp/xiangqi-mcp-llm-duel")
    ap.add_argument("--red-name", default="赤练")
    ap.add_argument("--black-name", default="玄戈")
    ap.add_argument("--max-own-moves", type=int, default=6, help="each seat's move cap before resign")
    ap.add_argument(
        "--preclaim",
        action="store_true",
        help="DEBUG only: referee claims both seats and writes tokens into meta.json (breaks unified playbook)",
    )
    args = ap.parse_args()

    os.environ["XIANGQI_API_BASE"] = args.api_base
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    out = await mcp_server.create_duel(
        red={"type": "external", "name": args.red_name},
        black={"type": "external", "name": args.black_name},
        auto_claim=False,
    )
    if not out.get("ok"):
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    gid = out["game_id"]
    red_tok = black_tok = None
    if args.preclaim:
        red_claim = await mcp_server.claim_seat(gid, "red", args.red_name)
        black_claim = await mcp_server.claim_seat(gid, "black", args.black_name)
        if not red_claim.get("ok") or not black_claim.get("ok"):
            print(json.dumps({"red": red_claim, "black": black_claim}, ensure_ascii=False, indent=2))
            return 1
        red_tok = red_claim["seat_token"]
        black_tok = black_claim["seat_token"]

    meta = {
        "api_base": args.api_base,
        "game_id": gid,
        "max_own_moves": args.max_own_moves,
        "mode": "llm_stepwise",
        "join_playbook": "referee_open__agents_claim",
        "auto_claim": False,
        "preclaim": bool(args.preclaim),
        "red": {
            "name": args.red_name,
            "side": "red",
            **({"seat_token": red_tok} if red_tok else {}),
        },
        "black": {
            "name": args.black_name,
            "side": "black",
            **({"seat_token": black_tok} if black_tok else {}),
        },
        "create_duel": {
            k: out.get(k)
            for k in ("ok", "game_id", "auto_claim", "external_sides", "open_seats", "hint")
        },
    }
    (work / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (work / "REFEREE.md").write_text(_referee_readme(work, gid, args.api_base), encoding="utf-8")
    (work / "red-prompt.txt").write_text(
        _seat_prompt(
            name=args.red_name,
            side="red",
            game_id=gid,
            api_base=args.api_base,
            max_own_moves=args.max_own_moves,
            work_dir=work,
        ),
        encoding="utf-8",
    )
    (work / "black-prompt.txt").write_text(
        _seat_prompt(
            name=args.black_name,
            side="black",
            game_id=gid,
            api_base=args.api_base,
            max_own_moves=args.max_own_moves,
            work_dir=work,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "work_dir": str(work),
                "game_id": gid,
                "max_own_moves": args.max_own_moves,
                "join_playbook": "referee_open__agents_claim",
                "preclaim": bool(args.preclaim),
                "next": "Launch red/black agents with red-prompt.txt / black-prompt.txt (each claim_seat)",
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
