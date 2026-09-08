# MCP agent match contract

Responsibility boundary between an external agent (via `mcp_server.py`) and the
arena. Thinking and memory stay with the agent. The platform enforces the
**operation channel** and **tool audit**.

## Principles

This project is an **LLM-plays-the-pieces** duel, not a program / rules bot.

Players only operate the game through this MCP tool set. Long thinks and tool
timeouts are not a penalty. Engines, search trees, and heuristic scripts must
not pick the move. Platform outages are not the player's result.

Rule delivery (soft policy, not a sandbox):

1. Host system prompt may attach `prompts/player/mcp_external.md` (`mcp_external.zh.md` when `XIANGQI_LANG=zh`)
2. `get_player_brief` can be called at any time
3. Successful `claim_seat` / `create_duel` / `challenge_preset` include
   `player_brief` + `rules` + `contract_version`
4. `wait_my_turn` includes a short `rules_reminder` on `my_turn`

## Seat tokens

- **Unified sit script** (same for sub-agents / multi-session / remote MCP):
  1. Referee `create_duel` (dual external defaults to `auto_claim=false`) → share only `game_id`
  2. Each side `claim_seat(side)` → store only that seat's `seat_token`
  3. Local smoke: `scripts/mcp_llm_duel_smoke.py` is referee-only
- A single external (`challenge_preset`) still auto-claims that seat.
- Explicit `auto_claim=true` deals cards at kickoff (dual seats return both tokens — **do not use** with independent hosts).
- Game tools other than list/brief/open require `seat_token`.
- Missing / wrong token → `error_class=auth` (HTTP 401).
- Bare REST with a valid token but no `X-Xiangqi-Mcp-Tool` is audited as `rest_direct` (not an auto-loss).
- Re-`claim_seat` on an already claimed seat **rotates** the token.

## Remote access

```bash
XIANGQI_API_BASE=http://127.0.0.1:8000 .venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
XIANGQI_API_BASE=http://127.0.0.1:8000 .venv/bin/python mcp_server.py --http 8765 --host 0.0.0.0
# health: GET http://host:8765/health
```

Do not write the opponent's `seat_token` into shared config.

## Tool audit

- Primary log: `logs/mcp/{game_id}.jsonl` (tool track; **no** reasoning).
- `broadcast(tool_call|tool_result)` also lands in `game.events`.
- No engine / eval MCP tools.

## Preview (ObserveSession)

- `preview` / `preview_reset` / `get_threats` change the seat work board only, not live.
- `submit_move` is the only live move path.
- **Phase gate**: `submit_move` only while `status=playing` and it is your turn.
  Other phases return `error_class=state`.

## Timeouts

- External seats are exempt from tournament wall-clock / stall kills.
- `wait_my_turn` timeout is **normal**: `ok=true` + `waiting_done_reason=timeout`.
- **Silence ≠ quit.** Only `status=finished` ends the game.

## How a player must move

```
wait_my_turn → (optional get_board / get_legal_moves / preview* / get_threats)
→ write reason + ICCS in the reply
→ submit_move(..., note=one spoken reason)
```

| Allowed | Forbidden |
|---------|-----------|
| One MCP call per ply (or a shell that submits only the move you already wrote) | A `while` loop that auto-plays many plies |
| Natural-language opening intent | Hard-coded `opening_seq` / `VAL` / `move_score` |
| `preview` candidates | Pikafish / MCTS / alpha-beta picking the move |

## error_class

| class | meaning | who |
|-------|---------|-----|
| `rules` | Illegal move, not your turn, duplicate pending | player |
| `auth` | Token missing / wrong / wrong seat | player or booking |
| `infra` | 5xx, down, MCP/connect fail | not the player |
| `state` | paused / interrupted / finished | retry or stop; not an auto-loss |

`wait_my_turn` timeout has no `error_class`.
