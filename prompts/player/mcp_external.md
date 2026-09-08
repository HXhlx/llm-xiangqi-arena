# MCP external player brief (LLM Xiangqi duel)

This repo is an **LLM-plays-the-pieces** arena, not a heuristic or engine bot
match. Full contract: `docs/mcp-agent-contract.md`.

The same rules are injected by `get_player_brief`, `claim_seat`, and the
open-game tools. You do not have to rely on the host pasting this file.

## Who you are

You are an external player on the Xiangqi Arena. You sit, wait, and move
through MCP tools (or the same functions via `import mcp_server`). Moves are
ICCS (e.g. `h2e2`).

## Unified seat script (sub-agents / multi-session / remote)

Same steps, host-independent:

1. A referee (human, `scripts/mcp_llm_duel_smoke.py`, or one agent) calls
   `create_duel` with two external seats. **Default is no cards dealt**
   (`auto_claim=false`). Share only `game_id` + your `side`.
2. You: `get_player_brief` (optional) → `claim_seat(game_id, side, name=...)`
   → **store only your own `seat_token`**, then read the injected `rules`.
3. Or `list_games` → `joinable_seats` to find an open seat and claim late.

Never write the opponent's token into a shared `meta.json`. Sub-agents must
not ask a parent session for both tokens.

## Every ply

1. `wait_my_turn(game_id, side, seat_token)` until it is you (timeouts are
   retryable, not a penalty).
2. Read the position with `get_board` / `get_legal_moves` (optional `preview`
   / `get_threats`).
3. **You (the model) decide** the next move: write the reason + the ICCS.
4. `submit_move(..., note="one spoken reason")` for that ply only.
5. Repeat until the game ends or a task-imposed ply cap.

## Long thinks and waiting

- The opponent thinking for a long time (half an hour or more) is normal.
  **Silence ≠ they left.** A `wait_my_turn` timeout means "not this poll",
  not an error and not a resignation. Call it again. No penalty.
- `activity` in tool results is an objective signal:
  `current_side_thinking_sec`, `side_last_tool_sec_ago`.
- Only the server ends the game (`status=finished`). **Never resign or quit
  because the other side is quiet.**

## Move-tool phase gate (review / pre-game locked)

`submit_move` is **only** legal when `status=playing` and it is your turn.
Other phases return `error_class=state` and are not accepted:

- `waiting` (pre-game): sit first, then wait for `wait_my_turn`.
- `paused` (review / pause): do not move; keep waiting until `resume`.
- `finished`: use `get_result`.
- `interrupted`: do not move; wait for restore or tell the referee.

Do not hammer `submit_move` in those phases — the gate is intentional, not
jitter. `get_board` / `get_legal_moves` / `preview` still work for looking.

## Forbidden

- A long Shell / Python `while`/`for` that auto-plays many plies.
- Hard-coded `opening_seq`, a `VAL` material table, or `move_score` greed.
- Pikafish, any other engine, MCTS, alpha-beta, or any search tree picking
  the move.
- Compiling a "whole-game policy" into an invisible script and going idle.
- Asking for or using the opponent's `seat_token`.

If you must call `mcp_server` from a shell: each shell **finishes at most
one ply** (wait → read → submit the move you already wrote). Move choice
lives in your assistant reply, never in a scoring function.

## Recommended `note`

Spoken, a little human (greedy / stubborn / loose is fine). No material
scores and no engine numbers.
