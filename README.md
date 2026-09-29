# LLM Xiangqi Arena

**Large language models play Xiangqi (Chinese chess) against each other — with seat-gated tools, not a hidden engine.**

This is a backend-only arena. FastAPI runs the match. LangGraph orchestrates in-process LLM/AI seats. MCP is a seat-gated tool surface for external agents. LiteLLM is optional: the arena calls one OpenAI-compatible LiteLLM proxy endpoint for the built-in AI-player type, and routing happens in that proxy. Dual-MCP external seats do not need it.

Chinese characters on the **piece faces** (帅 / 将 / 车 / …) are board artwork and stay. Product copy, player prompts, and MCP briefs default to **English**. Set `XIANGQI_LANG=zh` for the original Chinese player track.

Product surface: [SCOPE.md](SCOPE.md).

## Problem

Public LLM board-game arenas usually target chess. This one is Xiangqi: inspectable rules, optional engine analysis, and external agents that cannot move without a seat token. Models sit, think, and submit one ICCS ply at a time. Multi-move scripts, material-score greed, and engine-picked moves are contract-forbidden.

## Architecture

```text
                 FastAPI match server (server.py)
                   Xiangqi rules + cycle detection
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
   LangGraph per-side thread      MCP proxy (mcp_server.py)
     (Redis Stack; LLM/AI)          claim_seat → wait_my_turn
              │                     → preview* → submit_move
              ▼                     (external; seat-gated)
     LiteLLM (optional)                    │
     one OpenAI-compatible proxy           │
     routing happens in the proxy          │
     AI-player type only                   │
              │                           │
              └─────────────┬─────────────┘
                            ▼
                   game.log + snapshots
```

LangGraph is in-process per-seat orchestration for LLM/AI seats. MCP is a parallel, seat-gated tool surface for external agents. They are complementary, not a substitute or a linear pipeline.

| Layer | What it does |
|-------|----------------|
| **Xiangqi rules** (`xiangqi.py`, `cycle_rules.py`) | Legal moves, checks, repetitions |
| **FastAPI** (`server.py`) | Concurrent games: random / AI (LiteLLM) / LLM / external. Pikafish is optional analysis, not a playing seat. There is no human seat. |
| **LangGraph** | Per-side threads on Redis Stack; compress/trim the window |
| **MCP** (`mcp_server.py`) | Stateless proxy. Seat token + tool audit. No engine tools. |
| **LiteLLM** (optional) | One OpenAI-compatible proxy endpoint for the **AI-player** type. Routing happens in the proxy. Dual-MCP external seats do not need `LITELLM_*`. |

External agents must `claim_seat` and `submit_move` one ply at a time.

## What this proves

- **Seat gate.** Move, preview, wait, and resign require a `seat_token` from `claim_seat`. The first claim of an empty external seat needs `game_id` and `side`. Re-claiming an occupied seat requires that seat's current token or `XIANGQI_REFEREE_SECRET` (both the env value and the `X-Xiangqi-Referee-Secret` header are whitespace-stripped; an unset secret never matches). `POST /finish` accepts a client move list only for local games (neither side is `llm`, `ai`, or `external`). An already-finished server-driven game returns `already_finished` and does not rewrite history.
- **One ply.** Each external seat calls `submit_move` once per turn. Multi-move scripts are contract-forbidden.
- **Complementary surfaces.** LangGraph orchestrates in-process LLM/AI seats. MCP is a parallel gate for external agents. Neither replaces the other.
- **Dual-external without LiteLLM.** Two MCP seats can play each other. `LITELLM_API_BASE` / `LITELLM_API_KEY` are unused on that path.
- **Trust model.** Single-host arena, trusted referee. Control endpoints (create, start, pause, resume, seek, reset, restore) are unauthenticated unless a route says otherwise. Unauthenticated `POST /api/game/create` with `type: ai` can name any model and spend the shared `LITELLM_API_KEY` (a request `api_base` or `api_key` is ignored; the seat always uses the env proxy). Set optional `LITELLM_ALLOWED_MODELS` (comma-separated) to reject other model names with 400; when it is unset, every model name is accepted. Pause, then seek, then resume can roll an external game back to an earlier ply and drop queued external moves. Do not bind them on an untrusted network (`0.0.0.0` exposes that control plane). Snapshots do not store `seat_tokens`: after `POST /restore`, both external seats are unclaimed, so either side can be claimed first. Snapshots also omit API keys. A restored `ai` seat is bound again from `LITELLM_API_BASE`, `LITELLM_API_KEY`, and the preset or stored model. If those variables are missing, restore fails for that game instead of calling the proxy with an empty key. `XIANGQI_REFEREE_SECRET` is an operator credential: do not hand it, or the `X-Xiangqi-Referee-Secret` header value, to player agents or paste it into player prompts.
- **Tests.** `python -m unittest discover -s tests -q` covers rules, MCP, prompts, observe, and LangGraph. The suite passes with `XIANGQI_LANG` unset and with `XIANGQI_LANG=zh`.

## How to run

```bash
# 1) Python env
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: uv venv --python 3.12 && uv pip install -r requirements.txt

# 2) Local config (no secrets in git)
cp .env.example .env
cp config.example.yaml config.yaml

# 3) Redis Stack (LangGraph checkpointer — RediSearch / FT.* required)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
# then set REDIS_URL in .env
```

Pikafish is optional analysis, not a playing seat. Download a CPU-matched binary plus `pikafish.nnue` from [Pikafish releases](https://github.com/official-pikafish/Pikafish/releases) into `pikafish/` (see `pikafish/README.md`). Random, LLM, and AI games work without it. There is no human seat and no Pikafish seat.

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
# or: python server.py
```

- Service: `http://127.0.0.1:8000/`
- OpenAPI: `http://127.0.0.1:8000/docs`

### LLM / AI players

- **AI player** (`type: ai`): resolved at create time to an in-process LLM seat. The arena calls one OpenAI-compatible LiteLLM proxy (`LITELLM_API_BASE` + `LITELLM_API_KEY`); routing happens in the proxy. The model name is the preset's `model`, or `LITELLM_MODEL` when there is no preset. Optional `LITELLM_ALLOWED_MODELS` limits which names unauthenticated create may spend the shared key on. A preset may also supply `context_window` and the other LLM fields; without a preset, the request must include `context_window`. Missing proxy settings, or the placeholder key `sk-xxxxxxxx`, are a 400 at create, restore, and resume — the loop does not start and then fail. The seat is server-driven (`game_loop` plays it). `POST /finish` cannot overwrite a game that has an `ai` seat.
- **Illegal moves.** `XIANGQI_MAX_ILLEGAL_MOVES` (default 3) caps rejected `make_move` calls on one ply. Hitting the cap interrupts the game with no winner, the same outcome as the text-only strike cap (`status=interrupted`), not a forfeit. A legal `make_move` before the cap is played, and the next ply starts at zero. `XIANGQI_MAX_TOOL_ROUNDS` (default 32, clamped to 4–128) caps model/tool rounds on one ply. Exceeding it interrupts the game. Each round retries a failed call up to 3 times, so a stuck ply can send about 4 requests per round (the original call plus those retries).
- **Custom LLM** (`type: llm`): `api_base` / `api_key` / `model` in `config.yaml` → `models:[]`. A local endpoint with no auth (Ollama, vLLM) still needs any non-empty `api_key`; a blank or placeholder key is HTTP 400 before the loop starts.
- Default player prompt is **English** (`prompt_name: en`). Chinese: `prompt_name: zh` or `XIANGQI_LANG=zh`. An unknown `XIANGQI_LANG` (for example `fr` or `zh-TW`) logs one warning per process and falls back to English.
- Per-side threads via LangGraph `AsyncRedisSaver` + `REDIS_URL`

### MCP (external agents)

```bash
.venv/bin/python mcp_server.py                 # stdio
.venv/bin/python mcp_server.py --http 8765     # http://127.0.0.1:8765/mcp
```

`XIANGQI_API_BASE` defaults to `http://127.0.0.1:8000`. Move, preview, wait, and resign need the seat token. An empty seat is claimed with `game_id` and `side`; rotating an occupied seat needs the current token or the referee secret (`X-Xiangqi-Referee-Secret`, whitespace-stripped). The MCP `claim_seat` parameter that forwards the secret is for the referee process only. Do not hand that value to player agents, and do not name it in player briefs. Control endpoints other than those gates — including pause, resume, and seek — are unauthenticated. Pause + seek + resume can roll an external game back. Keep the server on localhost unless you trust every client. After `/restore`, seat tokens are gone (snapshots do not store them), so either side may claim first.

Contract: [`docs/mcp-agent-contract.md`](docs/mcp-agent-contract.md). Player brief: [`prompts/player/mcp_external.md`](prompts/player/mcp_external.md) (`mcp_external.zh.md` when `XIANGQI_LANG=zh`). Referee helper: `python scripts/mcp_llm_duel_smoke.py` (seat/referee copy follows `XIANGQI_LANG`).

### Dual-MCP demo (no LiteLLM)

Two external seats play through MCP. LiteLLM is not required.

```bash
# Terminal 1 — match server
uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2 — MCP HTTP surface
python mcp_server.py --http 8765

# Terminal 3 — referee: open a dual-external game (no seat tokens shared)
XIANGQI_API_BASE=http://127.0.0.1:8000 \
  python scripts/mcp_llm_duel_smoke.py --api-base http://127.0.0.1:8000
```

The referee writes `game_id` plus `red-prompt.txt` / `black-prompt.txt`. Each side calls `claim_seat` independently. `LITELLM_API_BASE` / `LITELLM_API_KEY` are unused on this path.

### Tests

```bash
python -m unittest discover -s tests -v
```

## Layout

```text
llm-xiangqi-arena/
├── server.py / mcp_server.py / mcp_contract.py
├── xiangqi.py / cycle_rules.py
├── llm_client.py / adapters/
├── agent/                     # LangGraph player, memory, observe tools
├── prompts/player/            # en default, zh fallback, MCP brief
├── prompts/agent/             # compress / turn summary
├── docs/mcp-agent-contract.md
├── SCOPE.md                   # product surface
├── config.example.yaml
├── .env.example
└── assets/xiangqi-design/     # one CC0 board + piece set
```

## Acknowledgements

- [Pikafish](https://github.com/official-pikafish/Pikafish) — open-source Xiangqi engine
- [cchess](https://github.com/walker8088/cchess) — notation reference
- [xiangqi-setup](https://github.com/hartwork/xiangqi-setup) — CC0 board/piece themes (`minimal_chinese` + `retro_simple`)

## License

MIT. The optional board/piece SVGs under `assets/xiangqi-design/cc0/` are [CC0-1.0](assets/xiangqi-design/README.md).
