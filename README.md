# LLM Xiangqi Arena

**Large language models play Xiangqi (Chinese chess) against each other — with seat-gated tools, not a hidden engine.**

This is a backend-only arena. FastAPI runs the match. LangGraph orchestrates in-process LLM/AI seats. MCP is a seat-gated tool surface for external agents. LiteLLM is an optional OpenAI-compatible router for the built-in AI-player type only; dual-MCP external seats do not need it.

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
     OpenAI-compatible router              │
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
| **FastAPI** (`server.py`) | Concurrent games: Human / Random / AI / LLM / Pikafish / external |
| **LangGraph** | Per-side threads on Redis Stack; compress/trim the window |
| **MCP** (`mcp_server.py`) | Stateless proxy. Seat token + tool audit. No engine tools. |
| **LiteLLM** (optional) | OpenAI-compatible router for the **AI-player** type only. Dual-MCP external seats do not need `LITELLM_*`. |

External agents must `claim_seat` and `submit_move` one ply at a time.

## What this proves

- **Seat gate.** External agents cannot move without a `seat_token` from `claim_seat`. Bare REST without a token cannot move for that seat.
- **One ply.** Each external seat calls `submit_move` once per turn. Multi-move scripts are contract-forbidden.
- **Complementary surfaces.** LangGraph orchestrates in-process LLM/AI seats. MCP is a parallel gate for external agents. Neither replaces the other.
- **Dual-external without LiteLLM.** Two MCP seats can play each other. `LITELLM_API_BASE` / `LITELLM_API_KEY` are unused on that path.
- **Tests.** 359 unit tests (`python -m unittest discover -s tests -q`) cover rules, MCP, prompts, observe, and LangGraph.

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

Pikafish is optional. Download a CPU-matched binary plus `pikafish.nnue` from [Pikafish releases](https://github.com/official-pikafish/Pikafish/releases) into `pikafish/` (see `pikafish/README.md`). LLM games work without it; engine eval and Pikafish seats need the local files.

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
# or: python server.py
```

- Service: `http://127.0.0.1:8000/`
- OpenAPI: `http://127.0.0.1:8000/docs`

### LLM / AI players

- **AI player** (optional LiteLLM): `LITELLM_API_BASE` + `LITELLM_API_KEY`
- **Custom LLM**: `api_base` / `api_key` / `model` in `config.yaml` → `models:[]`
- Default player prompt is **English** (`prompt_name: en`). Chinese: `prompt_name: zh` or `XIANGQI_LANG=zh`
- Per-side threads via LangGraph `AsyncRedisSaver` + `REDIS_URL`

### MCP (external agents)

```bash
.venv/bin/python mcp_server.py                 # stdio
.venv/bin/python mcp_server.py --http 8765     # http://127.0.0.1:8765/mcp
```

`XIANGQI_API_BASE` defaults to `http://127.0.0.1:8000`. After kickoff each side holds a `seat_token`. Bare REST without a token cannot move for that seat.

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
