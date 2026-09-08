# LLM Xiangqi Arena

**Large language models play Xiangqi (Chinese chess) against each other — with seat-gated tools, not a hidden engine.**

This is a backend-only arena. FastAPI runs the match. LangGraph keeps each side’s conversation. MCP tools gate external agents so they cannot move without a seat token. LiteLLM routes many model vendors through one OpenAI-compatible proxy.

Chinese characters on the **piece faces** (帅 / 将 / 车 / …) are board artwork and stay. Product copy, player prompts, and MCP briefs default to **English**. Set `XIANGQI_LANG=zh` for the original Chinese player track.

This repository is the **duel core** extracted from a private research stack. Video, TTS, subtitles, and social publish tooling are out of scope — see [SCOPE.md](SCOPE.md).

## Problem

Chess-style LLM arenas exist. A public, inspectable **Xiangqi** duel with real rules, engine-optional analysis, and seat-gated external agents does not. Models sit, think, and submit one ICCS ply at a time. Multi-move scripts, material-score greed, and engine-picked moves are contract-forbidden.

## Architecture

```text
LiteLLM / vendor API
        │
        ▼
 FastAPI match server (server.py)
        │  LangGraph per-side thread (Redis Stack)
        │  Xiangqi rules + cycle detection
        ▼
 game.log + snapshots
        │
        ▼
 MCP proxy (mcp_server.py)
   claim_seat → wait_my_turn → preview* → submit_move
```

| Layer | What it does |
|-------|----------------|
| **Xiangqi rules** (`xiangqi.py`, `cycle_rules.py`) | Legal moves, checks, repetitions |
| **FastAPI** (`server.py`) | Concurrent games: Human / Random / AI / LLM / Pikafish / external |
| **LangGraph** | Per-side threads on Redis Stack; compress/trim the window |
| **MCP** (`mcp_server.py`) | Stateless proxy. Seat token + tool audit. No engine tools. |
| **LiteLLM** | One `LITELLM_API_BASE` / `LITELLM_API_KEY` for the AI-player type |

LangGraph is in-process per-seat orchestration for LLM/AI seats. MCP is a parallel, seat-gated tool surface for external agents. They are complementary, not a substitute or a linear pipeline.

External agents must `claim_seat` and `submit_move` one ply at a time.

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

- **AI player**: `LITELLM_API_BASE` + `LITELLM_API_KEY`
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

### Tests

```bash
python -m unittest discover -s tests -v
```

Video / postmortem / publish tests from the private research repo are **not shipped** here.

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
