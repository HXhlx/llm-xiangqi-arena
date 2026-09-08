# Scope

This repository is the **duel core**: a public, inspectable Xiangqi match server with in-process LLM/AI seats and a seat-gated MCP surface for external agents.

## Product surface

| Area | What ships |
|------|------------|
| Xiangqi rules | `xiangqi.py`, `cycle_rules.py` |
| Match server | FastAPI `server.py` — human / random / AI / LLM / Pikafish / external seats |
| LangGraph orchestration | `agent/player.py`, `agent/graph.py`, Redis checkpointer, compress/trim |
| MCP tool gating | `mcp_server.py`, `mcp_contract.py`, `external_mcp.py`, seat tokens |
| LiteLLM client (optional) | `llm_client.py` + vendor `adapters/` — AI-player type only; not required for MCP external seats |
| Prompts | English player/agent YAML; optional `XIANGQI_LANG=zh` fallbacks |
| Config templates | `config.example.yaml`, `.env.example` (placeholders only) |
| Tests | 359 unit tests — duel / MCP / prompt / observe / LangGraph (`python -m unittest discover -s tests -q`) |
| Board art | **One** CC0 set: `minimal_chinese` board + `retro_simple` pieces |

LangGraph and MCP are complementary surfaces. LiteLLM is unused on the dual-MCP external path.

## Secrets policy

- Never commit `.env`, `config.yaml`, cookies, PEMs, or credential JSON
- Examples use `sk-xxxxxxxx` / `YOUR_PASSWORD` only

## Locale

```bash
unset XIANGQI_LANG          # English default
# export XIANGQI_LANG=zh    # Chinese player prompts + MCP briefs
```

Piece-face CJK and ICCS/`played_zh` move names are artwork / notation, not product copy.
