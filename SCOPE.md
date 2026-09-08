# Scope

This repository is a **public, secret-free showcase** of multi-LLM Xiangqi agents with MCP gating.

## In scope

| Area | What ships |
|------|------------|
| Xiangqi rules | `xiangqi.py`, `cycle_rules.py` |
| Match server | FastAPI `server.py` — human / random / AI / LLM / Pikafish / external seats |
| LangGraph orchestration | `agent/player.py`, `agent/graph.py`, Redis checkpointer, compress/trim |
| MCP tool gating | `mcp_server.py`, `mcp_contract.py`, `external_mcp.py`, seat tokens |
| LiteLLM client | `llm_client.py` + vendor `adapters/` |
| Prompts | English player/agent YAML; optional `XIANGQI_LANG=zh` fallbacks |
| Config templates | `config.example.yaml`, `.env.example` (placeholders only) |
| Tests | Duel / MCP / prompt / observe / LangGraph unit tests |
| Board art | **One** CC0 set: `minimal_chinese` board + `retro_simple` pieces |

Shipped unit tests (this tree): **351 passed** (`python -m unittest discover -s tests -q`).

## Secrets policy

- Never commit `.env`, `config.yaml`, cookies, PEMs, or credential JSON
- Examples use `sk-xxxxxxxx` / `YOUR_PASSWORD` only

## Locale

```bash
unset XIANGQI_LANG          # English default
# export XIANGQI_LANG=zh    # Chinese player prompts + MCP briefs
```

Piece-face CJK and ICCS/`played_zh` move names are artwork / notation, not product copy.
