# Scope

This repository is the **duel core**: a public, inspectable Xiangqi match server with in-process LLM/AI seats and a seat-gated MCP surface for external agents.

## Product surface

| Area | What ships |
|------|------------|
| Xiangqi rules | `xiangqi.py`, `cycle_rules.py` |
| Match server | FastAPI `server.py` — random / AI (LiteLLM) / LLM / external seats. Pikafish is optional analysis, not a seat. There is no human seat. |
| LangGraph orchestration | `agent/player.py`, `agent/graph.py`, Redis checkpointer, compress/trim |
| MCP tool gating | `mcp_server.py`, `mcp_contract.py`, `external_mcp.py`, seat tokens |
| LiteLLM client (optional) | `type: ai` calls one OpenAI-compatible LiteLLM proxy (`LITELLM_API_BASE` / `LITELLM_API_KEY`). The model name is `LITELLM_MODEL` or a preset model; routing happens in the proxy. Optional `LITELLM_ALLOWED_MODELS` limits names that can spend the shared key. Not required for MCP external seats. Vendor `adapters/` shape OpenAI-compatible requests. |
| Prompts | English player/agent YAML; optional `XIANGQI_LANG=zh` fallbacks |
| Config templates | `config.example.yaml`, `.env.example` (placeholders only) |
| Tests | Unit tests for the duel, MCP, prompts, observe, and LangGraph (`python -m unittest discover -s tests -q`), with `XIANGQI_LANG` unset and with `XIANGQI_LANG=zh` |
| Board art | **One** CC0 set: `minimal_chinese` board + `retro_simple` pieces |

LangGraph and MCP are complementary surfaces. LiteLLM is unused on the dual-MCP external path.

## Secrets policy

- Never commit `.env`, `config.yaml`, cookies, PEMs, or credential JSON
- Examples use `sk-xxxxxxxx` / `YOUR_PASSWORD` only

## Locale

```bash
unset XIANGQI_LANG          # English default
# export XIANGQI_LANG=zh    # Chinese player prompts + MCP briefs
# Unknown values (fr, zh-TW, …) log one warning per process and fall back to en.
```

Piece-face CJK and ICCS/`played_zh` move names are artwork / notation, not product copy.
