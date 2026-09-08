# Scope

This repository is a **public, secret-free showcase** of multi-LLM Xiangqi agents with MCP gating.

It is extracted from the English-first surface of the private research repo `HXhlx/xiangqi-llm-duel` (`cursor/public-english-release-99b0`). That private repo stays private.

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

## Out of scope (intentionally omitted)

These live only in the private research stack. Tests that import them are **skipped / not shipped**.

Shipped unit tests (this tree): **351 passed** (`python -m unittest discover -s tests -q`).

Not shipped (and not run here):

- `tests/test_*postmortem*`, commentary / host / dossier / review
- TTS / subtitle / WhisperX / Remotion / HyperFrames / watermark / publish
- Tournament daemon (`scripts/run_tournament.py`) including 5 `poll_live` cases that used to live in `test_external_player.py`

- `agent/postmortem/` — commentary, dossier, host, review
- Remotion / HyperFrames video backends (`video/`)
- TTS, subtitles, WhisperX, watermarking
- Publish / SEO / sau upload (`xiangqi-publish`, Aliyun, Bilibili, Douyin, Kuaishou)
- Vendor catalogs: `assets/厂商`, `config/characters.json`, `config/role_avatars.json`, `config/panels.json`, `config/published_series.json`
- Music beds, cover fonts, `assets/covers-new`
- Tournament daemon / extra runner ports
- Production cookies, LiteLLM keys, Redis passwords, platform account IDs

## Secrets policy

- Never commit `.env`, `config.yaml`, cookies, PEMs, or credential JSON
- Examples use `sk-xxxxxxxx` / `YOUR_PASSWORD` only
- Making the **private** repo public is not part of this project

## Locale

```bash
unset XIANGQI_LANG          # English default
# export XIANGQI_LANG=zh    # Chinese player prompts + MCP briefs
```

Piece-face CJK and ICCS/`played_zh` move names are artwork / notation, not product copy.
