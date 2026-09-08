# AGENTS.md — LLM Xiangqi Arena

English-first duel core.

## Locale

- Default: English player prompts (`prompts/player/en.yaml`)
- `XIANGQI_LANG=zh` loads `*.zh.yaml` / `mcp_external.zh.md`
- Piece-face CJK is artwork, not locale
- Threat/root labels from `observe` (`有根` / `无根` / `「无明显威胁」`) stay Chinese even under the EN default — they match `prompts/player/tools.yaml` and `agent/observe.py`

## Match loop

```text
game.log ← FastAPI server.py
 ├─ LangGraph per-side thread (Redis Stack) # in-process LLM/AI seats
 └─ MCP seat token → submit_move            # parallel external surface
```

LangGraph and MCP are complementary, not a linear pipeline. LangGraph = in-process per-seat orchestration; MCP = external seat-gated tool surface.

- One work-dir / log tree per game under `logs/` (gitignored)
- External agents: `claim_seat` then one `submit_move` per ply
- No engine tools on the MCP surface
- LiteLLM is optional and only used by the built-in AI-player type

## Commands

```bash
cp .env.example .env && cp config.example.yaml config.yaml
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
python mcp_server.py --http 8765
python -m unittest discover -s tests -q
```

## Do not

- Commit `.env`, `config.yaml`, cookies, or real API keys
- Push secrets “for convenience”
- Expand beyond the duel core (rules, match server, LangGraph seats, MCP gate)
