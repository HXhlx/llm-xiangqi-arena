# AGENTS.md — LLM Xiangqi Arena

English-first duel core. Video / TTS / publish live only in the private research repo and are **not** in this tree.

## Locale

- Default: English player prompts (`prompts/player/en.yaml`)
- `XIANGQI_LANG=zh` loads `*.zh.yaml` / `mcp_external.zh.md`
- Piece-face CJK is artwork, not locale

## Match loop

```text
game.log  ←  FastAPI server.py
              LangGraph per-side thread (Redis Stack)
              MCP seat token → submit_move
```

- One work-dir / log tree per game under `logs/` (gitignored)
- External agents: `claim_seat` then one `submit_move` per ply
- No engine tools on the MCP surface
- LangGraph = in-process per-seat orchestration; MCP = external seat-gated tool surface. Complementary, not a substitute or a linear pipeline.

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
- Add Remotion / TTS / subtitle / publish code here
- Push secrets “for convenience”
