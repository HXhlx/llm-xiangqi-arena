#!/usr/bin/env bash
# Bootstrap a local duel-core checkout (no secrets).
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Wrote .env from .env.example — set REDIS_URL and (optionally) LITELLM_*."
fi
if [[ ! -f config.yaml ]]; then
  cp config.example.yaml config.yaml
  echo "Wrote config.yaml from config.example.yaml."
fi

echo "Next:"
echo "  docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest"
echo "  source .venv/bin/activate"
echo "  uvicorn server:app --host 127.0.0.1 --port 8000"
