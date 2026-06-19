#!/usr/bin/env bash
# Run schema_solo_agent + build the HTML report, in one line.
#
#   scripts/run.sh <run-name> [extra tau2 run args...]
#
# Local vLLM (default): records tokens + snapshots prefix-cache hit, then reports.
#   scripts/run.sh qwen3_8b/v1 --task-set-name telecom_full --num-tasks 100 --max-concurrency 3
#
# API proxy (set ENV_FILE → creds from that file, skip local cache snapshot):
#   ENV_FILE=.env AGENT_LLM=openai/gpt-5.4 scripts/run.sh gpt54/v1 --task-set-name telecom_full --num-tasks 30
#
# Defaults: domain telecom, schema_solo_agent, dummy_user, num-trials 1, max-steps 100,
# agent-llm openai/qwen3-8b. Override via env: AGENT_LLM, MAX_STEPS, ENV_FILE, OPENAI_API_BASE.
set -eu
cd "$(dirname "$0")/.."
RUN="${1:?usage: scripts/run.sh <run-name> [extra tau2 run args...]}"; shift || true
DIR="data/simulations/$RUN"; mkdir -p "$DIR"
AGENT_LLM="${AGENT_LLM:-openai/qwen3-8b}"
AGENT="${AGENT:-schema_solo_agent}"   # override e.g. AGENT=llm_agent_solo for the bare-solo baseline
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRIALS="${NUM_TRIALS:-1}"   # set 4 for pass^4
ENV_FILE="${ENV_FILE:-}"

common=(--domain telecom --agent "$AGENT" --user dummy_user
        --agent-llm "$AGENT_LLM" --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS"
        --save-to "$RUN")

if [ -n "$ENV_FILE" ]; then
  # API-proxy mode: base/key from env-file; no local /metrics → cache n/a.
  echo n | SCHEMAFLEX_TOKEN_LOG="$DIR/tokens.json" \
    uv run --env-file "$ENV_FILE" tau2 run "${common[@]}" "$@"
  echo '{"queries":0,"hits":0}' > "$DIR/cache.json"
else
  # Local vLLM mode: inject base/key + snapshot prefix-cache hit delta.
  BASE="${OPENAI_API_BASE:-http://127.0.0.1:9000/v1}"
  METRICS="${VLLM_METRICS:-${BASE%/v1}/metrics}"
  snap(){ curl -s "$METRICS" 2>/dev/null | awk '/gpu_prefix_cache_queries_total/{q=$2}/gpu_prefix_cache_hits_total/{h=$2}END{print q+0,h+0}'; }
  read q0 h0 < <(snap)
  OPENAI_API_BASE="$BASE" OPENAI_BASE_URL="$BASE" OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}" \
  SCHEMAFLEX_TOKEN_LOG="$DIR/tokens.json" \
    uv run tau2 run "${common[@]}" "$@"
  read q1 h1 < <(snap)
  python3 -c "import json;json.dump({'queries':$q1-$q0,'hits':$h1-$h0},open('$DIR/cache.json','w'))"
fi

python3 scripts/report.py "$DIR"
echo "→ $DIR/report.html"
