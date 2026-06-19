#!/usr/bin/env bash
# Run schema_user_agent (NON-solo dialogue) + build the HTML report.
#
#   scripts/run_user.sh <run-name> [extra tau2 run args...]
#
# Agent  = local qwen3-8b (vLLM), routed via AGENT_BASE (default port 9000) so it
#          does NOT contend with the solo run on 8000.
# User   = strong API model (gpt-5.4) via micuapi (OPENAI_API_BASE in .env).
# Specs  = memory_user/ (isolated copy; solo's memory/ untouched).
#
# Env overrides: AGENT_LLM, AGENT_BASE, USER_LLM, NUM_TRIALS, MAX_STEPS, TASK_SET, NUM_TASKS.
set -eu
cd "$(dirname "$0")/.."

RUN="${1:?usage: scripts/run_user.sh <run-name> [extra tau2 run args...]}"; shift || true
DIR="data/simulations/$RUN"; mkdir -p "$DIR"

AGENT_LLM="${AGENT_LLM:-openai/qwen3-8b}"
AGENT_BASE="${AGENT_BASE:-http://127.0.0.1:9000/v1}"
USER_LLM="${USER_LLM:-openai/gpt-5.4}"
NUM_TRIALS="${NUM_TRIALS:-1}"
MAX_STEPS="${MAX_STEPS:-40}"
TASK_SET="${TASK_SET:-telecom_small}"

# Agent talks to AGENT_BASE via per-call api_base; user-sim uses the global
# OPENAI_API_BASE (micuapi) from .env. Key is the real micuapi key (vLLM ignores it).
AGENT_ARGS="{\"temperature\":0.0,\"api_base\":\"$AGENT_BASE\"}"

common=(--domain telecom --agent schema_user_agent --user user_simulator
        --agent-llm "$AGENT_LLM" --agent-llm-args "$AGENT_ARGS"
        --user-llm "$USER_LLM"
        --task-set-name "$TASK_SET"
        --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS" --save-to "$RUN")

echo n | SCHEMAFLEX_SPEC_DIR=memory_user \
         SCHEMAFLEX_TOKEN_LOG="$DIR/tokens.json" \
         SCHEMAFLEX_STATE_LOG="$DIR/state_log.jsonl" \
  uv run --env-file .env tau2 run "${common[@]}" "$@"

echo '{"queries":0,"hits":0}' > "$DIR/cache.json"
python3 scripts/report.py "$DIR" || true
echo "→ $DIR  (report.html, state_log.jsonl)"
