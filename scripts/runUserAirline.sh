#!/usr/bin/env bash
# Run the unified schema_user_agent on the AIRLINE domain (dialogue) + HTML report, one line.
# Mirror of runUser.sh but pinned to --domain airline + the airline schema dir.
#
#   scripts/runUserAirline.sh <run-name> [extra tau2 run args...]
#
# Iterate on the 20-task test split, report on all 50:
#   scripts/runUserAirline.sh air_test  --task-split-name test
#   scripts/runUserAirline.sh air_full                                  # all 50
#   scripts/runUserAirline.sh air_smoke --task-split-name test --num-tasks 3
#
# Two LLMs: agent = local vLLM 8B (:9000); user-sim = gpt-5.4 via .env proxy.
# WATCHDOG defaults ON here (dialogue runs must be time-bounded — see project notes).
# Overridable via env: AGENT_LLM, USER_LLM, VLLM_BASE, MAX_STEPS, NUM_TRIALS, ENV_FILE,
#   SPEC_DIR, and any SCHEMAFLEX_* lever.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: scripts/runUserAirline.sh <run-name> [extra tau2 run args...]}"; shift || true

DIR="data/simulations/$RUN"
rm -rf "$DIR"; mkdir -p "$DIR"

AGENT="${AGENT:-schema_user_agent}"
TAU_USER="${TAU_USER:-user_simulator}"
AGENT_LLM="${AGENT_LLM:-openai/qwen3-8b}"
USER_LLM="${USER_LLM:-openai/gpt-5.4}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:9000/v1}"
MAX_STEPS="${MAX_STEPS:-50}"
NUM_TRIALS="${NUM_TRIALS:-1}"
TEMP="${TEMP:-0.0}"
ENV_FILE="${ENV_FILE:-.env}"
SPEC_DIR="${SPEC_DIR:-memory_airline}"

# Local vLLM + remote proxy: don't route either through a corp proxy.
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

# Airline schema specs + side-call logs (StateUpdater/MacroInit never land in results.json).
export SCHEMAFLEX_SPEC_DIR="$SPEC_DIR"
export SCHEMAFLEX_TOKEN_LOG="$DIR/tokens.json"
export SCHEMAFLEX_STATE_LOG="$DIR/state_log.jsonl"
export SCHEMAFLEX_TOKEN_TRACE="$DIR/token_trace.jsonl"
# Time-bound dialogue by default (project rule: never run dialogue without a watchdog).
export SCHEMAFLEX_WATCHDOG="${SCHEMAFLEX_WATCHDOG:-1}"

echo "[$(date '+%H:%M:%S')] runUserAirline $RUN | agent=$AGENT($AGENT_LLM@$VLLM_BASE) user=$TAU_USER($USER_LLM) spec=$SPEC_DIR"
echo "  levers: WATCHDOG=${SCHEMAFLEX_WATCHDOG} REPLAN=${SCHEMAFLEX_REPLAN:-0} ELICIT=${SCHEMAFLEX_ELICIT:-0} SEQ=${SCHEMAFLEX_SEQ:-0} | trials=$NUM_TRIALS max-steps=$MAX_STEPS"

METRICS="${VLLM_METRICS:-${VLLM_BASE%/v1}/metrics}"
snap(){ curl -s "$METRICS" 2>/dev/null | awk '/gpu_prefix_cache_queries_total/{q=$2}/gpu_prefix_cache_hits_total/{h=$2}END{print q+0,h+0}'; }
read q0 h0 < <(snap) || { q0=0; h0=0; }

uv run --env-file "$ENV_FILE" tau2 run \
  --domain airline --agent "$AGENT" --user "$TAU_USER" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "{\"temperature\":$TEMP,\"api_base\":\"$VLLM_BASE\"}" \
  --user-llm "$USER_LLM" \
  --num-trials "$NUM_TRIALS" --max-steps "$MAX_STEPS" \
  --save-to "$RUN" "$@"
rc=$?

read q1 h1 < <(snap) || { q1=0; h1=0; }
python3 -c "import json;json.dump({'queries':$q1-$q0,'hits':$h1-$h0},open('$DIR/cache.json','w'))" 2>/dev/null || true

echo "[$(date '+%H:%M:%S')] done rc=$rc"
echo "[hint] infra errors should be ~0:  grep -c infrastructure_error $DIR/results.json"
python3 scripts/report.py "$DIR" 2>/dev/null && echo "→ $DIR/report.html" \
  || echo "[hint] report skipped"
