#!/usr/bin/env bash
# Run schema_user_agent (USER mode) + build the HTML report, in one line.
# Local-relative (uv), replaces the per-experiment run_*.sh that hardcoded /mnt/ssd2/xll.
#
#   scripts/runUser.sh <run-name> [extra tau2 run args...]
#
# Two LLMs: the agent (small, local vLLM) and the user-sim (gpt-5.4 via .env proxy).
#   - user-llm creds/base come from ENV_FILE (.env → OPENAI_API_BASE = the gpt proxy).
#   - agent-llm is pinned to the LOCAL vLLM via --agent-llm-args api_base (VLLM_BASE).
#
# Levers are env switches, inherited by the run (OFF by default = faithful baseline):
#   baseline:    scripts/runUser.sh user_base --task-set-name telecom_small --num-trials 4
#   best stack:  SCHEMAFLEX_REPLAN=1 SCHEMAFLEX_GROUND=1 SCHEMAFLEX_ELICIT=1 SCHEMAFLEX_SEQ=1 \
#                scripts/runUser.sh user_full --task-set-name telecom_small --num-trials 4
#   (CLOSE is always-on; SHOTGUN/SAGE/SUPERVISOR/DIAG are experimental — see tutorial.md §5.)
#
# Overridable via env: AGENT_LLM, USER_LLM, VLLM_BASE, MAX_STEPS, NUM_TRIALS, ENV_FILE,
#   AGENT, TAU_USER, TEMP, and any SCHEMAFLEX_* lever.
set -uo pipefail   # NOT -e: a nonzero tau2 run must still reach the report step
cd "$(dirname "$0")/.."
RUN="${1:?usage: scripts/runUser.sh <run-name> [extra tau2 run args...]}"; shift || true

DIR="data/simulations/$RUN"
rm -rf "$DIR"; mkdir -p "$DIR"

AGENT="${AGENT:-schema_user_agent}"
TAU_USER="${TAU_USER:-user_simulator}"   # NOT USER: that's a shell builtin (=root), would leak in
AGENT_LLM="${AGENT_LLM:-openai/qwen3-8b}"
USER_LLM="${USER_LLM:-openai/gpt-5.4}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:9000/v1}"   # local vLLM serving the agent's small model
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRIALS="${NUM_TRIALS:-1}"     # set 4 for pass^4
TEMP="${TEMP:-0.0}"
ENV_FILE="${ENV_FILE:-.env}"      # provides the user-llm proxy creds/base

# Local vLLM + remote proxy: don't route either through a corp proxy.
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

# Side-call logs (StateUpdater/MacroInit never land in results.json).
export SCHEMAFLEX_TOKEN_LOG="$DIR/tokens.json"
export SCHEMAFLEX_STATE_LOG="$DIR/state_log.jsonl"
export SCHEMAFLEX_TOKEN_TRACE="$DIR/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] runUser $RUN | agent=$AGENT($AGENT_LLM@$VLLM_BASE) user=$TAU_USER($USER_LLM)"
echo "  levers: REPLAN=${SCHEMAFLEX_REPLAN:-0} GROUND=${SCHEMAFLEX_GROUND:-0} ELICIT=${SCHEMAFLEX_ELICIT:-0} SEQ=${SCHEMAFLEX_SEQ:-0}"\
"| trials=$NUM_TRIALS max-steps=$MAX_STEPS"

# Snapshot local vLLM prefix-cache hit delta around the run (best-effort).
METRICS="${VLLM_METRICS:-${VLLM_BASE%/v1}/metrics}"
snap(){ curl -s "$METRICS" 2>/dev/null | awk '/gpu_prefix_cache_queries_total/{q=$2}/gpu_prefix_cache_hits_total/{h=$2}END{print q+0,h+0}'; }
read q0 h0 < <(snap) || { q0=0; h0=0; }

uv run --env-file "$ENV_FILE" tau2 run \
  --domain telecom --agent "$AGENT" --user "$TAU_USER" \
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
  || echo "[hint] report skipped; analyze with: python analyze_arm.py $DIR <baseline_dir>"
