#!/bin/bash
# 全量坐实：REPLAN + L-CLOSE(常开) + L-GROUND + L-SHOTGUN，20 任务 × 4 trials（=80 sims）。
# 目的：用 num-trials 4 控掉 gpt-5.4 user-sim 的跑间方差，给稳定 pass^1 + 真正的 pass^k，
# 坐实 1-trial 那次的 pass^1=0.90（mms 6/6）。对照 = run_replan_ground_full.sh（SHOTGUN 关，pass^1≈0.61/4t）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_shotgun_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_SHOTGUN=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE+GROUND+SHOTGUN: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] infra 错应≈0： grep -c infrastructure_error data/simulations/$SAVE/results.json"
echo "[hint] 分析(pass^k/家族)： python analyze_arm.py data/simulations/$SAVE data/simulations/user_replan_ground_full"
