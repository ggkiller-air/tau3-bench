#!/bin/bash
# strat90 泛化基线：REPLAN + L-CLOSE + L-GROUND（SHOTGUN/ELICIT 关），90 任务 × 1 trial = 90 sims。
# strat90 极度复合（29/30 多故障，最多 8 故障同时坏，23/30 含连通/数据/漫游）→ 40 步多数装不下，
# 通过率会低，按"故障数分层"解读（单/双能过、5+ 基本不可能）。这是 benchmark 压力测试，非回归。
# 对照 = run_strat90_shotgun.sh（SHOTGUN 开，预期复合 mms 掉点）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_strat90_ground
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
# SHOTGUN / ELICIT 关
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start strat90 REPLAN+CLOSE+GROUND (no SHOTGUN): 90 tasks x 1 trial"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_strat90 \
  --num-trials 1 --max-steps 40 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] infra 错： grep -c infrastructure_error data/simulations/$SAVE/results.json"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE"
