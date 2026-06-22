#!/bin/bash
# 冒烟：REPLAN + SUPERVISOR 叠加，3 任务 1 trial，先验证 supervisor 通路能开火、gpt-5.4 能连上 micuapi。
# agent=本地 qwen3-8b@9000，user-sim & supervisor=gpt-5.4@micuapi。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_sup_smoke
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_SUPERVISOR=1
export SCHEMAFLEX_SUPERVISOR_MAX="${SCHEMAFLEX_SUPERVISOR_MAX:-4}"   # 可外部覆盖做预算 A/B
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start REPLAN+SUPERVISOR smoke (3 tasks, MAX=$SCHEMAFLEX_SUPERVISOR_MAX)"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 1 --max-steps 100 --num-tasks 3 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] smoke done rc=$?"
echo "[hint] 看 supervisor 是否开火： grep -c '\"kind\": \"supervise\"' data/simulations/$SAVE/state_log.jsonl"
