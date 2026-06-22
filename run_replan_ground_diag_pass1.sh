#!/bin/bash
# pass^1 快跑：REPLAN + L-CLOSE(常开) + L-GROUND + DIAG(诊断先行)，全 20 任务 × 1 trial。
# DIAG：REPLAN tier-4 不再让"投机修复"(precondition 还没被诊断测的修复，如 mms 的
# fixStoragePerm/fixSmsPerm)抢跑——先让相关诊断跑、确认故障后再开修复。治 type(b) mms M1。
# 对照(同 trial 数，DIAG 关) = run_replan_ground_pass1.sh → user_replan_ground_pass1。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_diag_pass1
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_DIAG=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start pass^1 REPLAN+CLOSE+GROUND+DIAG: 20 tasks x 1 trial"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 1 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] 核机制: grep '\"kind\": \"replan\"' data/simulations/$SAVE/state_log.jsonl | grep -o 'checkWifiCalling\\|fixWifiCalling' | sort | uniq -c"
