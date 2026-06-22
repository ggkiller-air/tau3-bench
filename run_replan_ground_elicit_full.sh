#!/bin/bash
# 臂 B+ELICIT：REPLAN + L-CLOSE(常开) + L-GROUND + L-ELICIT（无 SHOTGUN / 无 SEQ）。
# 目的：在新最佳栈 B 上单独隔离 ELICIT 的净效应（此前 ELICIT 只在 B+SHOTGUN 上测过）。
# 预期：overdue 仍 0/4（pay_allowed 被填上，但 makePayment 第二层死锁未修），其余任务中性。
# 对照 = user_replan_ground_full（臂 B，已有数据）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_elicit_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_ELICIT=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE+GROUND+ELICIT: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] FULL REPLAN+CLOSE+GROUND+ELICIT done rc=$?"
echo "[hint] elicit 事件： grep -c '\"kind\": \"elicit\"' data/simulations/$SAVE/state_log.jsonl"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE data/simulations/user_replan_ground_full"
