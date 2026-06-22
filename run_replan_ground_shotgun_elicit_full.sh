#!/bin/bash
# 全量：REPLAN + L-CLOSE + L-GROUND + L-SHOTGUN + L-ELICIT，20 任务 × 4 trials（=80 sims）。
# L-ELICIT：give-up 处，若某 on-chain 修复仅被一个"null 且无人 produce 的上下文前置字段"
# (pay_allowed/is_abroad…)卡住，主动问用户 yes/no 授权→解析填字段→修复解锁链式跑完。
# 靶子 = overdue_bill_suspension（agent 发缴费请求后没问授权就放弃）。
# 对照 = run_replan_ground_shotgun_full.sh（ELICIT 关）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_shotgun_elicit_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_SHOTGUN=1
export SCHEMAFLEX_ELICIT=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE+GROUND+SHOTGUN+ELICIT: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] elicit 事件： grep -c '\"kind\": \"elicit\"' data/simulations/$SAVE/state_log.jsonl"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE data/simulations/user_replan_ground_shotgun_full"
