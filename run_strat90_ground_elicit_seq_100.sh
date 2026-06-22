#!/bin/bash
# strat90 泛化：新最佳栈 B+ELICIT+SEQ（REPLAN+CLOSE+GROUND+ELICIT+SEQ），max_steps=100，90 任务 × 1 trial。
# 对照 = user_strat90_ground_100（B 栈，pass^1=0.578）/ solo 参考 0.689。
# 预期：ELICIT(填 pay_allowed/is_abroad)+SEQ(makePayment 链) 主要帮 service 家族；
#       mms 6/30 是诊断/复合瓶颈，ELICIT/SEQ 够不到。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_strat90_ground_elicit_seq_100
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_ELICIT=1
export SCHEMAFLEX_SEQ=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start strat90 USER REPLAN+CLOSE+GROUND+ELICIT+SEQ @ max_steps=100: 90 tasks x 1 trial"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_strat90 \
  --num-trials 1 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE data/simulations/user_strat90_ground_100"
