#!/bin/bash
# 臂 B+ELICIT+SEQ：REPLAN + L-CLOSE(常开) + L-GROUND + L-ELICIT + L-SEQ（无 SHOTGUN）。
# 目的：修 overdue_bill_suspension。两层都治齐：
#   L-ELICIT 填 pay_allowed（第一层：授权藏在 task_instructions，schema 恒 null）；
#   L-SEQ   让 makePayment 的 check_payment_request → make_payment 真正按序跑完
#           （第二层：check 的 when 不翻 → _next_action 循环选 check 选不到 make_payment，
#            且 check 输出被幻觉解析成 bill_paid=False → 子任务提前 done）。
# 预期：overdue 0/4 → 4/4，pass^1 0.925→~0.95、pass^4 0.85→~0.90；其余任务零回归
#       （L-SEQ 对单动作子任务、FIX_* 数学 inert：done 字段只由末位 retest 产出）。
# 对照 = user_replan_ground_elicit_full（只差 SEQ）+ user_replan_ground_full（臂 B）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_elicit_seq_full
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

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE+GROUND+ELICIT+SEQ: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] FULL REPLAN+CLOSE+GROUND+ELICIT+SEQ done rc=$?"
echo "[hint] elicit 事件： grep -c '\"kind\": \"elicit\"' data/simulations/$SAVE/state_log.jsonl"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE data/simulations/user_replan_ground_elicit_full"
