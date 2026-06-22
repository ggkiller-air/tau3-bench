#!/bin/bash
# pass^1 快跑：REPLAN + L-CLOSE(常开) + L-GROUND + L-SHOTGUN，全 20 任务 × 1 trial（=20 sims）。
# L-SHOTGUN：mms family 一旦基础连通即跳过全部诊断、按固定安全序投机执行 6 修复 + 1 retest，
# 由 L-CLOSE 收尾。靶子 = 3 个 type-b mms 硬骨头(bad_wifi_calling/break_apn_mms_setting/
# bad_network_preference[mms])。对照(同 trial 数) = run_replan_ground_pass1.sh（SHOTGUN 关，0.75）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_shotgun_pass1
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_SHOTGUN=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start pass^1 REPLAN+CLOSE+GROUND+SHOTGUN: 20 tasks x 1 trial"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 1 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] done rc=$?"
echo "[hint] infra 错应≈0： grep -c infrastructure_error data/simulations/$SAVE/results.json"
echo "[hint] 分析： python analyze_arm.py data/simulations/$SAVE data/simulations/user_replan_ground_pass1"
