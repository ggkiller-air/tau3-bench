#!/bin/bash
# 主 arm：REPLAN + L-CLOSE(常开) + L-GROUND，telecom_small 全 20 任务 × 4 trial。
# L-GROUND：硬化 agent↔user 指令接缝(治 M2)——单动作 + 用 tool/args 精确命名设置，
# 不重排(无 derail 风险)、无额外 LLM 成本(只换 schema_executor 的 prompt)。
# 对照 = user_replan_close_full（REPLAN+CLOSE，GROUND 关）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE+GROUND: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] FULL REPLAN+CLOSE+GROUND done rc=$?"
