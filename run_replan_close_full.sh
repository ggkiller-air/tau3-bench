#!/bin/bash
# 主 arm：REPLAN + L-CLOSE，telecom_small 全 20 任务 × 4 trial（pass^4）。
# L-CLOSE 现为码内常开（无开关）：family success_when_all 满足即置 resolved 主动收尾，
# 救"已解决却撞 max_steps 被评测器硬归零"的假阴性。无 gpt-5.4 额外成本（纯确定性收尾）。
# 对照 = committed 的旧 REPLAN arm（user_replan_full, pass^1=0.450，那是 close 改动之前跑的）。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_close_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+CLOSE: 20 tasks x 4 trials"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] FULL REPLAN+CLOSE done rc=$?"
