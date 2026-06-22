#!/bin/bash
# 主 arm：REPLAN + SUPERVISOR 叠加，telecom_small 全 20 任务 × 4 trial（pass^4）。
# 对照committed 的 REPLAN arm（user_replan_full, 0.450）看 supervisor 在重排之上还能不能救 B 桶。
# agent=本地 qwen3-8b@9000，user-sim & supervisor=gpt-5.4@micuapi。
# 注意：supervisor 用 gpt-5.4，每 episode ≤ MAX 次调用 → 80 sims 最多 80*MAX 次强模型调用，留意 micuapi 花销。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_sup_full
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_SUPERVISOR=1
export SCHEMAFLEX_SUPERVISOR_MAX="${SCHEMAFLEX_SUPERVISOR_MAX:-4}"   # 可外部覆盖做预算 A/B
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start FULL REPLAN+SUPERVISOR: 20 tasks x 4 trials (MAX=$SCHEMAFLEX_SUPERVISOR_MAX)"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 4 --max-steps 100 --max-concurrency 8 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] FULL REPLAN+SUPERVISOR done rc=$?"
