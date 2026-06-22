#!/bin/bash
# 冒烟：REPLAN + L-CLOSE(常开) + L-GROUND，3 任务 1 trial。
# L-GROUND 是纯指令措辞改动(无额外 LLM 成本)。冒烟主要看 agent 给用户的指令是否
# 变成「单动作 + 精确命名设置」(而非复合/UI 导航)。跑完肉眼看 transcript 的 assistant 指令。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

set -a; . .env; set +a
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SAVE=user_replan_ground_smoke
rm -rf "data/simulations/$SAVE"
mkdir -p "data/simulations/$SAVE"
export SCHEMAFLEX_SPEC_DIR=memory
export SCHEMAFLEX_REPLAN=1
export SCHEMAFLEX_GROUND=1
export SCHEMAFLEX_STATE_LOG="data/simulations/$SAVE/state_log.jsonl"
export SCHEMAFLEX_TOKEN_LOG="data/simulations/$SAVE/tokens.json"
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

echo "[$(date '+%H:%M:%S')] start REPLAN+CLOSE+GROUND smoke (3 tasks)"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 1 --max-steps 100 --num-tasks 3 \
  --save-to "$SAVE"
echo "[$(date '+%H:%M:%S')] smoke done rc=$?"
echo "[hint] 看指令是否单动作+精确: python dump_sim.py $SAVE <sim_id> | grep -A1 assistant"
