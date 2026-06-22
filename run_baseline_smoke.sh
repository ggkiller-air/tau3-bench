#!/bin/bash
# Baseline 冒烟：telecom user 模式，3 任务，1 trial。agent=本地 qwen3-8b@9000，user-sim=gpt-5.4@micuapi。
set -uo pipefail
cd /mnt/ssd2/xll/tau3-bench

# 1) 加载 micuapi key/base（给 user-sim gpt-5.4 用）
set -a; . .env; set +a

# 2) 代理全绕：localhost(agent) 和 micuapi(user) 都直连，避开未装 socksio 的 SOCKS 代理
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

# 3) SchemaFlex 日志（baseline：两个 lever 默认全关，不设 SCHEMAFLEX_SAGE / SCHEMAFLEX_SUPERVISOR）
mkdir -p data/simulations/user_base
export SCHEMAFLEX_SPEC_DIR=memory   # CLAUDE.md §9 写的 memory_user 是笔误，实际目录是 ./memory
export SCHEMAFLEX_STATE_LOG=data/simulations/user_base/state_log.jsonl
export SCHEMAFLEX_TOKEN_LOG=data/simulations/user_base/tokens.json
export SCHEMAFLEX_TOKEN_TRACE="data/simulations/$SAVE/token_trace.jsonl"

# 冒烟可重复跑：清掉上一次 user_base 结果，避免 try_resume 卡 FileExistsError
rm -rf data/simulations/user_base
mkdir -p data/simulations/user_base

echo "[$(date '+%H:%M:%S')] start baseline smoke (3 tasks)"
conda run --no-capture-output -n tau3bench tau2 run \
  --domain telecom --agent schema_user_agent --user user_simulator \
  --agent-llm openai/qwen3-8b \
  --agent-llm-args '{"temperature":0.0,"api_base":"http://127.0.0.1:9000/v1"}' \
  --user-llm openai/gpt-5.4 \
  --task-set-name telecom_small \
  --num-trials 1 --max-steps 100 --num-tasks 3 \
  --save-to user_base
echo "[$(date '+%H:%M:%S')] smoke done rc=$?"
